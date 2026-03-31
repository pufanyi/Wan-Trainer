"""Wan2.2 I2V model wrapper for training.

Wraps frozen (text_encoder, vae) and trainable (transformer, transformer_2)
components from the WanImageToVideoPipeline.
"""

import html
from dataclasses import dataclass

import ftfy
import regex as re
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.models import WanTransformer3DModel
from peft import LoraConfig
from transformers import UMT5EncoderModel


@dataclass
class LoRATrainConfig:
    """LoRA configuration for Wan I2V transformers."""

    rank: int = 16
    lora_alpha: int = 16
    target_modules: list[str] | None = None
    lora_dropout: float = 0.0

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["to_q", "to_k", "to_v", "to_out.0"]


def _clean_prompt(text: str) -> str:
    """Replicate the pipeline's prompt_clean: ftfy + html unescape + whitespace."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


class WanI2VForTraining:
    """Wan2.2 I2V model for flow-matching training.

    Frozen: text_encoder, vae.
    Trainable: transformer (high-noise expert), transformer_2 (low-noise expert).
    """

    def __init__(self, model_path: str, lora_config: LoRATrainConfig | None = None):
        pipe = WanImageToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)

        # ---- Frozen components ----
        self.tokenizer = pipe.tokenizer
        self.text_encoder: UMT5EncoderModel = pipe.text_encoder
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        self.vae: AutoencoderKLWan = pipe.vae
        self.vae.requires_grad_(False)
        self.vae.eval()

        # ---- Transformers ----
        self.transformer: WanTransformer3DModel = pipe.transformer
        self.transformer_2: WanTransformer3DModel = pipe.transformer_2

        # ---- LoRA or full fine-tuning ----
        self.lora_config = lora_config
        if lora_config is not None:
            peft_config = LoraConfig(
                r=lora_config.rank,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
            )
            self.transformer.add_adapter(peft_config)
            self.transformer_2.add_adapter(peft_config)
            # Freeze base weights; only LoRA params are trainable
            self.transformer.requires_grad_(False)
            self.transformer_2.requires_grad_(False)
            for m in [self.transformer, self.transformer_2]:
                for name, param in m.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

        self.transformer.train()
        self.transformer_2.train()
        self.transformer.enable_gradient_checkpointing()
        self.transformer_2.enable_gradient_checkpointing()

        # ---- VAE normalization constants ----
        vae_cfg = self.vae.config
        self.latents_mean = torch.tensor(vae_cfg.latents_mean).view(1, vae_cfg.z_dim, 1, 1, 1)
        self.latents_std_inv = (1.0 / torch.tensor(vae_cfg.latents_std)).view(1, vae_cfg.z_dim, 1, 1, 1)

        # ---- Scale factors (from VAE config, not hardcoded) ----
        self.vae_scale_factor_spatial: int = vae_cfg.scale_factor_spatial
        self.vae_scale_factor_temporal: int = vae_cfg.scale_factor_temporal

        # ---- MoE config ----
        self.num_train_timesteps = 1000
        boundary_ratio = pipe.config.boundary_ratio
        self.boundary_timestep = int(boundary_ratio * self.num_train_timesteps)  # 900

        del pipe

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return a list (not generator) of all trainable parameters."""
        return [p for p in self.transformer.parameters() if p.requires_grad] + [
            p for p in self.transformer_2.parameters() if p.requires_grad
        ]

    def save_lora(self, path: str):
        """Save LoRA adapter weights for both transformers."""
        from pathlib import Path

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(out / "transformer")
        self.transformer_2.save_pretrained(out / "transformer_2")

    # ------------------------------------------------------------------
    # Encoding helpers (all run under torch.no_grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_text(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        """Encode prompts to text embeddings. Returns (B, 512, text_dim)."""
        max_length = 512
        prompts = [_clean_prompt(p) for p in prompts]

        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        mask = tokens.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        embeds = self.text_encoder(input_ids, mask).last_hidden_state
        # Zero-pad to max_length (replicate pipeline logic)
        embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
        embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_length - u.size(0), u.size(1))]) for u in embeds],
            dim=0,
        )
        return embeds.to(torch.bfloat16)

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode pixel video to normalized latents.

        Args:
            video: (B, C, T, H, W) in [-1, 1].

        Returns:
            (B, z_dim, T', H', W') normalized latents.
        """
        latents = self.vae.encode(video.to(self.vae.dtype)).latent_dist.sample()
        mean = self.latents_mean.to(latents.device, latents.dtype)
        std_inv = self.latents_std_inv.to(latents.device, latents.dtype)
        return ((latents - mean) * std_inv).to(torch.bfloat16)

    @torch.no_grad()
    def prepare_condition(
        self,
        image: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Build the channel-concatenated condition tensor [mask, cond_latents].

        Replicates the WanImageToVideoPipeline.prepare_latents condition path:
        - Encodes [first_frame, zeros...] with VAE (mode, not sample)
        - Normalizes with latents_mean / latents_std
        - Constructs first-frame mask with temporal expansion

        Args:
            image: (B, C, H, W) first frame in [-1, 1].
            num_frames: Number of pixel-space frames (e.g. 81).
            height: Pixel height.
            width: Pixel width.

        Returns:
            (B, 4 + z_dim, T', H', W') condition tensor.
        """
        B = image.shape[0]
        latent_h = height // self.vae_scale_factor_spatial
        latent_w = width // self.vae_scale_factor_spatial

        # ---- Encode condition video: [first_frame, zeros...] ----
        cond_video = torch.zeros(B, 3, num_frames, height, width, device=image.device, dtype=image.dtype)
        cond_video[:, :, 0] = image
        # Use mode() (argmax) like the pipeline does for condition
        cond_latents = self.vae.encode(cond_video.to(self.vae.dtype)).latent_dist.mode()
        mean = self.latents_mean.to(cond_latents.device, cond_latents.dtype)
        std_inv = self.latents_std_inv.to(cond_latents.device, cond_latents.dtype)
        cond_latents = ((cond_latents - mean) * std_inv).to(torch.bfloat16)

        # ---- Construct mask (replicate pipeline_wan_i2v.py L468-481) ----
        # Start with pixel-frame-level mask: 1 at frame 0, 0 elsewhere
        mask = torch.ones(B, 1, num_frames, latent_h, latent_w, device=image.device)
        mask[:, :, 1:] = 0

        # Expand first frame across vae_scale_factor_temporal positions
        first_frame_mask = mask[:, :, 0:1]
        first_frame_mask = first_frame_mask.repeat(1, 1, self.vae_scale_factor_temporal, 1, 1)
        mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)
        # Reshape to latent temporal grid: (B, T', vae_temporal, H', W') -> transpose -> (B, vae_temporal, T', H', W')
        mask = mask.view(B, -1, self.vae_scale_factor_temporal, latent_h, latent_w).transpose(1, 2)

        return torch.cat([mask.to(cond_latents), cond_latents], dim=1)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        video_latents: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flow-matching loss for one training step.

        Flow matching formulation:
            noisy = sigma * noise + (1 - sigma) * x0
            target = noise - x0  (velocity)
            loss = MSE(model_pred, target)

        MoE routing:
            timestep >= boundary (900) -> transformer  (high-noise expert)
            timestep <  boundary (900) -> transformer_2 (low-noise expert)

        Args:
            video_latents: (B, z_dim, T', H', W') normalized latents.
            condition: (B, 4 + z_dim, T', H', W') from prepare_condition.
            prompt_embeds: (B, 512, text_dim).

        Returns:
            Scalar loss.
        """
        B = video_latents.shape[0]
        device = video_latents.device

        # Sample random timesteps; sigma = t / 1000
        timesteps = torch.randint(0, self.num_train_timesteps, (B,), device=device)
        sigmas = (timesteps.float() / self.num_train_timesteps).view(B, 1, 1, 1, 1)

        # Flow matching: noisy = sigma * noise + (1 - sigma) * x0
        noise = torch.randn_like(video_latents)
        noisy_latents = sigmas * noise + (1.0 - sigmas) * video_latents

        # Target velocity: v = noise - x0
        target = noise - video_latents

        # Model input: [noisy_latents, condition] along channel dim -> 36 channels
        model_input = torch.cat([noisy_latents, condition], dim=1)

        # Route to the correct MoE expert
        high_mask = timesteps >= self.boundary_timestep
        low_mask = ~high_mask

        loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        count = 0

        for mask, transformer in [
            (high_mask, self.transformer),
            (low_mask, self.transformer_2),
        ]:
            if not mask.any():
                continue
            pred = transformer(
                hidden_states=model_input[mask],
                timestep=timesteps[mask],
                encoder_hidden_states=prompt_embeds[mask],
                return_dict=False,
            )[0]
            loss = loss + F.mse_loss(pred.float(), target[mask].float(), reduction="sum")
            count += mask.sum().item() * target[0].numel()

        return loss / count if count > 0 else loss
