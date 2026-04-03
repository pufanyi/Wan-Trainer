"""Wan2.2 I2V model wrapper for training.

Wraps frozen (text_encoder, vae) and trainable (transformer, transformer_2)
components from the WanImageToVideoPipeline.
"""

import html
import json
import math
from pathlib import Path

import ftfy
import regex as re
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.models import WanTransformer3DModel
from loguru import logger
from peft import LoraConfig
from pydantic import BaseModel
from transformers import AutoTokenizer, UMT5EncoderModel


class LoRATrainConfig(BaseModel):
    """LoRA configuration for Wan I2V transformers."""

    rank: int = 16
    lora_alpha: int = 16
    target_modules: list[str] = ["to_q", "to_k", "to_v", "to_out.0"]
    lora_dropout: float = 0.0


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

    def __init__(
        self,
        model_path: str,
        lora_config: LoRATrainConfig | None = None,
        train_experts: str = "both",
        train_text_encoder: bool = False,
        gradient_checkpointing: bool = True,
    ):
        assert train_experts in ("both", "high", "low"), (
            f"train_experts must be 'both', 'high', or 'low', got '{train_experts}'"
        )
        self.train_experts = train_experts
        self.train_text_encoder = train_text_encoder

        model_dir = Path(model_path)

        # ---- Read pipeline config for boundary_ratio ----
        with open(model_dir / "model_index.json") as f:
            pipe_config = json.load(f)
        self.num_train_timesteps = 1000
        boundary_ratio = pipe_config.get("boundary_ratio", 0.9)
        self.boundary_timestep = int(boundary_ratio * self.num_train_timesteps)  # 900
        self.boundary_idx = int((1.0 - self.boundary_timestep / self.num_train_timesteps) * self.num_train_timesteps)

        # ---- Load components sequentially ----
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        logger.info("Loaded tokenizer")

        # ---- Text encoder ----
        self.text_encoder: UMT5EncoderModel = UMT5EncoderModel.from_pretrained(
            model_dir / "text_encoder", torch_dtype=torch.bfloat16
        )
        logger.info("Loaded text_encoder")
        if train_text_encoder:
            self.text_encoder.train()
            if gradient_checkpointing:
                self.text_encoder.gradient_checkpointing_enable()
        else:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

        # ---- VAE (always frozen) ----
        self.vae: AutoencoderKLWan = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.bfloat16)
        logger.info("Loaded vae")
        self.vae.requires_grad_(False)
        self.vae.eval()

        # ---- Transformers (only load what we need) ----
        self.transformer: WanTransformer3DModel | None = None
        self.transformer_2: WanTransformer3DModel | None = None
        if train_experts in ("both", "high"):
            self.transformer = WanTransformer3DModel.from_pretrained(
                model_dir / "transformer", torch_dtype=torch.bfloat16
            )
            logger.info("Loaded transformer")
        if train_experts in ("both", "low"):
            self.transformer_2 = WanTransformer3DModel.from_pretrained(
                model_dir / "transformer_2", torch_dtype=torch.bfloat16
            )
            logger.info("Loaded transformer_2")

        # ---- LoRA or full fine-tuning ----
        self.lora_config = lora_config
        if lora_config is not None:
            peft_config = LoraConfig(
                r=lora_config.rank,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
            )
            for m in [self.transformer, self.transformer_2]:
                if m is None:
                    continue
                m.add_adapter(peft_config)
                m.requires_grad_(False)
                for name, param in m.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

        for m in [self.transformer, self.transformer_2]:
            if m is None:
                continue
            # Ensure uniform dtype for FSDP2 (some params load as float32)
            m.to(torch.bfloat16)
            m.train()
            if gradient_checkpointing:
                m.enable_gradient_checkpointing()

        # ---- VAE normalization constants ----
        vae_cfg = self.vae.config
        self.latents_mean = torch.tensor(vae_cfg.latents_mean).view(1, vae_cfg.z_dim, 1, 1, 1)
        self.latents_std_inv = (1.0 / torch.tensor(vae_cfg.latents_std)).view(1, vae_cfg.z_dim, 1, 1, 1)

        # ---- Scale factors (from VAE config, not hardcoded) ----
        self.vae_scale_factor_spatial: int = vae_cfg.scale_factor_spatial
        self.vae_scale_factor_temporal: int = vae_cfg.scale_factor_temporal

        # ---- Shifted sigma schedule (shift=5, matching DiffSynth) ----
        shift = 5.0
        linear_sigmas = torch.linspace(1.0, 0.0, self.num_train_timesteps + 1)[:-1]
        self.shifted_sigmas = shift * linear_sigmas / (1 + (shift - 1) * linear_sigmas)
        # Derive timesteps from shifted sigmas (for passing to transformer)
        self.shifted_timesteps = (self.shifted_sigmas * self.num_train_timesteps).float()

        # ---- BSMNTW loss weighting (Gaussian centered at t=500) ----
        bsmntw = torch.exp(-2.0 * ((self.shifted_timesteps - 500.0) / 1000.0) ** 2)
        bsmntw = bsmntw - bsmntw.min()
        self.bsmntw = bsmntw * (self.num_train_timesteps / bsmntw.sum())
        self._latent_stat_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._training_buffer_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._condition_mask_cache: dict[tuple[str, int, int, int, torch.dtype], torch.Tensor] = {}

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return a list (not generator) of all trainable parameters."""
        params = []
        if self.train_text_encoder:
            params.extend(p for p in self.text_encoder.parameters() if p.requires_grad)
        for m in [self.transformer, self.transformer_2]:
            if m is not None:
                params.extend(p for p in m.parameters() if p.requires_grad)
        return params

    def save_lora(self, path: str):
        """Save LoRA adapter weights for active transformers."""
        from pathlib import Path

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        if self.transformer is not None:
            self.transformer.save_pretrained(out / "transformer")
        if self.transformer_2 is not None:
            self.transformer_2.save_pretrained(out / "transformer_2")

    def _get_latent_stats(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), str(dtype))
        cached = self._latent_stat_cache.get(key)
        if cached is None:
            cached = (
                self.latents_mean.to(device=device, dtype=dtype),
                self.latents_std_inv.to(device=device, dtype=dtype),
            )
            self._latent_stat_cache[key] = cached
        return cached

    def _get_training_buffers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = str(device)
        cached = self._training_buffer_cache.get(key)
        if cached is None:
            cached = (
                self.shifted_sigmas.to(device=device),
                self.shifted_timesteps.to(device=device),
                self.bsmntw.to(device=device),
            )
            self._training_buffer_cache[key] = cached
        return cached

    def _get_condition_mask_template(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        key = (str(device), num_frames, height, width, dtype)
        cached = self._condition_mask_cache.get(key)
        if cached is not None:
            return cached

        latent_h = height // self.vae_scale_factor_spatial
        latent_w = width // self.vae_scale_factor_spatial
        mask = torch.ones(1, 1, num_frames, latent_h, latent_w, device=device, dtype=dtype)
        mask[:, :, 1:] = 0
        first_frame_mask = mask[:, :, 0:1].repeat(1, 1, self.vae_scale_factor_temporal, 1, 1)
        mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)
        cached = mask.view(1, -1, self.vae_scale_factor_temporal, latent_h, latent_w).transpose(1, 2).contiguous()
        self._condition_mask_cache[key] = cached
        return cached

    # ------------------------------------------------------------------
    # Encoding helpers (all run under torch.no_grad)
    # ------------------------------------------------------------------

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

        with torch.set_grad_enabled(self.train_text_encoder):
            embeds = self.text_encoder(input_ids, mask).last_hidden_state
        embeds = embeds.masked_fill(~mask.bool().unsqueeze(-1), 0)
        return embeds.to(torch.bfloat16)

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode pixel video to normalized latents.

        Args:
            video: (B, C, T, H, W) in [-1, 1].

        Returns:
            (B, z_dim, T', H', W') normalized latents.
        """
        latents = self.vae.encode(video.to(self.vae.dtype)).latent_dist.mode()
        mean, std_inv = self._get_latent_stats(latents.device, latents.dtype)
        return ((latents - mean) * std_inv).to(torch.bfloat16)

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

        # ---- Encode condition video: [first_frame, zeros...] ----
        cond_video = image.new_zeros((B, 3, num_frames, height, width))
        cond_video[:, :, 0] = image
        # Use mode() (argmax) like the pipeline does for condition
        with torch.no_grad():
            cond_latents = self.vae.encode(cond_video.to(self.vae.dtype)).latent_dist.mode()
        mean, std_inv = self._get_latent_stats(cond_latents.device, cond_latents.dtype)
        cond_latents = ((cond_latents - mean) * std_inv).to(torch.bfloat16)

        # ---- Construct mask (replicate pipeline_wan_i2v.py L468-481) ----
        mask = self._get_condition_mask_template(image.device, cond_latents.dtype, num_frames, height, width).expand(
            B, -1, -1, -1, -1
        )

        return torch.cat([mask, cond_latents], dim=1)

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

        Flow matching formulation (shifted sigma schedule, shift=5):
            sigma = 5s / (1 + 4s) where s = linear_sigma
            noisy = sigma * noise + (1 - sigma) * x0
            target = noise - x0  (velocity)
            loss = BSMNTW_weight * MSE(model_pred, target)

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
        shifted_sigmas, shifted_timesteps, bsmntw = self._get_training_buffers(device)

        # Sample random timestep indices, then look up shifted sigma / timestep
        if self.train_experts == "high":
            indices = torch.randint(0, self.boundary_idx, (B,), device=device)
        elif self.train_experts == "low":
            indices = torch.randint(self.boundary_idx, self.num_train_timesteps, (B,), device=device)
        else:
            indices = torch.randint(0, self.num_train_timesteps, (B,), device=device)

        sigmas = shifted_sigmas.index_select(0, indices).view(B, 1, 1, 1, 1)
        timesteps = shifted_timesteps.index_select(0, indices)
        weights = bsmntw.index_select(0, indices)

        # Flow matching: noisy = sigma * noise + (1 - sigma) * x0
        noise = torch.randn_like(video_latents)
        noisy_latents = sigmas * noise + (1.0 - sigmas) * video_latents

        # Target velocity: v = noise - x0
        target = noise - video_latents

        # Model input: [noisy_latents, condition] along channel dim -> 36 channels
        model_input = torch.cat([noisy_latents, condition], dim=1)

        # Route to the correct MoE expert(s)
        experts = []
        if self.transformer is not None:
            experts.append(((timesteps >= self.boundary_timestep).nonzero(as_tuple=False).flatten(), self.transformer))
        if self.transformer_2 is not None:
            experts.append(((timesteps < self.boundary_timestep).nonzero(as_tuple=False).flatten(), self.transformer_2))

        loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_weight = torch.tensor(0.0, device=device, dtype=torch.float32)

        for selected, transformer in experts:
            if selected.numel() == 0:
                continue
            pred = transformer(
                hidden_states=model_input.index_select(0, selected),
                timestep=timesteps.index_select(0, selected),
                encoder_hidden_states=prompt_embeds.index_select(0, selected),
                return_dict=False,
            )[0]
            # Per-sample MSE weighted by BSMNTW
            per_sample_loss = F.mse_loss(pred.float(), target.index_select(0, selected).float(), reduction="none")
            per_sample_loss = per_sample_loss.mean(dim=list(range(1, per_sample_loss.ndim)))
            selected_weights = weights.index_select(0, selected)
            loss = loss + (per_sample_loss * selected_weights).sum()
            total_weight = total_weight + selected_weights.sum()

        return loss / total_weight if total_weight > 0 else loss

    # ------------------------------------------------------------------
    # COS (Chain-of-Search) piecewise flow matching
    # ------------------------------------------------------------------

    def compute_cos_loss(
        self,
        x_final: torch.Tensor,
        x_tau: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        tau_sigma: float = 0.5,
        boundary_noise_std: float = 0.02,
        use_standard_formula: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute piecewise flow-matching loss for COS training.

        Solution A (default): rescaled parameterization ensuring path continuity.
            High stage (sigma >= tau): s_h = (sigma - tau)/(1 - tau)
                x_t = s_h * z + (1 - s_h) * x_tau
                v_target = (z - x_tau) / (1 - tau)
            Low stage (sigma < tau): s_l = sigma / tau
                x_t = s_l * x_tau_tilde + (1 - s_l) * x_final
                v_target = (x_tau_tilde - x_final) / tau

        Solution B (ablation, use_standard_formula=True): standard sigma formula
            per segment (discontinuous at boundary but balanced magnitudes).
            High stage: x_t = sigma * z + (1-sigma) * x_tau, target = z - x_tau
            Low stage:  x_t = sigma * x_tau_tilde + (1-sigma) * x_final,
                        target = x_tau_tilde - x_final

        Args:
            x_final: (B, z_dim, T', H', W') latents of the final execution video.
            x_tau: (B, z_dim, T', H', W') latents of the search video.
            condition: (B, 4 + z_dim, T', H', W') from prepare_condition.
            prompt_embeds: (B, 512, text_dim).
            tau_sigma: Piecewise boundary in sigma space.
            boundary_noise_std: Gaussian perturbation std for x_tau in low stage.
            use_standard_formula: If True, use Solution B (ablation).

        Returns:
            (loss, debug_dict) where debug_dict has per-stage stats.
        """
        B = x_final.shape[0]
        device = x_final.device
        shifted_sigmas, shifted_timesteps, bsmntw = self._get_training_buffers(device)

        # Sample random timestep indices
        if self.train_experts == "high":
            indices = torch.randint(0, self.boundary_idx, (B,), device=device)
        elif self.train_experts == "low":
            indices = torch.randint(self.boundary_idx, self.num_train_timesteps, (B,), device=device)
        else:
            indices = torch.randint(0, self.num_train_timesteps, (B,), device=device)

        sigmas = shifted_sigmas.index_select(0, indices)  # (B,)
        timesteps = shifted_timesteps.index_select(0, indices)
        weights = bsmntw.index_select(0, indices)

        sigmas_5d = sigmas.view(B, 1, 1, 1, 1)
        noise = torch.randn_like(x_final)

        # Determine which samples fall in high vs low stage
        high_mask = sigmas >= tau_sigma  # (B,)
        low_mask = ~high_mask

        # Build x_t and v_target per sample
        x_t = torch.zeros_like(x_final)
        target = torch.zeros_like(x_final)

        # ---- High stage: noise -> x_tau ----
        if high_mask.any():
            s_high = sigmas_5d  # will be indexed below
            z_h = noise[high_mask]
            x_tau_h = x_tau[high_mask]
            sigma_h = sigmas_5d[high_mask]

            if use_standard_formula:
                # Solution B: x_t = sigma * z + (1-sigma) * x_tau
                x_t[high_mask] = sigma_h * z_h + (1.0 - sigma_h) * x_tau_h
                target[high_mask] = z_h - x_tau_h
            else:
                # Solution A: rescaled for continuity
                s_h = (sigma_h - tau_sigma) / (1.0 - tau_sigma)
                x_t[high_mask] = s_h * z_h + (1.0 - s_h) * x_tau_h
                target[high_mask] = (z_h - x_tau_h) / (1.0 - tau_sigma)

        # ---- Low stage: x_tau -> x_final ----
        if low_mask.any():
            x_tau_l = x_tau[low_mask]
            x_final_l = x_final[low_mask]
            sigma_l = sigmas_5d[low_mask]

            # Boundary perturbation
            if boundary_noise_std > 0:
                eps = torch.randn_like(x_tau_l) * boundary_noise_std
                x_tau_l = x_tau_l + eps

            if use_standard_formula:
                # Solution B: x_t = sigma * x_tau_tilde + (1-sigma) * x_final
                x_t[low_mask] = sigma_l * x_tau_l + (1.0 - sigma_l) * x_final_l
                target[low_mask] = x_tau_l - x_final_l
            else:
                # Solution A: rescaled for continuity
                s_l = sigma_l / tau_sigma
                x_t[low_mask] = s_l * x_tau_l + (1.0 - s_l) * x_final_l
                target[low_mask] = (x_tau_l - x_final_l) / tau_sigma

        # ---- Model forward with MoE routing (unchanged) ----
        model_input = torch.cat([x_t, condition], dim=1)

        experts = []
        if self.transformer is not None:
            experts.append(((timesteps >= self.boundary_timestep).nonzero(as_tuple=False).flatten(), self.transformer))
        if self.transformer_2 is not None:
            experts.append(((timesteps < self.boundary_timestep).nonzero(as_tuple=False).flatten(), self.transformer_2))

        loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_weight = torch.tensor(0.0, device=device, dtype=torch.float32)

        for selected, transformer in experts:
            if selected.numel() == 0:
                continue
            pred = transformer(
                hidden_states=model_input.index_select(0, selected),
                timestep=timesteps.index_select(0, selected),
                encoder_hidden_states=prompt_embeds.index_select(0, selected),
                return_dict=False,
            )[0]
            per_sample_loss = F.mse_loss(pred.float(), target.index_select(0, selected).float(), reduction="none")
            per_sample_loss = per_sample_loss.mean(dim=list(range(1, per_sample_loss.ndim)))
            selected_weights = weights.index_select(0, selected)
            loss = loss + (per_sample_loss * selected_weights).sum()
            total_weight = total_weight + selected_weights.sum()

        final_loss = loss / total_weight if total_weight > 0 else loss

        # ---- Debug stats ----
        with torch.no_grad():
            n_high = high_mask.sum().item()
            n_low = low_mask.sum().item()

            def _per_sample_norm(t: torch.Tensor) -> float:
                # (N, ...) -> mean of per-sample L2 norms
                return t.float().reshape(t.shape[0], -1).norm(dim=1).mean().item()

            debug = {
                "n_high": n_high,
                "n_low": n_low,
                "target_norm_high": _per_sample_norm(target[high_mask]) if n_high > 0 else 0.0,
                "target_norm_low": _per_sample_norm(target[low_mask]) if n_low > 0 else 0.0,
                "sigma_mean": sigmas.mean().item(),
                "sigma_min": sigmas.min().item(),
                "sigma_max": sigmas.max().item(),
            }

        return final_loss, debug

    # ------------------------------------------------------------------
    # Flow-GRPO: SDE sampling & log-probability computation
    # ------------------------------------------------------------------

    def _get_expert_for_timestep(self, timestep: float) -> "WanTransformer3DModel":
        """Route a single scalar timestep to the correct MoE expert."""
        if self.transformer is not None and self.transformer_2 is not None:
            return self.transformer if timestep >= self.boundary_timestep else self.transformer_2
        return self.transformer if self.transformer is not None else self.transformer_2

    def _sde_step(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        sigma: float,
        sigma_prev: float,
        sde_noise_scale: float = 0.7,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single SDE denoising step with log-probability computation.

        Converts the deterministic ODE step into a stochastic SDE step following
        Flow-GRPO (arXiv:2505.05470). The transition becomes Gaussian, enabling
        tractable log-probability and importance ratio computation.

        Args:
            sample: Current noisy latent x_t. Shape (B, C, T, H, W).
            model_output: Velocity prediction v_θ(x_t, t). Same shape.
            sigma: Current noise level (going from 1→0 during denoising).
            sigma_prev: Next noise level (closer to 0).
            sde_noise_scale: Controls exploration. 'a' in σ_sde = a*√(t/(1-t)).
            sigma_min: Floor for SDE noise std.
            sigma_max: Ceiling for SDE noise std.
            noise: Pre-sampled noise (for reproducibility). If None, sampled here.

        Returns:
            (prev_sample, prev_sample_mean, log_prob):
            - prev_sample: x_{t-1} after stochastic step
            - prev_sample_mean: deterministic mean (for KL computation)
            - log_prob: per-sample log probability, shape (B,)
        """
        dt = sigma_prev - sigma  # negative (denoising direction)

        # SDE noise std: interpolate between sigma_min and sigma_max based on sigma
        std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma

        # Transition mean (SDE drift = ODE drift + score correction)
        # prev_mean = x + [v + std²/(2σ) * (x + (1-σ)*v)] * dt
        if sigma > 1e-8:
            score_coeff = std_dev_t**2 / (2.0 * sigma)
            prev_sample_mean = (
                sample * (1.0 + score_coeff * dt)
                + model_output * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * dt
            )
        else:
            # At sigma ≈ 0, skip score correction to avoid division by zero
            prev_sample_mean = sample + model_output * dt

        # Stochastic noise injection
        noise_scale = std_dev_t * math.sqrt(max(-dt, 0.0))
        if noise is None:
            noise = torch.randn_like(sample)
        prev_sample = prev_sample_mean + noise_scale * noise

        # Log probability under the Gaussian transition
        if noise_scale > 1e-8:
            log_prob = (
                -((prev_sample - prev_sample_mean) ** 2) / (2.0 * noise_scale**2)
                - math.log(noise_scale)
                - 0.5 * math.log(2.0 * math.pi)
            )
            # Mean across all dims except batch → per-sample scalar
            log_prob = log_prob.mean(dim=list(range(1, log_prob.ndim)))
        else:
            log_prob = torch.zeros(sample.shape[0], device=sample.device)

        return prev_sample, prev_sample_mean, log_prob

    @torch.no_grad()
    def sde_generate(
        self,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        num_sampling_steps: int = 10,
        sde_noise_scale: float = 0.7,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        cfg_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> dict:
        """Generate video latents via SDE sampling, storing per-step data for GRPO.

        Runs the full denoising loop using the SDE formulation, collecting
        intermediate latents and log-probabilities needed for policy gradient.

        Args:
            condition: (B, 4+z_dim, T', H', W') from prepare_condition.
            prompt_embeds: (B, 512, text_dim) text embeddings.
            num_sampling_steps: Number of denoising steps T.
            sde_noise_scale: 'a' parameter for SDE noise.
            sigma_min: Noise floor.
            sigma_max: Noise ceiling.
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
            generator: Optional RNG for reproducibility.

        Returns:
            dict with keys:
                - latents: list of T+1 latents [x_T, x_{T-1}, ..., x_0]
                - log_probs: list of T per-step log probabilities
                - timesteps: list of T timestep values
                - sigmas: list of T+1 sigma values
                - noises: list of T noise vectors (for recomputation)
        """
        B = condition.shape[0]
        device = condition.device
        latent_shape = (B, condition.shape[1] - 4, *condition.shape[2:])  # (B, z_dim, T', H', W')

        # Build sigma schedule for sampling: T+1 values from 1→0
        # Use linspace in [0, 1] then apply the shifted schedule
        t_values = torch.linspace(1.0, 0.0, num_sampling_steps + 1, device=device)
        shift = 5.0
        sigmas = shift * t_values / (1.0 + (shift - 1.0) * t_values)

        # Start from pure noise
        latent = torch.randn(latent_shape, device=device, dtype=torch.bfloat16, generator=generator)

        all_latents = [latent]
        all_log_probs = []
        all_timesteps = []
        all_noises = []

        for i in range(num_sampling_steps):
            sigma = sigmas[i].item()
            sigma_prev = sigmas[i + 1].item()
            timestep_val = sigma * self.num_train_timesteps

            # Select expert based on timestep
            transformer = self._get_expert_for_timestep(timestep_val)

            # Build model input: [noisy_latents, condition]
            model_input = torch.cat([latent, condition], dim=1)

            # Forward pass through transformer
            timestep_tensor = torch.tensor([timestep_val], device=device, dtype=torch.bfloat16).expand(B)
            model_output = transformer(
                hidden_states=model_input,
                timestep=timestep_tensor,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            # CFG (if scale > 1)
            if cfg_scale > 1.0 and self.transformer is not None:
                # Unconditional forward with zero prompt
                uncond_embeds = torch.zeros_like(prompt_embeds)
                uncond_output = transformer(
                    hidden_states=model_input,
                    timestep=timestep_tensor,
                    encoder_hidden_states=uncond_embeds,
                    return_dict=False,
                )[0]
                model_output = uncond_output + cfg_scale * (model_output - uncond_output)

            # SDE step
            noise = torch.randn_like(latent, generator=generator)
            latent, prev_mean, log_prob = self._sde_step(
                sample=latent,
                model_output=model_output,
                sigma=sigma,
                sigma_prev=sigma_prev,
                sde_noise_scale=sde_noise_scale,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                noise=noise,
            )

            all_latents.append(latent)
            all_log_probs.append(log_prob)
            all_timesteps.append(timestep_val)
            all_noises.append(noise)

        return {
            "latents": all_latents,
            "log_probs": all_log_probs,
            "timesteps": all_timesteps,
            "sigmas": sigmas,
            "noises": all_noises,
        }

    def compute_log_prob_and_kl(
        self,
        latent: torch.Tensor,
        next_latent: torch.Tensor,
        noise: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        sigma: float,
        sigma_prev: float,
        sde_noise_scale: float = 0.7,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        use_ref: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log-probability and KL for a stored (x_t, x_{t-1}) pair.

        Used in the GRPO training phase to compute importance ratios and KL
        divergence against the reference policy.

        Args:
            latent: x_t, the noisy latent at step t. Shape (B, C, T, H, W).
            next_latent: x_{t-1}, the denoised latent at step t-1.
            noise: The noise vector used in the original SDE step.
            condition: (B, 4+z_dim, T', H', W').
            prompt_embeds: (B, 512, text_dim).
            sigma: Noise level at step t.
            sigma_prev: Noise level at step t-1.
            sde_noise_scale: SDE noise parameter.
            sigma_min: Noise floor.
            sigma_max: Noise ceiling.
            use_ref: If True, disable LoRA adapters to use reference policy.

        Returns:
            (log_prob, kl_div, prev_sample_mean):
            - log_prob: Per-sample log probability under current/ref policy, shape (B,).
            - kl_div: Per-sample KL divergence between current and ref means, shape (B,).
            - prev_sample_mean: The predicted mean (for KL computation).
        """
        timestep_val = sigma * self.num_train_timesteps
        B = latent.shape[0]
        device = latent.device

        # Select expert
        transformer = self._get_expert_for_timestep(timestep_val)

        # Optionally switch to reference policy
        if use_ref:
            transformer.disable_adapters()

        # Forward pass
        model_input = torch.cat([latent, condition], dim=1)
        timestep_tensor = torch.tensor([timestep_val], device=device, dtype=torch.bfloat16).expand(B)
        model_output = transformer(
            hidden_states=model_input,
            timestep=timestep_tensor,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        # Re-enable adapters
        if use_ref:
            transformer.enable_adapters()

        # Compute mean from the SDE step formula
        dt = sigma_prev - sigma
        std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma

        if sigma > 1e-8:
            prev_sample_mean = (
                latent * (1.0 + std_dev_t**2 / (2.0 * sigma) * dt)
                + model_output * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * dt
            )
        else:
            prev_sample_mean = latent + model_output * dt

        # Log probability
        noise_scale = std_dev_t * math.sqrt(max(-dt, 0.0))
        if noise_scale > 1e-8:
            log_prob = (
                -((next_latent - prev_sample_mean) ** 2) / (2.0 * noise_scale**2)
                - math.log(noise_scale)
                - 0.5 * math.log(2.0 * math.pi)
            )
            log_prob = log_prob.mean(dim=list(range(1, log_prob.ndim)))
        else:
            log_prob = torch.zeros(B, device=device)

        # KL divergence placeholder (computed externally between current and ref means)
        kl_div = torch.zeros(B, device=device)

        return log_prob, kl_div, prev_sample_mean

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized latents back to pixel space via VAE.

        Args:
            latents: (B, z_dim, T', H', W') normalized latents.

        Returns:
            (B, C, T, H, W) pixel-space video in [-1, 1].
        """
        mean, std_inv = self._get_latent_stats(latents.device, latents.dtype)
        # Undo normalization: latents = (raw - mean) * std_inv → raw = latents / std_inv + mean
        raw_latents = latents / std_inv + mean
        return self.vae.decode(raw_latents.to(self.vae.dtype)).sample
