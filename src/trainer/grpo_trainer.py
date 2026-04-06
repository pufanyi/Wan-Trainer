"""Wan2.2 I2V Flow-GRPO Trainer with FSDP2 + DCP.

Implements Flow-GRPO (arXiv:2505.05470): online RL for flow matching models
by converting ODE sampling to SDE for tractable log-probability computation.

Training loop:
  1. Sampling phase: SDE-generate G videos per prompt (no_grad)
  2. Reward phase: compute rewards for generated videos
  3. Advantage phase: per-prompt group normalization
  4. Training phase: policy gradient with clipped importance ratio + KL penalty
"""

import math
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from src.trainer.base_trainer import BaseTrainer
from src.trainer.config import TrainConfig
from src.trainer.utils import cosine_lr, format_eta, shard_transformer, to_model_pixels


class GRPOTrainer(BaseTrainer):
    """Flow-GRPO trainer for Wan2.2 I2V models.

    Supports both LoRA (reference = base model via disable_adapter) and
    full fine-tuning (reference = frozen deepcopy of initial model).
    """

    def __init__(self, cfg: TrainConfig):
        assert cfg.grpo_group_size is not None and cfg.grpo_group_size > 0, (
            "grpo_group_size must be > 0 for GRPO training"
        )
        super().__init__(cfg)
        logger.info("Flow-GRPO | G={} T={}", cfg.grpo_group_size, cfg.grpo_num_sampling_steps)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _pre_fsdp_setup(self, cfg: TrainConfig) -> None:
        """Create frozen reference policy copies for full fine-tuning."""
        self.is_lora = cfg.lora_rank > 0
        self.ref_transformers: dict[str, torch.nn.Module] = {}
        if not self.is_lora:
            logger.info("Full fine-tuning mode: creating frozen reference policy copies")
            if self.model.transformer is not None:
                self.ref_transformers["transformer"] = deepcopy(self.model.transformer).requires_grad_(False).eval()
            if self.model.transformer_2 is not None:
                self.ref_transformers["transformer_2"] = deepcopy(self.model.transformer_2).requires_grad_(False).eval()

    def _setup_fsdp(self, cfg: TrainConfig) -> list[torch.nn.Module]:
        sync_modules = super()._setup_fsdp(cfg)
        # Also shard frozen reference transformers
        for _name, ref in self.ref_transformers.items():
            shard_transformer(ref, self.mesh, self.mp_policy)
        return sync_modules

    def _compute_total_steps(self) -> int:
        # GRPO: each batch = one optimizer step (no gradient_accumulation_steps splitting)
        return self.cfg.num_epochs * len(self.dataloader)

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_reward_neg_loss(
        self,
        generated_latents: torch.Tensor,
        gt_video_latents: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Reward = -flow_matching_loss against ground truth video."""
        B = gt_video_latents.shape[0]
        device = gt_video_latents.device
        shifted_sigmas, shifted_timesteps, bsmntw = self.model._get_training_buffers(device)

        indices = torch.randint(0, self.model.num_train_timesteps, (B,), device=device)
        sigmas = shifted_sigmas.index_select(0, indices).view(B, 1, 1, 1, 1)
        timesteps = shifted_timesteps.index_select(0, indices)

        noise = torch.randn_like(gt_video_latents)
        noisy = sigmas * noise + (1.0 - sigmas) * gt_video_latents
        target = noise - gt_video_latents

        model_input = torch.cat([noisy, condition], dim=1)

        transformer = self.model._get_expert_for_timestep(timesteps[0].item())
        pred = transformer(
            hidden_states=model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        per_sample_loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        per_sample_loss = per_sample_loss.mean(dim=list(range(1, per_sample_loss.ndim)))
        return -per_sample_loss  # (B,)

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Group-relative advantage normalization.

        Args:
            rewards: (B, G) rewards for each sample in each group.

        Returns:
            (B, G) advantages, normalized per group (per row).
        """
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True) + 1e-4
        advantages = (rewards - mean) / std
        return advantages.clamp(-self.cfg.grpo_adv_clip_max, self.cfg.grpo_adv_clip_max)

    # ------------------------------------------------------------------
    # Reference policy forward
    # ------------------------------------------------------------------

    def _ref_forward(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep_val: float,
    ) -> torch.Tensor:
        """Forward pass through reference policy. Returns velocity prediction."""
        B = latent.shape[0]
        device = latent.device
        model_input = torch.cat([latent, condition], dim=1)
        timestep_tensor = torch.tensor([timestep_val], device=device, dtype=torch.bfloat16).expand(B)

        if self.is_lora:
            transformer = self.model._get_expert_for_timestep(timestep_val)
            transformer.disable_adapters()
            try:
                out = transformer(
                    hidden_states=model_input,
                    timestep=timestep_tensor,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            finally:
                transformer.enable_adapters()
            return out
        else:
            if timestep_val >= self.model.boundary_timestep:
                ref = self.ref_transformers.get("transformer")
            else:
                ref = self.ref_transformers.get("transformer_2")
            if ref is None:
                ref = next(iter(self.ref_transformers.values()))
            return ref(
                hidden_states=model_input,
                timestep=timestep_tensor,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

    # ------------------------------------------------------------------
    # Core GRPO step
    # ------------------------------------------------------------------

    def _grpo_step(self, batch: dict) -> dict[str, float]:
        """One full GRPO step: sample -> reward -> advantage -> policy gradient.

        Uses ``grpo_sample_batch_size`` (S) to control how many of the G samples
        are processed in one forward pass.  S=1 is fully serial (safest for
        memory); S=G is fully batched (fastest but may OOM).  Intermediate
        values (e.g. S=2 or 4) give a practical trade-off.
        """
        cfg = self.cfg
        G = cfg.grpo_group_size
        S = cfg.grpo_sample_batch_size
        T = cfg.grpo_num_sampling_steps
        device = self.device

        # ---- Encode inputs once (B samples) ----
        prompt_embeds = self.model.encode_text(batch["prompt"], device)
        video = to_model_pixels(batch["video"], device)
        image = to_model_pixels(batch["image"], device)
        gt_video_latents = self.model.encode_video(video)
        condition = self.model.prepare_condition(image, video.shape[2], video.shape[-2], video.shape[-1])

        B = gt_video_latents.shape[0]

        # ---- Phase 1: SDE Sampling in chunks of S (no_grad) ----
        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.eval()

        all_chunk_trajs = []   # list of trajectory dicts, each with batch B*cur_S
        reward_chunks = []     # list of (B, cur_S) reward tensors

        for g_start in range(0, G, S):
            cur_S = min(S, G - g_start)
            cond_s = condition.repeat_interleave(cur_S, dim=0)       # (B*cur_S, ...)
            pe_s = prompt_embeds.repeat_interleave(cur_S, dim=0)     # (B*cur_S, ...)
            gt_s = gt_video_latents.repeat_interleave(cur_S, dim=0)  # (B*cur_S, ...)

            traj = self.model.sde_generate(
                condition=cond_s,
                prompt_embeds=pe_s,
                num_sampling_steps=T,
                sde_noise_scale=cfg.grpo_sde_noise_scale,
                sigma_min=cfg.grpo_sde_sigma_min,
                sigma_max=cfg.grpo_sde_sigma_max,
                cfg_scale=cfg.grpo_cfg_scale,
            )
            all_chunk_trajs.append((traj, cur_S))

            # Reward for this chunk
            reward_flat = self._compute_reward_neg_loss(
                traj["latents"][-1], gt_s, cond_s, pe_s,
            )  # (B*cur_S,)
            reward_chunks.append(reward_flat.view(B, cur_S))

        rewards = torch.cat(reward_chunks, dim=1)  # (B, G)

        # ---- Phase 2: Advantage Computation ----
        if self.world_size > 1:
            all_ranks_rewards = [torch.zeros_like(rewards) for _ in range(self.world_size)]
            dist.all_gather(all_ranks_rewards, rewards)
            gathered_rewards = torch.cat(all_ranks_rewards, dim=0)
            global_mean = gathered_rewards.mean()
            global_std = gathered_rewards.std() + 1e-4
            advantages = ((rewards - global_mean) / global_std).clamp(
                -cfg.grpo_adv_clip_max, cfg.grpo_adv_clip_max
            )
        else:
            advantages = self._compute_advantages(rewards)

        # ---- Phase 3: GRPO Training (with grad) ----
        # Iterate timesteps × chunks.  Each chunk forward has batch B*cur_S.
        # Gradients accumulate across all T*num_chunks micro-steps.
        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.train()

        total_policy_loss = 0.0
        total_kl_loss = 0.0
        num_chunks = len(all_chunk_trajs)
        total_accum_steps = T * num_chunks

        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        g_offset = 0  # tracks which slice of advantages[:, g_offset:g_offset+cur_S]
        for chunk_idx, (traj, cur_S) in enumerate(all_chunk_trajs):
            BS = B * cur_S
            cond_s = condition.repeat_interleave(cur_S, dim=0).detach()
            pe_s = prompt_embeds.repeat_interleave(cur_S, dim=0).detach()
            adv_chunk = advantages[:, g_offset : g_offset + cur_S].reshape(BS)  # (B*cur_S,)
            g_offset += cur_S

            for t_idx in range(T):
                is_last = (chunk_idx == num_chunks - 1 and t_idx == T - 1)
                self._set_requires_gradient_sync(is_last)

                sigma = traj["sigmas"][t_idx].item()
                sigma_prev = traj["sigmas"][t_idx + 1].item()
                latent = traj["latents"][t_idx].detach()          # (B*cur_S, ...)
                next_latent = traj["latents"][t_idx + 1].detach()
                old_log_prob = traj["log_probs"][t_idx].detach()  # (B*cur_S,)
                timestep_val = traj["timesteps"][t_idx]

                # Current policy forward (with grad)
                timestep_tensor = torch.tensor([timestep_val], device=device, dtype=torch.bfloat16).expand(BS)
                transformer = self.model._get_expert_for_timestep(timestep_val)
                model_input = torch.cat([latent, cond_s], dim=1)

                model_output = transformer(
                    hidden_states=model_input,
                    timestep=timestep_tensor,
                    encoder_hidden_states=pe_s,
                    return_dict=False,
                )[0]

                # Compute current log prob from SDE transition
                dt = sigma_prev - sigma
                std_dev_t = cfg.grpo_sde_sigma_min + (cfg.grpo_sde_sigma_max - cfg.grpo_sde_sigma_min) * sigma
                noise_scale = std_dev_t * math.sqrt(max(-dt, 0.0))

                if sigma > 1e-8:
                    prev_mean = (
                        latent * (1.0 + std_dev_t**2 / (2.0 * sigma) * dt)
                        + model_output * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * dt
                    )
                else:
                    prev_mean = latent + model_output * dt

                if noise_scale > 1e-8:
                    new_log_prob = (
                        -((next_latent - prev_mean) ** 2) / (2.0 * noise_scale**2)
                        - math.log(noise_scale)
                        - 0.5 * math.log(2.0 * math.pi)
                    )
                    new_log_prob = new_log_prob.mean(dim=list(range(1, new_log_prob.ndim)))
                else:
                    new_log_prob = torch.zeros(BS, device=device)

                # Importance ratio + clipped surrogate
                ratio = torch.exp(new_log_prob - old_log_prob)
                unclipped = -adv_chunk * ratio
                clipped = -adv_chunk * ratio.clamp(1.0 - cfg.grpo_clip_range, 1.0 + cfg.grpo_clip_range)
                policy_loss = torch.max(unclipped, clipped).mean()

                # KL penalty against reference policy
                kl_loss = torch.tensor(0.0, device=device)
                if cfg.grpo_kl_coeff > 0:
                    with torch.no_grad():
                        ref_output = self._ref_forward(latent, cond_s, pe_s, timestep_val)
                    if noise_scale > 1e-8:
                        ref_mean = _compute_ref_mean(latent, ref_output, sigma, sigma_prev, std_dev_t, dt)
                        kl_loss = ((prev_mean - ref_mean) ** 2).mean(dim=list(range(1, prev_mean.ndim))).mean()
                        kl_loss = kl_loss / (2.0 * noise_scale**2)

                loss = (policy_loss + cfg.grpo_kl_coeff * kl_loss) / total_accum_steps
                loss.backward()

                total_policy_loss += policy_loss.item()
                total_kl_loss += kl_loss.item()

        return {
            "policy_loss": total_policy_loss / (T * num_chunks),
            "kl_loss": total_kl_loss / (T * num_chunks),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        global_step = self.train_state.step
        start_epoch = self.train_state.epoch
        train_start_time = time.monotonic()
        train_start_step = global_step

        for epoch in range(start_epoch, cfg.num_epochs):
            self.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(self.dataloader):
                metrics = self._grpo_step(batch)

                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm).item()

                lr = cosine_lr(global_step, cfg.warmup_steps, self.total_steps, cfg.learning_rate)
                for opt in self.optimizers:
                    for pg in opt.param_groups:
                        pg["lr"] = lr
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                if self.ema is not None:
                    self.ema.update()

                global_step += 1

                if self.rank == 0 and global_step % cfg.log_steps == 0:
                    elapsed = time.monotonic() - train_start_time
                    steps_done = global_step - train_start_step
                    if steps_done > 0:
                        secs_per_step = elapsed / steps_done
                        eta_str = format_eta(secs_per_step * (self.total_steps - global_step))
                        speed_str = f"{secs_per_step:.2f}"
                    else:
                        eta_str, speed_str = "?", "?"

                    logger.info(
                        "step={}/{} epoch={} policy_loss={:.4f} kl_loss={:.4f} reward={:.4f}+/-{:.4f} "
                        "lr={:.2e} grad_norm={:.4f} eta={} ({} s/it)",
                        global_step, self.total_steps, epoch,
                        metrics["policy_loss"], metrics["kl_loss"],
                        metrics["reward_mean"], metrics["reward_std"],
                        lr, grad_norm, eta_str, speed_str,
                    )

                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            "grpo/policy_loss": metrics["policy_loss"],
                            "grpo/kl_loss": metrics["kl_loss"],
                            "grpo/reward_mean": metrics["reward_mean"],
                            "grpo/reward_std": metrics["reward_std"],
                            "grpo/advantage_mean": metrics["advantage_mean"],
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/epoch": epoch,
                        }, step=global_step)

                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    self.train_state.step = global_step
                    self.train_state.epoch = epoch
                    self.train_state.batch_idx = batch_idx + 1
                    self._save_checkpoint(output_dir / f"checkpoint-{global_step}")

            self.train_state.step = global_step
            self.train_state.epoch = epoch + 1
            self.train_state.batch_idx = 0
            self._save_checkpoint(output_dir / f"checkpoint-epoch{epoch}")
            logger.info("Epoch {} done.", epoch)

        if self.use_wandb:
            import wandb
            wandb.finish()
        dist.destroy_process_group()


def _compute_ref_mean(
    latent: torch.Tensor,
    ref_output: torch.Tensor,
    sigma: float,
    sigma_prev: float,
    std_dev_t: float,
    dt: float,
) -> torch.Tensor:
    """Compute the transition mean under the reference policy."""
    if sigma > 1e-8:
        return (
            latent * (1.0 + std_dev_t**2 / (2.0 * sigma) * dt)
            + ref_output * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * dt
        )
    return latent + ref_output * dt
