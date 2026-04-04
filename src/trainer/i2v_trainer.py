"""Wan2.2 I2V Trainer with FSDP2 + Distributed Checkpoint (DCP)."""

import time
from pathlib import Path

import torch
from loguru import logger

from src.trainer.base_trainer import BaseTrainer
from src.trainer.config import TrainConfig
from src.trainer.flops import MFUMonitor, compute_wan_seq_len, estimate_wan_forward_flops, get_gpu_peak_flops_bf16
from src.trainer.utils import cosine_lr, format_eta, to_model_pixels


class I2VTrainer(BaseTrainer):
    def _post_init(self, cfg: TrainConfig) -> None:
        self.mfu_monitor = self._setup_mfu()

    def _setup_mfu(self) -> MFUMonitor | None:
        """Pre-compute FLOPs and create MFU monitor. Returns None if GPU is unrecognized."""
        gpu_peak = get_gpu_peak_flops_bf16()
        if gpu_peak is None:
            return None

        bi = self.model.boundary_idx
        N = self.model.num_train_timesteps
        experts = []
        if self.model.transformer is not None:
            prob = bi / N if self._effective_train_experts == "both" else 1.0
            experts.append((prob, self.model.transformer))
        if self.model.transformer_2 is not None:
            prob = (N - bi) / N if self._effective_train_experts == "both" else 1.0
            experts.append((prob, self.model.transformer_2))

        est_cfg = self.dataset._item_configs[0]
        if est_cfg.fixed_height is not None and est_cfg.fixed_width is not None:
            est_h, est_w = est_cfg.fixed_height, est_cfg.fixed_width
        else:
            est_h = est_w = int(est_cfg.max_area**0.5)

        weighted_fwd_flops = 0.0
        seq_len = 0
        for prob, t in experts:
            t_cfg = t.config
            seq_len = compute_wan_seq_len(
                est_cfg.num_frames, est_h, est_w,
                patch_size=tuple(t_cfg.patch_size),
                vae_temporal_factor=self.model.vae_scale_factor_temporal,
                vae_spatial_factor=self.model.vae_scale_factor_spatial,
            )
            fwd = estimate_wan_forward_flops(
                num_layers=t_cfg.num_layers,
                num_heads=t_cfg.num_attention_heads,
                head_dim=t_cfg.attention_head_dim,
                ffn_dim=t_cfg.ffn_dim,
                seq_len=seq_len,
            )
            weighted_fwd_flops += prob * fwd

        flops_per_step = 3 * weighted_fwd_flops * self.cfg.batch_size * self.cfg.gradient_accumulation_steps

        logger.info(
            "MFU monitor: seq_len={}, forward={:.2e} FLOPs/sample, step={:.2e} FLOPs, GPU={} ({:.0f} TFLOPS bf16)",
            seq_len, weighted_fwd_flops, flops_per_step,
            torch.cuda.get_device_name(0), gpu_peak / 1e12,
        )
        return MFUMonitor(flops_per_step, gpu_peak)

    # ------------------------------------------------------------------
    # Training loop
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
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(self.dataloader):
                is_last_micro_step = (batch_idx + 1) % cfg.gradient_accumulation_steps == 0
                self._set_requires_gradient_sync(is_last_micro_step)
                loss = self._train_step(batch)
                scaled_loss = loss / cfg.gradient_accumulation_steps
                scaled_loss.backward()

                if is_last_micro_step:
                    self._last_grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm).item()

                    lr = cosine_lr(global_step, cfg.warmup_steps, self.total_steps, cfg.learning_rate)
                    for opt in self.optimizers:
                        for pg in opt.param_groups:
                            pg["lr"] = lr
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update()
                    global_step += 1
                    if self.mfu_monitor is not None:
                        self.mfu_monitor.step()

                    if self.rank == 0 and global_step % cfg.log_steps == 0:
                        mfu = self.mfu_monitor.flush() if self.mfu_monitor is not None else None
                        mfu_str = f"{mfu:.1%}" if mfu is not None else "-"

                        # ETA
                        elapsed = time.monotonic() - train_start_time
                        steps_done = global_step - train_start_step
                        if steps_done > 0:
                            secs_per_step = elapsed / steps_done
                            eta_secs = secs_per_step * (self.total_steps - global_step)
                            eta_str = format_eta(eta_secs)
                            s_it_str = f"{secs_per_step:.1f}"
                        else:
                            eta_str = "?"
                            s_it_str = "?"

                        logger.info(
                            "step={}/{} epoch={} loss={:.4f} lr={:.2e} grad_norm={:.4f} mfu={} eta={} ({} s/it)",
                            global_step, self.total_steps, epoch,
                            loss.item(), lr, self._last_grad_norm, mfu_str, eta_str, s_it_str,
                        )

                        if self.use_wandb:
                            import wandb

                            log_metrics = {
                                "train/loss": loss.item(),
                                "train/lr": lr,
                                "train/epoch": epoch,
                                "train/grad_norm": self._last_grad_norm,
                            }
                            if mfu is not None:
                                log_metrics["train/mfu"] = mfu
                            wandb.log(log_metrics, step=global_step)

                    if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                        self.train_state.step = global_step
                        self.train_state.epoch = epoch
                        self.train_state.batch_idx = batch_idx + 1
                        self._save_checkpoint(output_dir / f"checkpoint-{global_step}")

            # End-of-epoch save
            self.train_state.step = global_step
            self.train_state.epoch = epoch + 1
            self.train_state.batch_idx = 0
            self._save_checkpoint(output_dir / f"checkpoint-epoch{epoch}")
            logger.info("Epoch {} done.", epoch)

        if self.use_wandb:
            import wandb
            wandb.finish()
        import torch.distributed as dist
        dist.destroy_process_group()

    def _train_step(self, batch: dict) -> torch.Tensor:
        """Single forward pass: encode frozen inputs, compute loss."""
        prompt_embeds = self.model.encode_text(batch["prompt"], self.device)
        video = to_model_pixels(batch["video"], self.device)
        image = to_model_pixels(batch["image"], self.device)
        video_latents = self.model.encode_video(video)
        condition = self.model.prepare_condition(image, video.shape[2], video.shape[-2], video.shape[-1])
        return self.model.compute_loss(video_latents, condition, prompt_embeds)
