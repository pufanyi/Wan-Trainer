"""Wan2.2 I2V Trainer with FSDP2 + Distributed Checkpoint (DCP)."""

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, DistributedSampler

from src.data.i2v_dataset import I2VDataset
from src.models.wan_i2v import LoRATrainConfig, WanI2VForTraining
from src.trainer.checkpoint import TrainState
from src.trainer.config import TrainConfig
from src.trainer.ema import EMA
from src.trainer.flops import MFUMonitor, compute_wan_seq_len, estimate_wan_forward_flops, get_gpu_peak_flops_bf16
from src.trainer.utils import collate, cosine_lr, setup_loguru, shard_transformer, to_model_pixels


def _format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


class I2VTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        # ---- Distributed ----
        dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(cfg.seed + self.rank)

        setup_loguru(self.rank)

        logger.info("World size: {}", self.world_size)

        # ---- Model ----
        self.model = self._build_model(cfg)

        # ---- FSDP2 ----
        self.sync_modules = self._setup_fsdp(cfg)

        # ---- torch.compile ----
        if cfg.torch_compile:
            self._compile_modules(cfg)

        # ---- Dataset / DataLoader ----
        dataset, self.sampler = self._build_dataset(cfg)
        self.dataloader = self._build_dataloader(dataset, cfg)

        # ---- Optimizer ----
        self.params, self.optimizers, self.optimizer_te, self.optimizer_1, self.optimizer_2 = (
            self._build_optimizers(cfg)
        )

        # ---- EMA ----
        self.ema = None
        if cfg.ema_decay > 0:
            ema_models: dict[str, torch.nn.Module] = {}
            if cfg.train_text_encoder:
                ema_models["text_encoder"] = self.model.text_encoder
            if self.model.transformer is not None:
                ema_models["transformer"] = self.model.transformer
            if self.model.transformer_2 is not None:
                ema_models["transformer_2"] = self.model.transformer_2
            self.ema = EMA(ema_models, decay=cfg.ema_decay)
            logger.info("EMA enabled (decay={}, {} shadow params)", cfg.ema_decay, len(self.ema.shadow))

        self.total_steps = cfg.num_epochs * len(self.dataloader) // cfg.gradient_accumulation_steps
        logger.info(
            "Dataset: {} samples, {} batches/epoch, {} total optimizer steps",
            len(dataset),
            len(self.dataloader),
            self.total_steps,
        )

        # ---- DCP state ----
        self.train_state = TrainState(
            text_encoder=self.model.text_encoder if cfg.train_text_encoder else None,
            transformer=self.model.transformer,
            transformer_2=self.model.transformer_2,
            optimizer_te=self.optimizer_te,
            optimizer_1=self.optimizer_1,
            optimizer_2=self.optimizer_2,
        )

        # ---- MFU monitor ----
        self.mfu_monitor = self._setup_mfu()

        # ---- Wandb ----
        self.use_wandb = cfg.wandb_project is not None and self.rank == 0
        if self.use_wandb:
            import wandb

            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=cfg.model_dump(),
            )

        # ---- Resume ----
        resume_path = cfg.resume_from or (self._find_latest_checkpoint() if cfg.auto_resume else None)
        if resume_path:
            self._load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_model(self, cfg: TrainConfig) -> WanI2VForTraining:
        lora_cfg = (
            LoRATrainConfig(rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
            if cfg.lora_rank > 0
            else None
        )
        logger.info(
            "Loading model from {} (lora_rank={}, experts={}) ...",
            cfg.model_path,
            cfg.lora_rank,
            cfg.train_experts,
        )
        model = WanI2VForTraining(
            cfg.model_path,
            lora_config=lora_cfg,
            train_experts=cfg.train_experts,
            train_text_encoder=cfg.train_text_encoder,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )
        model.text_encoder.to(self.device)
        model.vae.to(self.device)
        return model

    def _setup_fsdp(self, cfg: TrainConfig) -> list[torch.nn.Module]:
        mesh = init_device_mesh("cuda", (self.world_size,))
        dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        mp_policy = MixedPrecisionPolicy(
            param_dtype=dtype_map[cfg.param_dtype],
            reduce_dtype=dtype_map[cfg.reduce_dtype],
        )
        if cfg.train_text_encoder:
            fully_shard(self.model.text_encoder, mesh=mesh, mp_policy=mp_policy)
        if self.model.transformer is not None:
            shard_transformer(self.model.transformer, mesh, mp_policy)
        if self.model.transformer_2 is not None:
            shard_transformer(self.model.transformer_2, mesh, mp_policy)
        return [
            m
            for m in [
                self.model.text_encoder if cfg.train_text_encoder else None,
                self.model.transformer,
                self.model.transformer_2,
            ]
            if m is not None
        ]

    def _compile_modules(self, cfg: TrainConfig) -> None:
        compile_kwargs = {"backend": cfg.torch_compile_backend}
        if cfg.torch_compile_mode is not None:
            compile_kwargs["mode"] = cfg.torch_compile_mode
        self.model.vae = torch.compile(self.model.vae, **compile_kwargs)
        logger.info("Compiled vae")
        if not cfg.train_text_encoder:
            self.model.text_encoder = torch.compile(self.model.text_encoder, **compile_kwargs)
            logger.info("Compiled text_encoder")
        if self.model.transformer is not None:
            self.model.transformer = torch.compile(self.model.transformer, **compile_kwargs)
            logger.info("Compiled transformer")
        if self.model.transformer_2 is not None:
            self.model.transformer_2 = torch.compile(self.model.transformer_2, **compile_kwargs)
            logger.info("Compiled transformer_2")
        logger.info("torch.compile enabled (backend={}, mode={})", cfg.torch_compile_backend, cfg.torch_compile_mode)

    def _build_dataset(self, cfg: TrainConfig) -> tuple:
        dataset = I2VDataset(
            json_path=cfg.dataset_json,
            num_frames=cfg.num_frames,
            max_area=cfg.max_area,
            height=cfg.height,
            width=cfg.width,
            fps=cfg.fps,
        )
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, seed=cfg.seed)
        return dataset, sampler

    def _build_dataloader(self, dataset, cfg: TrainConfig) -> DataLoader:
        kwargs = dict(
            dataset=dataset,
            batch_size=cfg.batch_size,
            sampler=self.sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate,
            drop_last=True,
        )
        if cfg.num_workers > 0:
            kwargs["persistent_workers"] = cfg.persistent_workers
            kwargs["prefetch_factor"] = cfg.prefetch_factor
        return DataLoader(**kwargs)

    def _build_optimizers(self, cfg: TrainConfig):
        optimizer_te = None
        optimizer_1 = None
        optimizer_2 = None
        params = []
        total_params = 0
        optim_kwargs = dict(lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), fused=True)

        if cfg.train_text_encoder:
            params_te = [p for p in self.model.text_encoder.parameters() if p.requires_grad]
            params.extend(params_te)
            total_params += sum(p.numel() for p in self.model.text_encoder.parameters())
            optimizer_te = torch.optim.AdamW(params_te, **optim_kwargs)

        if self.model.transformer is not None:
            params_1 = [p for p in self.model.transformer.parameters() if p.requires_grad]
            params.extend(params_1)
            total_params += sum(p.numel() for p in self.model.transformer.parameters())
            optimizer_1 = torch.optim.AdamW(params_1, **optim_kwargs)

        if self.model.transformer_2 is not None:
            params_2 = [p for p in self.model.transformer_2.parameters() if p.requires_grad]
            params.extend(params_2)
            total_params += sum(p.numel() for p in self.model.transformer_2.parameters())
            optimizer_2 = torch.optim.AdamW(params_2, **optim_kwargs)

        trainable_count = sum(p.numel() for p in params)
        logger.info(
            "Trainable: {:.1f}M / {:.1f}M ({:.2f}%)",
            trainable_count / 1e6,
            total_params / 1e6,
            100 * trainable_count / total_params,
        )

        optimizers = [opt for opt in [optimizer_te, optimizer_1, optimizer_2] if opt is not None]
        return params, optimizers, optimizer_te, optimizer_1, optimizer_2

    def _setup_mfu(self) -> MFUMonitor | None:
        """Pre-compute FLOPs and create MFU monitor. Returns None if GPU is unrecognized."""
        gpu_peak = get_gpu_peak_flops_bf16()
        if gpu_peak is None:
            return None

        bi = self.model.boundary_idx
        N = self.model.num_train_timesteps
        experts = []
        if self.model.transformer is not None:
            prob = bi / N if self.cfg.train_experts == "both" else 1.0
            experts.append((prob, self.model.transformer))
        if self.model.transformer_2 is not None:
            prob = (N - bi) / N if self.cfg.train_experts == "both" else 1.0
            experts.append((prob, self.model.transformer_2))

        if self.cfg.height is not None and self.cfg.width is not None:
            est_h, est_w = self.cfg.height, self.cfg.width
        else:
            est_h = est_w = int(self.cfg.max_area**0.5)

        weighted_fwd_flops = 0.0
        seq_len = 0
        for prob, t in experts:
            t_cfg = t.config
            seq_len = compute_wan_seq_len(
                self.cfg.num_frames,
                est_h,
                est_w,
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
            seq_len,
            weighted_fwd_flops,
            flops_per_step,
            torch.cuda.get_device_name(0),
            gpu_peak / 1e12,
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
        resume_batch_idx = self.train_state.batch_idx
        train_start_time = time.monotonic()
        train_start_step = global_step

        for epoch in range(start_epoch, cfg.num_epochs):
            self.sampler.set_epoch(epoch)
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx < resume_batch_idx:
                    continue
                resume_batch_idx = 0

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
                            eta_str = _format_eta(eta_secs)
                            it_s_str = f"{1 / secs_per_step:.2f}"
                        else:
                            eta_str = "?"
                            it_s_str = "?"

                        logger.info(
                            "step={}/{} epoch={} loss={:.4f} lr={:.2e} grad_norm={:.4f} mfu={} eta={} ({} it/s)",
                            global_step,
                            self.total_steps,
                            epoch,
                            loss.item(),
                            lr,
                            self._last_grad_norm,
                            mfu_str,
                            eta_str,
                            it_s_str,
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
        dist.destroy_process_group()

    def _train_step(self, batch: dict) -> torch.Tensor:
        """Single forward pass: encode frozen inputs, compute loss."""
        prompt_embeds = self.model.encode_text(batch["prompt"], self.device)
        video = to_model_pixels(batch["video"], self.device)
        image = to_model_pixels(batch["image"], self.device)
        video_latents = self.model.encode_video(video)
        condition = self.model.prepare_condition(image, self.cfg.num_frames, video.shape[-2], video.shape[-1])
        return self.model.compute_loss(video_latents, condition, prompt_embeds)

    def _set_requires_gradient_sync(self, requires_gradient_sync: bool) -> None:
        for module in self.sync_modules:
            if hasattr(module, "set_requires_gradient_sync"):
                module.set_requires_gradient_sync(requires_gradient_sync, recurse=True)

    # ------------------------------------------------------------------
    # DCP checkpointing
    # ------------------------------------------------------------------

    def _find_latest_checkpoint(self) -> str | None:
        """Scan output_dir for the latest valid DCP checkpoint (contains .metadata)."""
        output_dir = Path(self.cfg.output_dir)
        if not output_dir.exists():
            return None
        candidates: list[tuple[int, Path]] = []
        for d in output_dir.iterdir():
            if not d.is_dir() or not (d / ".metadata").exists():
                continue
            name = d.name
            if name.startswith("checkpoint-epoch"):
                try:
                    int(name.removeprefix("checkpoint-epoch"))  # validate format
                    candidates.append((int(d.stat().st_mtime_ns), d))
                except ValueError:
                    continue
            elif name.startswith("checkpoint-"):
                try:
                    int(name.removeprefix("checkpoint-"))
                    candidates.append((int(d.stat().st_mtime_ns), d))
                except ValueError:
                    continue
        if not candidates:
            return None
        candidates.sort()
        latest = candidates[-1][1]
        logger.info("Auto-resume: found checkpoint {}", latest)
        return str(latest)

    def _save_checkpoint(self, path: Path):
        """Save with DCP. All ranks participate; each writes its own shards."""
        state: dict = {"train_state": self.train_state}
        if self.ema is not None:
            state["ema"] = self.ema
        dcp.save(state, checkpoint_id=str(path))
        if self.rank == 0 and self.model.lora_config is not None:
            self.model.save_lora(str(path / "lora"))
        if self.rank == 0:
            logger.info("Saved DCP checkpoint to {}", path)
        dist.barrier()

    def _load_checkpoint(self, path: str):
        """Load with DCP. Supports resharding across different world sizes."""
        logger.info("Resuming from {} ...", path)
        state: dict = {"train_state": self.train_state}
        has_legacy_ema = (Path(path) / "ema").is_dir()
        if self.ema is not None and not has_legacy_ema:
            state["ema"] = self.ema
        try:
            dcp.load(state, checkpoint_id=path)
        except Exception:
            if "ema" in state:
                logger.warning("Failed to load EMA from DCP, retrying without EMA")
                dcp.load({"train_state": self.train_state}, checkpoint_id=path)
            else:
                raise
        if self.ema is not None and "ema" not in state:
            self.ema.reinitialize()
            logger.warning("EMA not in DCP checkpoint, reinitialized from loaded model weights")
        logger.info(
            "Resumed at step={} epoch={} batch_idx={}",
            self.train_state.step,
            self.train_state.epoch,
            self.train_state.batch_idx,
        )
