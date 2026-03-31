"""Wan2.2 I2V Trainer with FSDP2 + Distributed Checkpoint (DCP)."""

import math
import os
from collections import deque
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
from src.trainer.flops import MFUMonitor, compute_wan_seq_len, estimate_wan_forward_flops, get_gpu_peak_flops_bf16

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    """Linear warmup + cosine decay."""
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _collate(batch):
    collated = {}
    sample = batch[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            collated[key] = torch.stack([x[key] for x in batch])
    if "prompt" in sample:
        collated["prompt"] = [x["prompt"] for x in batch]
    if "index" in sample:
        collated["index"] = torch.tensor([x["index"] for x in batch], dtype=torch.long)
    return collated


def _to_model_pixels(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move uint8 pixels to GPU and normalize to [-1, 1] in bf16."""
    return tensor.to(device=device, dtype=torch.bfloat16, non_blocking=True).div(127.5).sub(1.0)


def _shard_transformer(module, mesh, mp_policy):
    """Apply FSDP2 fully_shard per-block then top-level."""
    for block in module.blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp_policy)
    fully_shard(module, mesh=mesh, mp_policy=mp_policy)


def _setup_loguru(rank: int) -> None:
    """Configure loguru: Rich sink on rank 0, silence other ranks."""
    from rich.console import Console
    from rich.text import Text

    logger.remove()
    if rank == 0:
        console = Console(stderr=True)

        _LEVEL_STYLES = {
            "DEBUG": "dim cyan",
            "INFO": "bold green",
            "SUCCESS": "bold green",
            "WARNING": "bold yellow",
            "ERROR": "bold red",
            "CRITICAL": "bold white on red",
        }

        def _rich_sink(message):
            record = message.record
            level = record["level"].name
            style = _LEVEL_STYLES.get(level, "")
            ts = record["time"].strftime("%H:%M:%S")

            line = Text()
            line.append(ts, style="dim")
            line.append(" | ", style="dim")
            line.append(f"{level:<8}", style=style)
            line.append(" | ", style="dim")
            line.append(str(record["message"]))
            console.print(line)

        logger.add(_rich_sink, level="INFO")
    else:
        logger.disable("src")


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class EMA:
    """Exponential Moving Average of model parameters.

    Works with FSDP2: each rank maintains EMA of its local parameter shards.
    Does not support resharding (changing world_size on resume).
    """

    def __init__(self, params: list[torch.nn.Parameter], decay: float):
        self.decay = decay
        self.params = params
        self.shadow = [p.data.clone() for p in params]

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params, strict=True):
            s.lerp_(p.data, 1 - self.decay)

    def state_dict(self) -> list[torch.Tensor]:
        return [s.clone() for s in self.shadow]

    def load_state_dict(self, state: list[torch.Tensor]):
        for s, loaded in zip(self.shadow, state, strict=True):
            s.copy_(loaded)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


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

        _setup_loguru(self.rank)

        logger.info("World size: {}", self.world_size)

        # ---- Model ----
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
        self.model = WanI2VForTraining(
            cfg.model_path,
            lora_config=lora_cfg,
            train_experts=cfg.train_experts,
            train_text_encoder=cfg.train_text_encoder,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )

        # Move frozen parts to GPU
        self.model.text_encoder.to(self.device)
        self.model.vae.to(self.device)

        # ---- Dataset / cache ----
        raw_dataset = I2VDataset(
            json_path=cfg.dataset_json,
            num_frames=cfg.num_frames,
            height=cfg.height,
            width=cfg.width,
            fps=cfg.fps,
        )
        dataset = raw_dataset

        # FSDP2 shard trainable modules
        mesh = init_device_mesh("cuda", (self.world_size,))
        dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        mp_policy = MixedPrecisionPolicy(
            param_dtype=dtype_map[cfg.param_dtype],
            reduce_dtype=dtype_map[cfg.reduce_dtype],
        )
        if cfg.train_text_encoder:
            fully_shard(self.model.text_encoder, mesh=mesh, mp_policy=mp_policy)
        if self.model.transformer is not None:
            _shard_transformer(self.model.transformer, mesh, mp_policy)
        if self.model.transformer_2 is not None:
            _shard_transformer(self.model.transformer_2, mesh, mp_policy)
        self.sync_modules = [
            module
            for module in [
                self.model.text_encoder if cfg.train_text_encoder else None,
                self.model.transformer,
                self.model.transformer_2,
            ]
            if module is not None
        ]

        # ---- torch.compile (after FSDP sharding) ----
        if cfg.torch_compile:
            compile_kwargs = {"backend": cfg.torch_compile_backend}
            if cfg.torch_compile_mode is not None:
                compile_kwargs["mode"] = cfg.torch_compile_mode
            # Frozen modules (inference-only, static shapes)
            self.model.vae = torch.compile(self.model.vae, **compile_kwargs)
            logger.info("Compiled vae")
            if not cfg.train_text_encoder:
                self.model.text_encoder = torch.compile(self.model.text_encoder, **compile_kwargs)
                logger.info("Compiled text_encoder")
            # Trainable transformers
            if self.model.transformer is not None:
                self.model.transformer = torch.compile(self.model.transformer, **compile_kwargs)
                logger.info("Compiled transformer")
            if self.model.transformer_2 is not None:
                self.model.transformer_2 = torch.compile(self.model.transformer_2, **compile_kwargs)
                logger.info("Compiled transformer_2")
            logger.info(
                "torch.compile enabled (backend={}, mode={})", cfg.torch_compile_backend, cfg.torch_compile_mode
            )

        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, seed=cfg.seed)
        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=_collate,
            drop_last=True,
        )
        if cfg.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = cfg.persistent_workers
            dataloader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        self.dataloader = DataLoader(**dataloader_kwargs)
        self.sampler = sampler

        # ---- Optimizer (one per FSDP module for correct DCP state_dict mapping) ----
        self.optimizer_te = None
        self.optimizer_1 = None
        self.optimizer_2 = None
        self.params = []
        total_params = 0
        optim_kwargs = dict(lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))

        if cfg.train_text_encoder:
            params_te = [p for p in self.model.text_encoder.parameters() if p.requires_grad]
            self.params.extend(params_te)
            total_params += sum(p.numel() for p in self.model.text_encoder.parameters())
            self.optimizer_te = torch.optim.AdamW(params_te, **optim_kwargs)

        if self.model.transformer is not None:
            params_1 = [p for p in self.model.transformer.parameters() if p.requires_grad]
            self.params.extend(params_1)
            total_params += sum(p.numel() for p in self.model.transformer.parameters())
            self.optimizer_1 = torch.optim.AdamW(params_1, **optim_kwargs)

        if self.model.transformer_2 is not None:
            params_2 = [p for p in self.model.transformer_2.parameters() if p.requires_grad]
            self.params.extend(params_2)
            total_params += sum(p.numel() for p in self.model.transformer_2.parameters())
            self.optimizer_2 = torch.optim.AdamW(params_2, **optim_kwargs)

        trainable_count = sum(p.numel() for p in self.params)
        logger.info(
            "Trainable: {:.1f}M / {:.1f}M ({:.2f}%)",
            trainable_count / 1e6,
            total_params / 1e6,
            100 * trainable_count / total_params,
        )

        # ---- EMA ----
        self.ema = None
        if cfg.ema_decay > 0:
            self.ema = EMA(self.params, decay=cfg.ema_decay)
            logger.info("EMA enabled (decay={}, {} shadow params)", cfg.ema_decay, len(self.params))

        self.total_steps = cfg.num_epochs * len(self.dataloader) // cfg.gradient_accumulation_steps
        logger.info(
            "Dataset: {} samples, {} batches/epoch, {} total optimizer steps",
            len(dataset),
            len(self.dataloader),
            self.total_steps,
        )

        # ---- DCP state ----
        self.optimizers = [opt for opt in [self.optimizer_te, self.optimizer_1, self.optimizer_2] if opt is not None]
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
        if cfg.resume_from:
            self._load_checkpoint(cfg.resume_from)

    # ------------------------------------------------------------------
    # MFU setup
    # ------------------------------------------------------------------

    def _setup_mfu(self) -> MFUMonitor | None:
        """Pre-compute FLOPs and create MFU monitor. Returns None if GPU is unrecognized."""
        gpu_peak = get_gpu_peak_flops_bf16()
        if gpu_peak is None:
            return None

        # Weighted forward FLOPs across active experts by routing probability
        bi = self.model.boundary_idx
        N = self.model.num_train_timesteps
        experts = []
        if self.model.transformer is not None:
            prob = bi / N if self.cfg.train_experts == "both" else 1.0
            experts.append((prob, self.model.transformer))
        if self.model.transformer_2 is not None:
            prob = (N - bi) / N if self.cfg.train_experts == "both" else 1.0
            experts.append((prob, self.model.transformer_2))

        weighted_fwd_flops = 0.0
        seq_len = 0
        for prob, t in experts:
            t_cfg = t.config
            seq_len = compute_wan_seq_len(
                self.cfg.num_frames,
                self.cfg.height,
                self.cfg.width,
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

        # Training = forward(1x) + backward(2x) per sample, times micro-batches per optimizer step
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

        # On resume, skip batches already processed in the interrupted epoch.
        # DistributedSampler with the same seed+epoch produces the same order,
        # so skipping reproduces the exact same data sequence.
        resume_batch_idx = self.train_state.batch_idx

        if self.rank == 0:
            from rich.console import Console
            from rich.live import Live
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
            from rich.table import Table as RichTable

            def build_metrics_table(rows):
                table = RichTable(show_header=True, expand=True, box=None, padding=(0, 1))
                table.add_column("step", style="bold")
                table.add_column("epoch", style="blue")
                table.add_column("loss", style="yellow")
                table.add_column("lr", style="cyan")
                table.add_column("grad_norm")
                table.add_column("mfu", style="green")
                for row in rows:
                    table.add_row(*row)
                return table

            console = Console()
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Training"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("<"),
                TimeRemainingColumn(),
                console=console,
                expand=True,
            )
            task_id = progress.add_task("Training", total=self.total_steps, completed=global_step)

            # Metrics table (rendered below progress bar, keep only the latest rows)
            metrics_rows = deque(maxlen=5)
            metrics_table = build_metrics_table(metrics_rows)

            from rich.console import Group

            live = Live(Group(progress, metrics_table), console=console, refresh_per_second=10)
            live.start()
        else:
            progress = None
            live = None

        for epoch in range(start_epoch, cfg.num_epochs):
            self.sampler.set_epoch(epoch)
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx < resume_batch_idx:
                    continue  # skip already-processed batches
                resume_batch_idx = 0  # only skip in the first (resumed) epoch

                is_last_micro_step = (batch_idx + 1) % cfg.gradient_accumulation_steps == 0
                self._set_requires_gradient_sync(is_last_micro_step)
                loss = self._train_step(batch)
                scaled_loss = loss / cfg.gradient_accumulation_steps
                scaled_loss.backward()

                if is_last_micro_step:
                    self._last_grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm).item()

                    lr = _cosine_lr(global_step, cfg.warmup_steps, self.total_steps, cfg.learning_rate)
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
                        progress.update(task_id, completed=global_step)

                        metrics_rows.append(
                            (
                                str(global_step),
                                str(epoch),
                                f"{loss.item():.4f}",
                                f"{lr:.2e}",
                                f"{self._last_grad_norm:.4f}",
                                f"{mfu:.1%}" if mfu is not None else "-",
                            )
                        )
                        metrics_table = build_metrics_table(metrics_rows)
                        live.update(Group(progress, metrics_table), refresh=True)

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
                    elif progress is not None:
                        progress.update(task_id, completed=global_step)

                    if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                        self.train_state.step = global_step
                        self.train_state.epoch = epoch
                        self.train_state.batch_idx = batch_idx + 1
                        self._save_checkpoint(output_dir / f"checkpoint-{global_step}")

            # End-of-epoch save (batch_idx=0 means start fresh next epoch)
            self.train_state.step = global_step
            self.train_state.epoch = epoch + 1
            self.train_state.batch_idx = 0
            self._save_checkpoint(output_dir / f"checkpoint-epoch{epoch}")
            logger.info("Epoch {} done.", epoch)

        if live is not None:
            live.stop()

        if self.use_wandb:
            import wandb

            wandb.finish()
        dist.destroy_process_group()

    def _train_step(self, batch: dict) -> torch.Tensor:
        """Single forward pass: encode frozen inputs, compute loss."""
        prompt_embeds = self.model.encode_text(batch["prompt"], self.device)
        video = _to_model_pixels(batch["video"], self.device)  # (B, C, T, H, W)
        image = _to_model_pixels(batch["image"], self.device)  # (B, C, H, W)
        video_latents = self.model.encode_video(video)
        condition = self.model.prepare_condition(image, self.cfg.num_frames, self.cfg.height, self.cfg.width)

        return self.model.compute_loss(video_latents, condition, prompt_embeds)

    def _set_requires_gradient_sync(self, requires_gradient_sync: bool) -> None:
        for module in self.sync_modules:
            if hasattr(module, "set_requires_gradient_sync"):
                module.set_requires_gradient_sync(requires_gradient_sync, recurse=True)

    # ------------------------------------------------------------------
    # DCP checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: Path):
        """Save with DCP. All ranks participate; each writes its own shards."""
        dcp.save({"train_state": self.train_state}, checkpoint_id=str(path))
        # Save EMA shards (each rank saves its own; does not support resharding)
        if self.ema is not None:
            ema_dir = path / "ema"
            ema_dir.mkdir(exist_ok=True)
            torch.save(self.ema.state_dict(), ema_dir / f"rank{self.rank}.pt")
        # Also save LoRA adapter weights in portable PEFT format (rank 0 only)
        if self.rank == 0 and self.model.lora_config is not None:
            self.model.save_lora(str(path / "lora"))
        if self.rank == 0:
            logger.info("Saved DCP checkpoint to {}", path)
        dist.barrier()

    def _load_checkpoint(self, path: str):
        """Load with DCP. Supports resharding across different world sizes."""
        logger.info("Resuming from {} ...", path)
        dcp.load({"train_state": self.train_state}, checkpoint_id=path)
        if self.ema is not None:
            ema_file = Path(path) / "ema" / f"rank{self.rank}.pt"
            if ema_file.exists():
                self.ema.load_state_dict(torch.load(ema_file, map_location=self.device, weights_only=True))
                logger.info("Loaded EMA state from {}", ema_file)
        logger.info(
            "Resumed at step={} epoch={} batch_idx={}",
            self.train_state.step,
            self.train_state.epoch,
            self.train_state.batch_idx,
        )
