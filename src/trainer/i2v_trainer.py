"""Wan2.2 I2V Trainer with FSDP2 + Distributed Checkpoint (DCP)."""

import logging
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, DistributedSampler

from src.data.i2v_dataset import I2VDataset
from src.models.wan_i2v import LoRATrainConfig, WanI2VForTraining
from src.trainer.checkpoint import TrainState
from src.trainer.config import TrainConfig
from src.trainer.flops import MFUMonitor, compute_wan_seq_len, estimate_wan_forward_flops, get_gpu_peak_flops_bf16

logger = logging.getLogger(__name__)


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
    return {
        "video": torch.stack([x["video"] for x in batch]),
        "image": torch.stack([x["image"] for x in batch]),
        "prompt": [x["prompt"] for x in batch],
    }


def _shard_transformer(module, mesh, mp_policy):
    """Apply FSDP2 fully_shard per-block then top-level."""
    for block in module.blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp_policy)
    fully_shard(module, mesh=mesh, mp_policy=mp_policy)


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

        if self.rank == 0:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
        else:
            logging.basicConfig(level=logging.WARNING)

        logger.info("World size: %d", self.world_size)

        # ---- Model ----
        lora_cfg = (
            LoRATrainConfig(rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
            if cfg.lora_rank > 0
            else None
        )
        logger.info(
            "Loading model from %s (lora_rank=%d, experts=%s) ...",
            cfg.model_path,
            cfg.lora_rank,
            cfg.train_experts,
        )
        self.model = WanI2VForTraining(
            cfg.model_path,
            lora_config=lora_cfg,
            train_experts=cfg.train_experts,
            train_text_encoder=cfg.train_text_encoder,
        )

        # Move frozen parts to GPU
        self.model.text_encoder.to(self.device)
        self.model.vae.to(self.device)

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

        # ---- Dataset ----
        dataset = I2VDataset(
            json_path=cfg.dataset_json,
            num_frames=cfg.num_frames,
            height=cfg.height,
            width=cfg.width,
            fps=cfg.fps,
        )
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, seed=cfg.seed)
        self.dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=_collate,
            drop_last=True,
        )
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
            "Trainable: %.1fM / %.1fM (%.2f%%)",
            trainable_count / 1e6,
            total_params / 1e6,
            100 * trainable_count / total_params,
        )

        self.total_steps = cfg.num_epochs * len(self.dataloader) // cfg.gradient_accumulation_steps
        logger.info(
            "Dataset: %d samples, %d batches/epoch, %d total optimizer steps",
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

        t = self.model.transformer or self.model.transformer_2
        t_cfg = t.config

        seq_len = compute_wan_seq_len(
            self.cfg.num_frames,
            self.cfg.height,
            self.cfg.width,
            patch_size=tuple(t_cfg.patch_size),
            vae_temporal_factor=self.model.vae_scale_factor_temporal,
            vae_spatial_factor=self.model.vae_scale_factor_spatial,
        )
        forward_flops = estimate_wan_forward_flops(
            num_layers=t_cfg.num_layers,
            num_heads=t_cfg.num_attention_heads,
            head_dim=t_cfg.attention_head_dim,
            ffn_dim=t_cfg.ffn_dim,
            seq_len=seq_len,
        )
        # Training = forward(1x) + backward(2x) per sample, times micro-batches per optimizer step
        flops_per_step = 3 * forward_flops * self.cfg.batch_size * self.cfg.gradient_accumulation_steps

        if self.rank == 0:
            logger.info(
                "MFU monitor: seq_len=%d, forward=%.2e FLOPs/sample, step=%.2e FLOPs, GPU=%s (%.0f TFLOPS bf16)",
                seq_len,
                forward_flops,
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

        for epoch in range(start_epoch, cfg.num_epochs):
            self.sampler.set_epoch(epoch)
            for opt in self.optimizers:
                opt.zero_grad()

            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx < resume_batch_idx:
                    continue  # skip already-processed batches
                resume_batch_idx = 0  # only skip in the first (resumed) epoch

                loss = self._train_step(batch)
                scaled_loss = loss / cfg.gradient_accumulation_steps
                scaled_loss.backward()

                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    self._last_grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm).item()

                    lr = _cosine_lr(global_step, cfg.warmup_steps, self.total_steps, cfg.learning_rate)
                    for opt in self.optimizers:
                        for pg in opt.param_groups:
                            pg["lr"] = lr
                        opt.step()
                        opt.zero_grad()
                    global_step += 1
                    if self.mfu_monitor is not None:
                        self.mfu_monitor.step()

                    if self.rank == 0 and global_step % cfg.log_steps == 0:
                        mfu = self.mfu_monitor.flush() if self.mfu_monitor is not None else None
                        mfu_str = f" mfu={mfu:.1%}" if mfu is not None else ""
                        logger.info(
                            "epoch=%d step=%d/%d loss=%.4f lr=%.2e%s",
                            epoch,
                            global_step,
                            self.total_steps,
                            loss.item(),
                            lr,
                            mfu_str,
                        )
                        if self.use_wandb:
                            import wandb

                            metrics = {
                                "train/loss": loss.item(),
                                "train/lr": lr,
                                "train/epoch": epoch,
                                "train/grad_norm": self._last_grad_norm,
                            }
                            if mfu is not None:
                                metrics["train/mfu"] = mfu
                            wandb.log(metrics, step=global_step)

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
            if self.rank == 0:
                logger.info("Epoch %d done.", epoch)

        if self.use_wandb:
            import wandb

            wandb.finish()
        dist.destroy_process_group()

    def _train_step(self, batch: dict) -> torch.Tensor:
        """Single forward pass: encode frozen inputs, compute loss."""
        video = batch["video"].to(self.device)  # (B, C, T, H, W)
        image = batch["image"].to(self.device)  # (B, C, H, W)
        prompts = batch["prompt"]

        prompt_embeds = self.model.encode_text(prompts, self.device)
        video_latents = self.model.encode_video(video)
        condition = self.model.prepare_condition(image, self.cfg.num_frames, self.cfg.height, self.cfg.width)

        return self.model.compute_loss(video_latents, condition, prompt_embeds)

    # ------------------------------------------------------------------
    # DCP checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: Path):
        """Save with DCP. All ranks participate; each writes its own shards."""
        dcp.save({"train_state": self.train_state}, checkpoint_id=str(path))
        # Also save LoRA adapter weights in portable PEFT format (rank 0 only)
        if self.rank == 0 and self.model.lora_config is not None:
            self.model.save_lora(str(path / "lora"))
        if self.rank == 0:
            logger.info("Saved DCP checkpoint to %s", path)
        dist.barrier()

    def _load_checkpoint(self, path: str):
        """Load with DCP. Supports resharding across different world sizes."""
        logger.info("Resuming from %s ...", path)
        dcp.load({"train_state": self.train_state}, checkpoint_id=path)
        logger.info(
            "Resumed at step=%d epoch=%d batch_idx=%d",
            self.train_state.step,
            self.train_state.epoch,
            self.train_state.batch_idx,
        )
