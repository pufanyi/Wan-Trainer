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
        lora_cfg = LoRATrainConfig(rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout) if cfg.lora_rank > 0 else None
        logger.info("Loading model from %s (lora_rank=%d) ...", cfg.model_path, cfg.lora_rank)
        self.model = WanI2VForTraining(cfg.model_path, lora_config=lora_cfg)

        # Move frozen parts to GPU
        self.model.text_encoder.to(self.device)
        self.model.vae.to(self.device)

        # FSDP2 shard trainable transformers
        mesh = init_device_mesh("cuda", (self.world_size,))
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        _shard_transformer(self.model.transformer, mesh, mp_policy)
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

        # ---- Optimizer (one per transformer for correct DCP state_dict mapping) ----
        self.params_1 = [p for p in self.model.transformer.parameters() if p.requires_grad]
        self.params_2 = [p for p in self.model.transformer_2.parameters() if p.requires_grad]
        self.params = self.params_1 + self.params_2
        total_params = sum(p.numel() for p in self.model.transformer.parameters()) + sum(p.numel() for p in self.model.transformer_2.parameters())
        trainable_count = sum(p.numel() for p in self.params)
        logger.info("Trainable: %.1fM / %.1fM (%.2f%%)", trainable_count / 1e6, total_params / 1e6, 100 * trainable_count / total_params)
        optim_kwargs = dict(lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
        self.optimizer_1 = torch.optim.AdamW(self.params_1, **optim_kwargs)
        self.optimizer_2 = torch.optim.AdamW(self.params_2, **optim_kwargs)

        self.total_steps = cfg.num_epochs * len(self.dataloader) // cfg.gradient_accumulation_steps
        logger.info(
            "Dataset: %d samples, %d batches/epoch, %d total optimizer steps",
            len(dataset),
            len(self.dataloader),
            self.total_steps,
        )

        # ---- DCP state ----
        self.train_state = TrainState(
            transformer=self.model.transformer,
            transformer_2=self.model.transformer_2,
            optimizer_1=self.optimizer_1,
            optimizer_2=self.optimizer_2,
        )

        # ---- Resume ----
        if cfg.resume_from:
            self._load_checkpoint(cfg.resume_from)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        global_step = self.train_state.step
        start_epoch = self.train_state.epoch

        for epoch in range(start_epoch, cfg.num_epochs):
            self.sampler.set_epoch(epoch)
            self.optimizer_1.zero_grad()
            self.optimizer_2.zero_grad()

            for batch_idx, batch in enumerate(self.dataloader):
                loss = self._train_step(batch)
                scaled_loss = loss / cfg.gradient_accumulation_steps
                scaled_loss.backward()

                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)

                    lr = _cosine_lr(global_step, cfg.warmup_steps, self.total_steps, cfg.learning_rate)
                    for opt in [self.optimizer_1, self.optimizer_2]:
                        for pg in opt.param_groups:
                            pg["lr"] = lr
                        opt.step()
                        opt.zero_grad()
                    global_step += 1

                    if self.rank == 0 and global_step % cfg.log_steps == 0:
                        logger.info(
                            "epoch=%d step=%d/%d loss=%.4f lr=%.2e",
                            epoch,
                            global_step,
                            self.total_steps,
                            loss.item(),
                            lr,
                        )

                    if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                        self.train_state.step = global_step
                        self.train_state.epoch = epoch
                        self._save_checkpoint(output_dir / f"checkpoint-{global_step}")

            # End-of-epoch save
            self.train_state.step = global_step
            self.train_state.epoch = epoch + 1
            self._save_checkpoint(output_dir / f"checkpoint-epoch{epoch}")
            if self.rank == 0:
                logger.info("Epoch %d done.", epoch)

        dist.destroy_process_group()

    def _train_step(self, batch: dict) -> torch.Tensor:
        """Single forward pass: encode frozen inputs, compute loss."""
        video = batch["video"].to(self.device)   # (B, C, T, H, W)
        image = batch["image"].to(self.device)    # (B, C, H, W)
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
        logger.info("Resumed at step=%d epoch=%d", self.train_state.step, self.train_state.epoch)
