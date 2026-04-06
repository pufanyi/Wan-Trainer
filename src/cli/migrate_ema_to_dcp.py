"""Migrate legacy EMA checkpoints (ema/rank*.pt) into DCP format.

Usage:
    torchrun --nproc_per_node=8 scripts/migrate_ema_to_dcp.py \
        --config configs/train_cube_test.yaml \
        --checkpoints storage/checkpoints/cube_test/checkpoint-200 \
                      storage/checkpoints/cube_test/checkpoint-400
"""

import argparse
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import yaml
from loguru import logger
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh

from src.models.wan_i2v import LoRATrainConfig, WanI2VForTraining
from src.trainer.checkpoint import TrainState
from src.trainer.config import TrainConfig
from src.trainer.ema import EMA
from src.trainer.utils import setup_loguru, shard_transformer


def migrate_checkpoint(
    ckpt_path: Path,
    train_state: TrainState,
    ema: EMA,
    rank: int,
    device: torch.device,
):
    """Load legacy EMA, merge into DCP, remove old ema/ directory."""
    ema_dir = ckpt_path / "ema"
    ema_file = ema_dir / f"rank{rank}.pt"
    if not ema_file.exists():
        logger.warning("No legacy EMA file at {}, skipping", ema_file)
        return

    # Load train_state from DCP (needed so EMA shadow DTensors have correct sharding)
    dcp.load({"train_state": train_state}, checkpoint_id=str(ckpt_path))
    logger.info("Loaded train_state from {}", ckpt_path)

    # Reinitialize EMA shadows from loaded model weights, then overwrite with legacy data
    ema.reinitialize()
    legacy_shadows: list[torch.Tensor] = torch.load(ema_file, map_location=device, weights_only=True)
    pairs = ema._pairs
    if len(legacy_shadows) != len(pairs):
        logger.error(
            "Legacy EMA has {} tensors but current model has {} trainable params, skipping {}",
            len(legacy_shadows),
            len(pairs),
            ckpt_path,
        )
        return
    for (_, shadow), loaded in zip(pairs, legacy_shadows, strict=True):
        shadow.copy_(loaded)
    logger.info("Loaded {} legacy EMA shadows from {}", len(legacy_shadows), ema_file)

    # Save both train_state and EMA via DCP (overwrite in-place)
    dcp.save({"train_state": train_state, "ema": ema}, checkpoint_id=str(ckpt_path))
    logger.info("Saved DCP checkpoint with EMA to {}", ckpt_path)

    # Remove old ema/ directory
    dist.barrier()
    if rank == 0:
        shutil.rmtree(ema_dir)
        logger.info("Removed legacy {}", ema_dir)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy EMA to DCP format")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True, help="Checkpoint paths to migrate")
    args = parser.parse_args()

    # ---- Distributed ----
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    setup_loguru(rank)

    # ---- Config ----
    cfg_dict = yaml.safe_load(Path(args.config).read_text()) or {}
    cfg = TrainConfig(**cfg_dict)

    # ---- Model + FSDP (same setup as trainer) ----
    lora_cfg = (
        LoRATrainConfig(rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
        if cfg.lora_rank > 0
        else None
    )
    model = WanI2VForTraining(
        cfg.model_path,
        lora_config=lora_cfg,
        train_experts=cfg.train_experts,
        train_text_encoder=cfg.train_text_encoder,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    model.text_encoder.to(device)

    mesh = init_device_mesh("cuda", (world_size,))
    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    mp_policy = MixedPrecisionPolicy(
        param_dtype=dtype_map[cfg.param_dtype],
        reduce_dtype=dtype_map[cfg.reduce_dtype],
    )
    if cfg.train_text_encoder:
        fully_shard(model.text_encoder, mesh=mesh, mp_policy=mp_policy)
    if model.transformer is not None:
        shard_transformer(model.transformer, mesh, mp_policy)
    if model.transformer_2 is not None:
        shard_transformer(model.transformer_2, mesh, mp_policy)

    # ---- Optimizers (needed for TrainState) ----
    optim_kwargs = dict(lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), fused=True)
    optimizer_te = optimizer_1 = optimizer_2 = None
    if cfg.train_text_encoder:
        optimizer_te = torch.optim.AdamW(
            [p for p in model.text_encoder.parameters() if p.requires_grad], **optim_kwargs
        )
    if model.transformer is not None:
        optimizer_1 = torch.optim.AdamW(
            [p for p in model.transformer.parameters() if p.requires_grad], **optim_kwargs
        )
    if model.transformer_2 is not None:
        optimizer_2 = torch.optim.AdamW(
            [p for p in model.transformer_2.parameters() if p.requires_grad], **optim_kwargs
        )

    train_state = TrainState(
        text_encoder=model.text_encoder if cfg.train_text_encoder else None,
        transformer=model.transformer,
        transformer_2=model.transformer_2,
        optimizer_te=optimizer_te,
        optimizer_1=optimizer_1,
        optimizer_2=optimizer_2,
    )

    # ---- EMA ----
    ema_models: dict[str, torch.nn.Module] = {}
    if cfg.train_text_encoder:
        ema_models["text_encoder"] = model.text_encoder
    if model.transformer is not None:
        ema_models["transformer"] = model.transformer
    if model.transformer_2 is not None:
        ema_models["transformer_2"] = model.transformer_2
    ema = EMA(ema_models, decay=cfg.ema_decay)

    # ---- Migrate each checkpoint ----
    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            logger.warning("Checkpoint {} does not exist, skipping", ckpt_path)
            continue
        if not (ckpt_path / "ema").is_dir():
            logger.info("No legacy ema/ in {}, already migrated or no EMA", ckpt_path)
            continue
        logger.info("Migrating {} ...", ckpt_path)
        migrate_checkpoint(ckpt_path, train_state, ema, rank, device)

    dist.destroy_process_group()
    if rank == 0:
        logger.info("Migration complete.")


if __name__ == "__main__":
    main()
