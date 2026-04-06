"""Shared utilities for training."""

import math

import torch
from loguru import logger
from torch.distributed._composable.fsdp import fully_shard


def apply_liger_rms_norm(model: torch.nn.Module) -> int:
    """Replace all torch.nn.RMSNorm modules with LigerRMSNorm (fused Triton kernel)."""
    from liger_kernel.transformers import LigerRMSNorm

    count = 0
    for _parent_name, parent in list(model.named_modules()):
        for name, module in list(parent.named_children()):
            if not isinstance(module, torch.nn.RMSNorm):
                continue
            (hidden_size,) = module.normalized_shape
            replacement = LigerRMSNorm(hidden_size, eps=module.eps, elementwise_affine=module.elementwise_affine)
            if module.weight is not None:
                replacement.weight = module.weight
            setattr(parent, name, replacement)
            count += 1
    return count


def format_eta(seconds: float) -> str:
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


def cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    """Linear warmup + cosine decay."""
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def collate(batch):
    """Collate function for I2V training batches."""
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


def to_model_pixels(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move uint8 pixels to GPU and normalize to [-1, 1] in bf16."""
    return tensor.to(device=device, dtype=torch.bfloat16, non_blocking=True).div(127.5).sub(1.0)


def shard_transformer(module, mesh, mp_policy):
    """Apply FSDP2 fully_shard per-block then top-level."""
    for block in module.blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp_policy)
    fully_shard(module, mesh=mesh, mp_policy=mp_policy)


def setup_loguru(rank: int) -> None:
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
