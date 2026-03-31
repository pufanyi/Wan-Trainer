"""Wan2.2 I2V training entry point.

Usage:
    torchrun --nproc_per_node=2 -m src.cli.train_i2v --config configs/train_i2v.json
"""

import argparse
import json
from pathlib import Path

from src.trainer import I2VTrainer, TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 I2V Training")
    parser.add_argument("--config", type=str, default=None, help="JSON config file")
    # CLI overrides (auto-generated from TrainConfig fields)
    for f, field_obj in TrainConfig.__dataclass_fields__.items():
        default = field_obj.default
        if default is None:
            parser.add_argument(f"--{f}", type=str, default=None)
        else:
            parser.add_argument(f"--{f}", type=type(default), default=None)
    args = parser.parse_args()

    # Build config: defaults -> JSON -> CLI
    cfg_dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text())
    for f in TrainConfig.__dataclass_fields__:
        v = getattr(args, f, None)
        if v is not None:
            cfg_dict[f] = v

    cfg = TrainConfig.from_dict(cfg_dict)
    trainer = I2VTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
