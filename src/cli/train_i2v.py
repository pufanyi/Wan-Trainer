"""Wan2.2 I2V training entry point.

Usage:
    torchrun --nproc_per_node=2 -m src.cli.train_i2v --config configs/train_i2v.yaml
"""

import argparse
from pathlib import Path

import yaml

from src.trainer import COSTrainer, I2VTrainer, TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 I2V Training")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config file")
    # CLI overrides (auto-generated from TrainConfig fields)
    for name, field_info in TrainConfig.model_fields.items():
        if field_info.annotation is bool:
            parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=None)
        elif field_info.annotation is int:
            parser.add_argument(f"--{name}", type=int, default=None)
        elif field_info.annotation is float:
            parser.add_argument(f"--{name}", type=float, default=None)
        else:
            parser.add_argument(f"--{name}", type=str, default=None)
    args = parser.parse_args()

    # Build config: defaults -> JSON -> CLI
    cfg_dict = {}
    if args.config:
        cfg_dict = yaml.safe_load(Path(args.config).read_text()) or {}
    for name in TrainConfig.model_fields:
        v = getattr(args, name, None)
        if v is not None:
            cfg_dict[name] = v

    cfg = TrainConfig(**cfg_dict)
    trainer_cls = COSTrainer if cfg.trainer == "cos" else I2VTrainer
    trainer = trainer_cls(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
