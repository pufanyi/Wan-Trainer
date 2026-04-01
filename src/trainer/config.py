"""Training configuration."""

from typing import Literal

from pydantic import BaseModel


class TrainConfig(BaseModel):
    # Model
    model_path: str = "storage/models/Wan2.2-I2V-A14B-Diffusers"

    # Data
    dataset_json: str = "data/train.json"
    num_frames: int = 81
    max_area: int = 480 * 832  # height * width budget; actual h/w derived per-video
    fps: int = 16
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Training
    output_dir: str = "storage/checkpoints"
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    save_steps: int = 500
    log_steps: int = 10
    seed: int = 42
    ema_decay: float = 0.0  # 0 = disabled; typical value: 0.9999

    # Which components to train
    train_experts: Literal["both", "high", "low"] = "both"
    train_text_encoder: bool = False
    gradient_checkpointing: bool = True

    # FSDP2 mixed precision
    param_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    reduce_dtype: Literal["float32", "bfloat16"] = "float32"

    # torch.compile
    torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str | None = None  # e.g. "reduce-overhead", "max-autotune"

    # LoRA (set lora_rank > 0 to enable)
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Checkpoint
    resume_from: str | None = None
    auto_resume: bool = True  # auto-detect latest checkpoint in output_dir

    # Logging
    wandb_project: str | None = None
    wandb_run_name: str | None = None
