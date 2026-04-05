"""Training configuration."""

from typing import Literal

from pydantic import BaseModel


class TrainConfig(BaseModel):
    # Model
    model_path: str = "storage/models/Wan2.2-I2V-A14B-Diffusers"

    # Data
    dataset_json: str = "data/train.json"
    num_frames: int | None = None  # override dataset JSON config; default 81
    max_area: int | None = None  # override dataset JSON config; default 480*832
    height: int | None = None  # override dataset JSON config; fixed height
    width: int | None = None  # override dataset JSON config; fixed width
    fps: int | None = None  # override dataset JSON config; default 16
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

    # Liger Kernel (fused Triton kernels)
    use_liger_kernel: bool = False

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
    reset_dataloader: bool | None = None  # None = auto (reset when resume_from is set, keep when auto-resuming from output_dir)

    # Logging
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # GRPO (set grpo_group_size > 0 to enable Flow-GRPO training)
    grpo_group_size: int | None = None  # G: number of samples per prompt. None = SFT mode
    grpo_sample_batch_size: int = 1  # how many G samples to batch together (tune for GPU memory)
    grpo_num_sampling_steps: int = 10  # T: denoising steps during SDE sampling
    grpo_clip_range: float = 1e-3  # PPO clipping epsilon
    grpo_kl_coeff: float = 0.004  # beta: KL penalty coefficient against reference policy
    grpo_sde_noise_scale: float = 0.7  # a in sigma_t = a * sqrt(t / (1-t))
    grpo_sde_sigma_min: float = 0.0  # noise floor for SDE std
    grpo_sde_sigma_max: float = 1.0  # noise ceiling for SDE std
    grpo_adv_clip_max: float = 5.0  # clamp advantages to [-max, max]
    grpo_reward_fn: str = "neg_loss"  # reward function name
    grpo_cfg_scale: float = 1.0  # classifier-free guidance scale during sampling

    # COS (Chain-of-Step) piecewise flow matching
    cos_tau_sigma: float = 0.5  # piecewise boundary in sigma space (independent of MoE boundary)
    cos_boundary_noise_std: float = 0.02  # Gaussian perturbation std for x_tau in low stage
    cos_use_standard_formula: bool = False  # ablation: use standard sigma formula per segment (discontinuous)
    cos_path_type: Literal["linear", "cosine", "cubic_hermite", "smooth_blend", "quadratic_bezier", "target_linear", "target_cosine"] = "linear"
    cos_smooth_blend_delta: float = 0.05  # half-width of blending window (only for smooth_blend path)

    # Trainer selection
    trainer: Literal["i2v", "cos"] = "i2v"  # "cos" for Chain-of-Step piecewise flow matching

    # Expert parallel: split MoE experts across GPU sub-groups
    expert_parallel: bool = False  # each expert gets world_size/2 GPUs with independent FSDP
