# Wan-Trainer

Wan2.2 Image-to-Video fine-tuning with FSDP2 + Distributed Checkpoint.

## Setup

```bash
# Python >= 3.12 required
uv sync
```

## Data Format

JSON (or YAML) array, one entry per video:

```json
[
    {
        "video": "1_step/00000.mp4",
        "image": ["1_step_frames/00000.jpg"],
        "prompt": "A robotic arm manipulates a Rubik cube."
    }
]
```

- `video` — path to video file
- `image` — reference image (string or single-element list); omit to use the first video frame
- `prompt` — text description

Paths can be absolute or relative to the JSON file's directory.

## Training

```bash
# Full fine-tuning, 8 GPUs
torchrun --nproc_per_node=8 -m src.cli.train_i2v --config configs/train_i2v.yaml

# LoRA fine-tuning
torchrun --nproc_per_node=8 -m src.cli.train_i2v --config configs/train_i2v_lora.yaml

# CLI overrides (any config field)
torchrun --nproc_per_node=8 -m src.cli.train_i2v \
    --config configs/train_i2v.yaml \
    --learning_rate 2e-5 \
    --num_epochs 3
```

Or via the fish wrapper:

```fish
fish scripts/train/i2v.fish --config configs/train_i2v.yaml
fish scripts/train/i2v.fish --nproc 4 -- --config configs/train_i2v.yaml
```

### Resume from Checkpoint

```yaml
resume_from: storage/checkpoints/checkpoint-500
```

Resumes model weights, optimizer state, and exact data position (epoch + batch index).

## Configuration

All fields with defaults — override in YAML or via CLI flags.

| Field | Default | Description |
|-------|---------|-------------|
| `model_path` | `storage/models/Wan2.2-I2V-A14B-Diffusers` | Pretrained model directory |
| `dataset_json` | `data/train.json` | Dataset JSON path |
| `num_frames` | 81 | Frames per video |
| `height` | 480 | Video height |
| `width` | 832 | Video width |
| `fps` | 16 | Target sampling FPS |
| `num_workers` | 4 | Dataloader workers |
| `output_dir` | `storage/checkpoints` | Checkpoint output dir |
| `batch_size` | 1 | Per-GPU batch size |
| `gradient_accumulation_steps` | 4 | Micro-batches per optimizer step |
| `num_epochs` | 1 | Training epochs |
| `learning_rate` | 1e-5 | Peak learning rate |
| `weight_decay` | 0.01 | AdamW weight decay |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `warmup_steps` | 100 | Linear warmup steps |
| `save_steps` | 500 | Save every N steps (0 = epoch-end only) |
| `log_steps` | 10 | Log every N steps |
| `seed` | 42 | Random seed |
| `train_experts` | `both` | Which MoE experts: `both`, `high`, `low` |
| `train_text_encoder` | false | Unfreeze and train UMT5 text encoder |
| `param_dtype` | `bfloat16` | FSDP parameter dtype |
| `reduce_dtype` | `float32` | FSDP gradient reduction dtype |
| `lora_rank` | 0 | LoRA rank (0 = full fine-tuning) |
| `lora_alpha` | 16 | LoRA alpha |
| `lora_dropout` | 0.0 | LoRA dropout |
| `resume_from` | null | Checkpoint path to resume |

## Inference

```bash
python -m src.cli.infer_i2v \
    --image path/to/image.jpg \
    --prompt "A robotic arm manipulates a Rubik cube." \
    --output output.mp4
```

## Architecture

```
src/
├── cli/
│   ├── train_i2v.py          # Training entry point
│   └── infer_i2v.py          # Inference entry point
├── data/
│   └── i2v_dataset.py        # Dataset (video + image + prompt)
├── models/
│   └── wan_i2v.py            # Model wrapper (frozen VAE/T5 + trainable transformers)
└── trainer/
    ├── config.py              # TrainConfig (pydantic)
    ├── checkpoint.py          # DCP state management
    └── i2v_trainer.py         # FSDP2 trainer + training loop
```

Wan2.2 uses a Mixture-of-Experts architecture with two transformer denoising experts:
- **transformer** — high-noise expert (timestep >= 900)
- **transformer_2** — low-noise expert (timestep < 900)

Both are FSDP2-sharded for multi-GPU training. The VAE and text encoder are frozen by default (text encoder can be unfrozen via `train_text_encoder: true`).
