# COS Trainer: Chain-of-Step Piecewise Flow Matching

## Motivation

Standard SFT flow matching trains a model to denoise along a **straight line** from pure noise to a target video:

```
noise ────────────────────────────> final_video
```

COS (Chain-of-Step) training replaces this with a **piecewise path** that passes through an intermediate "search" state:

```
noise ──────────> search_video ──────────> final_video
       high-noise stage         low-noise stage
```

The key insight is that the model learns a structured denoising trajectory: the high-noise stage develops coarse search-like structure (e.g., BFS frontier expansion in a maze), and the low-noise stage refines that into the final execution video (e.g., a ball moving along the solved path).

This is analogous to **Chain-of-Thought** reasoning but in video generation: the model first "thinks" (searches), then "acts" (executes).

## Mathematical Formulation

### Background: Standard Flow Matching

In standard flow matching, given a clean video latent `x_0` and noise `z ~ N(0,I)`, the interpolation path is:

```
x_sigma = sigma * z + (1 - sigma) * x_0
```

The model predicts the velocity `v = dx/dsigma = z - x_0`, and the ODE solver integrates from `sigma=1` (noise) to `sigma=0` (clean).

### COS: Piecewise Path

COS defines two segments joined at a boundary `tau` in sigma space (default `tau = 0.5`):

**High-noise stage** (`sigma >= tau`): noise -> search_video

```
s_h = (sigma - tau) / (1 - tau)          # local parameter in [0, 1]
x_sigma = s_h * z + (1 - s_h) * x_tau   # interpolation
v_target = (z - x_tau) / (1 - tau)       # dx/dsigma
```

**Low-noise stage** (`sigma < tau`): search_video -> final_video

```
s_l = sigma / tau                                    # local parameter in [0, 1]
x_sigma = s_l * x_tau_tilde + (1 - s_l) * x_final   # interpolation
v_target = (x_tau_tilde - x_final) / tau              # dx/dsigma
```

where `x_tau_tilde = x_tau + eps` is a small Gaussian perturbation for robustness.

### Path Properties

| Property | Value |
|----------|-------|
| `sigma = 1` | `x_sigma = z` (pure noise) |
| `sigma = tau` (from above) | `x_sigma = x_tau` (search video) |
| `sigma = tau` (from below) | `x_sigma = x_tau_tilde` (search video + small perturbation) |
| `sigma = 0` | `x_sigma = x_final` (final video) |

The path is **continuous at the boundary** (up to `eps` perturbation), so the ODE solver can integrate smoothly from `sigma=1` to `sigma=0` without special handling.

### Why `tau = 0.5`?

With `tau = 0.5`, the velocity magnitudes in both stages are balanced:

```
|v_high| = |z - x_tau| / (1 - 0.5) = 2 * |z - x_tau|
|v_low|  = |x_tau - x_final| / 0.5 = 2 * |x_tau - x_final|
```

Since `|z - x_tau|` and `|x_tau - x_final|` are similar in magnitude (both are differences between independent high-dimensional vectors), the ratio is approximately **1:1**. This avoids the training instability that would occur if we aligned the piecewise boundary with the MoE boundary (`tau = 0.978`), where the high-stage velocity would be ~45x larger.

### Relationship to MoE Routing

Wan2.2 A14B uses two transformer experts routed by noise level:

- **transformer** (high-noise expert): `timestep >= 900` (shifted `sigma >= ~0.978`)
- **transformer_2** (low-noise expert): `timestep < 900`

The COS piecewise boundary (`tau = 0.5`) is **independent** from the MoE routing boundary (`sigma ~= 0.978`). Both experts participate in both COS stages. This is by design: we want balanced velocity magnitudes while keeping the proven MoE routing unchanged.

## Data Format

Each training sample requires two videos and optionally a condition image:

```json
[
    {
        "video": "path/to/final_video.mp4",
        "search_video": "path/to/search_video.mp4",
        "image": "path/to/condition_image.jpg",
        "prompt": "A ball navigates through a maze from start to end."
    }
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `video` | Yes | Final execution video (the "answer") |
| `search_video` | Yes | Intermediate search video (the "reasoning") |
| `image` | No | Condition image (defaults to first frame of `video`) |
| `prompt` | Yes | Text description |

Both videos are encoded to the same latent space via the frozen VAE. They must depict the same scene/task but at different stages of completion.

### Example: Maze Solving

- **search_video**: Shows BFS frontier expansion from start and end, meeting in the middle
- **video** (final): Shows a ball moving along the solved shortest path
- **image**: The unsolved maze

## Usage

### Training

```bash
torchrun --nproc_per_node=8 -m src.cli.train_cos --config configs/train_cos.yaml
```

### Configuration

COS-specific fields (in addition to all standard training fields):

| Field | Default | Description |
|-------|---------|-------------|
| `cos_tau_sigma` | `0.5` | Piecewise boundary in sigma space |
| `cos_boundary_noise_std` | `0.02` | Gaussian perturbation std for `x_tau` in low stage |
| `cos_use_standard_formula` | `false` | Ablation: use standard sigma formula per segment |
| `expert_parallel` | `false` | Split MoE experts across GPU sub-groups |

Example config (`configs/train_cos.yaml`):

```yaml
model_path: storage/models/Wan2.2-I2V-A14B-Diffusers
dataset_json: data/train_cos.json
output_dir: storage/checkpoints/cos

num_frames: 81
height: 480
width: 832

batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-5

cos_tau_sigma: 0.5
cos_boundary_noise_std: 0.02

wandb_project: wan-cos
```

### Inference

Inference uses the **standard ODE solver** with no modifications. The learned velocity field naturally guides the trajectory through the search-like intermediate region:

```bash
python -m src.cli.infer_i2v \
    --image maze.jpg \
    --prompt "A ball navigates through a maze." \
    --output result.mp4
```

To visualize intermediate states, decode latents at various sigma values during the ODE integration.

## Debug Logging

Every `log_steps`, the trainer reports per-stage statistics:

```
step=100/1000 epoch=0 loss=0.4321 lr=1.00e-05 grad_norm=0.8765 mfu=12.3% eta=2h30m (0.50 it/s)
  COS: n_high=2 n_low=2 tnorm_high=252.3 tnorm_low=255.1 sigma=[0.012, 0.987] mean=0.501
```

| Metric | Description |
|--------|-------------|
| `n_high` / `n_low` | Number of samples in each stage (should be roughly equal) |
| `tnorm_high` / `tnorm_low` | Mean L2 norm of velocity targets (should be similar magnitude) |
| `sigma` range and mean | Distribution of sampled sigma values |

These are also logged to wandb under the `cos/` prefix.

## Ablation: Solution B (`cos_use_standard_formula`)

Setting `cos_use_standard_formula: true` switches to an alternative formulation where each segment uses the standard flow matching formula independently:

```
High: x_sigma = sigma * z + (1 - sigma) * x_tau,         target = z - x_tau
Low:  x_sigma = sigma * x_tau_tilde + (1 - sigma) * x_final,  target = x_tau_tilde - x_final
```

This has balanced velocity magnitudes by construction but introduces a **discontinuity at the boundary**: the high stage at `sigma=tau` gives `x = tau*z + (1-tau)*x_tau` (still noisy), while the low stage at `sigma=tau` gives `x = tau*x_tau + (1-tau)*x_final` (a video blend). The ODE solver must bridge this gap implicitly.

Use this for controlled comparison against Solution A.

## Dual Expert Training

By default, `compute_cos_loss` runs **one dedicated forward pass per MoE expert** every step. With `train_experts='both'`, both the high-noise expert (transformer) and low-noise expert (transformer_2) are guaranteed to receive training signal on every step, regardless of the sigma distribution.

Each expert samples sigma only in its own MoE routing range:
- **transformer**: sigma indices `[0, boundary_idx)` (high-noise, timestep >= 900)
- **transformer_2**: sigma indices `[boundary_idx, 1000)` (low-noise, timestep < 900)

Within each expert's sigma range, the COS piecewise path is constructed as usual (high stage if sigma >= tau, low stage otherwise).

## Expert Parallel Mode

Setting `expert_parallel: true` splits the GPU fleet into two groups, each training one expert independently:

```yaml
expert_parallel: true
train_experts: both  # required
```

```
Node 0: GPU 0,1,2,3 → high-noise expert (transformer)
Node 1: GPU 4,5,6,7 → low-noise expert (transformer_2)
```

### Benefits

- **Memory**: Each GPU only holds shards of one expert (half the trainable parameters)
- **Communication**: FSDP gradient sync within group only (4 GPUs instead of 8)
- **Throughput**: Each group iterates the dataset independently, doubling data diversity

### Data strategy

| Mode | Data per group | Implementation |
|------|---------------|----------------|
| COS | **Different** items | Different sampler seed per group |
| SFT | **Same** items | Same sampler seed (default) |

For COS, each group independently iterates through the dataset with a per-group seed. For SFT (I2VTrainer), both groups see the same items (matching dp_rank positions get identical indices).

### GPU placement

For optimal communication, place each expert group's GPUs on the **same node**. With `torchrun`, ranks are assigned contiguously per node by default, so the first half of GPUs (one node) becomes group 0 and the second half becomes group 1.

For multi-node setups with 8 GPUs per node:
- 2 nodes (16 GPUs): group 0 on node 0 (ranks 0-7), group 1 on node 1 (ranks 8-15)
- 4 nodes (32 GPUs): group 0 on nodes 0-1 (ranks 0-15), group 1 on nodes 2-3 (ranks 16-31)

### Checkpointing

In expert parallel mode, each group saves independently to subdirectories:

```
checkpoint-100/
  high/    # DCP shards from group 0
  low/     # DCP shards from group 1
```

Resume detects the subdirectory structure automatically. Loading a non-EP checkpoint into an EP setup also works (each group loads the relevant expert).

## Architecture

```
src/
├── cli/
│   └── train_cos.py              # COS training entry point
├── data/
│   └── i2v_dataset.py            # Dataset (loads search_video when present)
├── models/
│   └── wan_i2v.py                # compute_cos_loss() — per-expert dedicated sigma
└── trainer/
    ├── base_trainer.py            # Expert parallel: sub-mesh FSDP, sampler hooks
    ├── config.py                  # cos_*, expert_parallel
    └── cos_trainer.py             # COSTrainer with dual expert + EP sampler override
```

No changes to the model backbone, VAE, text encoder, MoE routing, or ODE solver.

## Expected Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| High-stage loss much larger than low-stage | `cos_tau_sigma` too close to 1 | Decrease toward 0.5 |
| Generated video jumps directly to final, skipping search | Search video too similar to final | Make search video visually distinct |
| Blurry boundary region during inference | Boundary handoff instability | Increase `cos_boundary_noise_std` slightly (e.g., 0.05) |
| ODE solver diverges | Step size too large for curved path | Increase inference steps (50-100) |
| Expert parallel: groups desync | Different dataset sizes or filtering | Use same dataset for both groups |
