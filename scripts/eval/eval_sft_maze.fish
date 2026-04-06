#!/usr/bin/env fish

# Evaluate SFT maze checkpoint (EMA weights).
# Supports multiple checkpoints — the base model is loaded once,
# then each DCP checkpoint is loaded and evaluated in turn.
#
# Edit the variables below to configure the run, then:
#   fish scripts/eval/eval_sft_maze.fish

# ── Configuration ────────────────────────────────────────────────────
set CHECKPOINTS \
    storage/checkpoints/sft_maze/checkpoint-epoch0
    # storage/checkpoints/cos_maze_cos_path/checkpoint-2000 \
    # storage/checkpoints/cos_maze_linear_path/checkpoint-2000
    # storage/checkpoints/cos_maze/checkpoint-epoch0
    # storage/checkpoints/cos_maze/checkpoint-3000
    # storage/checkpoints/cos_maze/checkpoint-4000
set NUM_GPUS 1
set NUM_SAMPLES 1         # leave empty to use all samples
set NUM_RENDER_STEPS     # leave empty to render all steps
set SCHEDULER           # leave empty for default; options: euler, euler_ancestral, ddim, dpm_solver, unipc, flow_match_euler
set EVAL_JSON /mnt/umm/users/pufanyi/workspace/maze/data/train_sft.json
# ─────────────────────────────────────────────────────────────────────

# Derive output dir: single checkpoint gets a specific name, multiple uses a generic base
if test (count $CHECKPOINTS) -eq 1
    set -l CKPT_NAME (string replace -a / _ (string trim -r -c / $CHECKPOINTS[1] | string replace -r '.*storage/checkpoints/' ''))
    set OUTPUT_DIR eval_out/{$CKPT_NAME}_ema
else
    set OUTPUT_DIR eval_out/multi_ckpt
end

# If NUM_SAMPLES is set, create a trimmed json and adjust output dir
if test -n "$NUM_SAMPLES"
    set TRIMMED_JSON /mnt/umm/users/pufanyi/workspace/maze/data/eval_maze_first_{$NUM_SAMPLES}.json
    uv run python -c "
import json, sys
data = json.load(open('$EVAL_JSON'))
n = int($NUM_SAMPLES)
json.dump(data[:n], open('$TRIMMED_JSON', 'w'), indent=2)
print(f'Created trimmed eval json with {min(n, len(data))} samples')
"
    set EVAL_JSON $TRIMMED_JSON
    set OUTPUT_DIR {$OUTPUT_DIR}_n{$NUM_SAMPLES}

    # For small debug runs, cap GPUs to sample count
    if test $NUM_SAMPLES -lt $NUM_GPUS
        set NUM_GPUS $NUM_SAMPLES
    end
end

set -l EXTRA_ARGS
if test -n "$NUM_RENDER_STEPS"
    set -a EXTRA_ARGS --num_render_steps $NUM_RENDER_STEPS
end
if test -n "$SCHEDULER"
    set -a EXTRA_ARGS --scheduler $SCHEDULER
end

echo "Checkpoints:    "(count $CHECKPOINTS)" total"
for ckpt in $CHECKPOINTS
    echo "  - $ckpt"
end
echo "Eval JSON:      $EVAL_JSON"
echo "Output:         $OUTPUT_DIR"
echo "GPUs:           $NUM_GPUS"
echo "Render steps:   "(test -n "$NUM_RENDER_STEPS" && echo $NUM_RENDER_STEPS || echo "all")
echo "Scheduler:      "(test -n "$SCHEDULER" && echo $SCHEDULER || echo "default")
echo "---"

if test $NUM_GPUS -gt 1
    uv run torchrun --nproc_per_node=$NUM_GPUS -m src.cli.eval_maze \
        --eval_json $EVAL_JSON \
        --output_dir $OUTPUT_DIR \
        --checkpoint $CHECKPOINTS \
        --use_ema $EXTRA_ARGS
else
    uv run python -m src.cli.eval_maze \
        --eval_json $EVAL_JSON \
        --output_dir $OUTPUT_DIR \
        --checkpoint $CHECKPOINTS \
        --use_ema $EXTRA_ARGS
end
