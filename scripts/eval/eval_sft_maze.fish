#!/usr/bin/env fish

# Evaluate SFT maze checkpoint (EMA weights).
#
# Usage:
#   fish scripts/eval/eval_sft_maze.fish <checkpoint>                        # full eval, 8 GPUs, all steps
#   fish scripts/eval/eval_sft_maze.fish <checkpoint> 1                      # first sample only
#   fish scripts/eval/eval_sft_maze.fish <checkpoint> 8 4                    # first 8 samples, 4 GPUs
#   fish scripts/eval/eval_sft_maze.fish <checkpoint> 8 4 3                  # + only render 3 steps (first, mid, last)
#
# Examples:
#   fish scripts/eval/eval_sft_maze.fish storage/checkpoints/sft_maze/checkpoint-4000
#   fish scripts/eval/eval_sft_maze.fish storage/checkpoints/sft_maze/checkpoint-4000 1 1
#   fish scripts/eval/eval_sft_maze.fish storage/checkpoints/sft_maze/checkpoint-4000 1 1 2   # only first & last step

set -l CHECKPOINT $argv[1]       # required: path to checkpoint
set -l NUM_SAMPLES $argv[2]      # optional: limit to first N samples
set -l NUM_GPUS $argv[3]         # optional: number of GPUs (default: 8)
set -l NUM_RENDER_STEPS $argv[4] # optional: render N evenly-spaced steps (default: all)

if test -z "$CHECKPOINT"
    echo "Usage: fish scripts/eval/eval_sft_maze.fish <checkpoint> [num_samples] [num_gpus] [num_render_steps]"
    exit 1
end

if test -z "$NUM_GPUS"
    set NUM_GPUS 8
end

set EVAL_JSON /mnt/umm/users/pufanyi/workspace/maze/data/train_sft.json

# Derive output dir name from checkpoint path (e.g. sft_maze/checkpoint-4000 -> sft_maze_checkpoint-4000)
set -l CKPT_NAME (string replace -a / _ (string trim -r -c / $CHECKPOINT | string replace -r '.*storage/checkpoints/' ''))
set OUTPUT_DIR eval_out/{$CKPT_NAME}_ema

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

set -l RENDER_ARGS
if test -n "$NUM_RENDER_STEPS"
    set RENDER_ARGS --num_render_steps $NUM_RENDER_STEPS
end

echo "Checkpoint:   $CHECKPOINT"
echo "Eval JSON:    $EVAL_JSON"
echo "Output:       $OUTPUT_DIR"
echo "GPUs:         $NUM_GPUS"
echo "Render steps: "(test -n "$NUM_RENDER_STEPS" && echo $NUM_RENDER_STEPS || echo "all")
echo "---"

if test $NUM_GPUS -gt 1
    uv run torchrun --nproc_per_node=$NUM_GPUS -m src.cli.eval_maze \
        --eval_json $EVAL_JSON \
        --output_dir $OUTPUT_DIR \
        --checkpoint $CHECKPOINT \
        --use_ema $RENDER_ARGS
else
    uv run python -m src.cli.eval_maze \
        --eval_json $EVAL_JSON \
        --output_dir $OUTPUT_DIR \
        --checkpoint $CHECKPOINT \
        --use_ema $RENDER_ARGS
end
