#!/usr/bin/env fish
# Wan2.2 I2V training launcher
# Usage: fish scripts/train/i2v.fish [--nproc N] [-- training args...]
#   e.g. fish scripts/train/i2v.fish --nproc 8 -- --config configs/train_i2v.json

set -l nproc 8

# Parse launcher args (before --)
set -l train_args
set -l parsing_launcher true
for arg in $argv
    if test "$arg" = "--"
        set parsing_launcher false
        continue
    end
    if $parsing_launcher
        if set -q _expect_nproc
            set nproc $arg
            set -e _expect_nproc
            continue
        end
        if test "$arg" = "--nproc"
            set -g _expect_nproc 1
            continue
        end
    else
        set -a train_args $arg
    end
end

# If no -- separator, treat all args as training args
if $parsing_launcher
    set train_args $argv
end

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

echo "Launching training with $nproc GPUs..."
LOGURU_LEVEL=DEBUG torchrun --nproc_per_node=$nproc -m src.cli.train_i2v $train_args
