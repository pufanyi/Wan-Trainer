#!/usr/bin/env fish
# Wan2.2 I2V inference launcher
# Usage: fish scripts/inference/i2v.fish --image path/to/img.jpg --prompt "desc"

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

python -m src.cli.infer_i2v $argv
