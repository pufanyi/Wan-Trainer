#!/usr/bin/env fish
# Wan2.2 I2V multi-node training launcher
#
# Expected environment variables (typically set by the cluster scheduler):
#   MASTER_ADDR  — hostname/IP of the master node
#   WORLD_SIZE   — number of nodes
#   RANK         — this node's rank (0-indexed)
#
# Optional environment variables:
#   MASTER_PORT  — port on master node (default: 29500)
#
# Usage: fish scripts/train/i2v_multinode.fish [--nproc N] [-- training args...]
#   e.g. fish scripts/train/i2v_multinode.fish --nproc 8 -- --config configs/train_i2v.yaml

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

# Validate required environment variables
if not set -q MASTER_ADDR; or test -z "$MASTER_ADDR"
    echo "ERROR: MASTER_ADDR is not set" >&2
    exit 1
end
if not set -q WORLD_SIZE; or test -z "$WORLD_SIZE"
    echo "ERROR: WORLD_SIZE is not set" >&2
    exit 1
end
if not set -q RANK; or test -z "$RANK"
    echo "ERROR: RANK is not set" >&2
    exit 1
end

set -l master_port (set -q MASTER_PORT; and echo $MASTER_PORT; or echo 29500)

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

echo "Launching multi-node training: node $RANK/$WORLD_SIZE, $NPROC GPUs/node, master=$MASTER_ADDR:$MASTER_PORT"
torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NPROC \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m src.cli.train_i2v $train_args
