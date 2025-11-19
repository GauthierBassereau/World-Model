#!/usr/bin/env bash
set -euo pipefail

# Arguments:
#   1: NUM_GPUS
#   2: GPU_TYPE (e.g. a100-40g, a100-80g, h200-141g, ...)
#   3: path/to/script.py
#   4+: [optional args to script]

# Helper for seeing available gpu and nodes: sinfo -p gpu -N   -O "NodeHost:15,StateShort:10,Gres:30,GresUsed:30"

NUM_GPUS="${1:?Usage: ./submit_multi_gpu.sh NUM_GPUS GPU_TYPE path/to/script.py [args...]}"
GPU_TYPE="${2:?Usage: ./submit_multi_gpu.sh NUM_GPUS GPU_TYPE path/to/script.py [args...]}"
shift 2
PYFILE="${1:?Usage: ./submit_multi_gpu.sh NUM_GPUS GPU_TYPE path/to/script.py [args...]}"
shift || true

REPO="$HOME/git/World-Model"
rel="${PYFILE#$REPO/}"
rel="${rel%.py}"
mod="${rel//\//.}"
name="$(basename "$PYFILE" .py)"
mkdir -p "logs"

# Safely quote all remaining args for re-injection into --wrap
args=""
for a in "$@"; do
  args+=" $(printf '%q' "$a")"
done

sbatch \
  --job-name="${name}-${NUM_GPUS}g" \
  --output="logs/%x.out" \
  --partition=gpu \
  --gres="gpu:${GPU_TYPE}:${NUM_GPUS}" \
  --nodes=1 --ntasks-per-node="${NUM_GPUS}" \
  --cpus-per-task=8 \
  --mem=128G --time=3-00:00:00 \
  --export=ALL \
  --wrap "bash -lc '
    set -euo pipefail
    source ~/.bashrc
    shopt -s expand_aliases

    echo
    echo \"=============================\"
    echo \"🚀 New multi-GPU run: $name x${NUM_GPUS} (${GPU_TYPE}) (\$(date))\"
    echo \"=============================\"
    echo
    
    wm

    cd \"$REPO\"
    torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} -m \"$mod\"$args
  '"
