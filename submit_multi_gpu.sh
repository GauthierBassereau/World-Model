#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:?Usage: ./submit_multi_gpu.sh NUM_GPUS path/to/script.py [args...]}"
shift
PYFILE="${1:?Usage: ./submit_multi_gpu.sh NUM_GPUS path/to/script.py [args...]}"
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
  --gres="gpu:tesla:${NUM_GPUS}" \
  --nodes=1 --ntasks-per-node="${NUM_GPUS}" \
  --cpus-per-task=8 \
  --mem=48G --time=00:15:00 \
  --export=ALL \
  --wrap "bash -lc '
    set -euo pipefail
    source ~/.bashrc
    shopt -s expand_aliases

    echo
    echo \"=============================\"
    echo \"🚀 New multi-GPU run: $name x${NUM_GPUS} (\$(date))\"
    echo \"=============================\"
    echo
    wm

    cd \"$REPO\"
    torchrun --nproc_per_node=${NUM_GPUS} -m \"$mod\"$args
  '"
