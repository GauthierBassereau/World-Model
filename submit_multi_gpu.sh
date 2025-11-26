#!/usr/bin/env bash
set -euo pipefail

# Usage: 
#   ./submit_multi_gpu.sh [-s SUFFIX] NUM_GPUS GPU_TYPE path/to/script.py [args...]

# --- 1. Parse Optional Suffix Flag ---
SUFFIX=""
if [[ "${1:-}" == "-s" ]] || [[ "${1:-}" == "--suffix" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "Error: --suffix requires an argument."
        exit 1
    fi
    SUFFIX="${2}"
    shift 2 # Remove -s and the suffix val from args
fi

# --- 2. Parse Required Arguments ---
NUM_GPUS="${1:?Usage: ./submit_multi_gpu.sh [-s SUFFIX] NUM_GPUS GPU_TYPE path/to/script.py [args...]}"
GPU_TYPE="${2:?Usage: ./submit_multi_gpu.sh [-s SUFFIX] NUM_GPUS GPU_TYPE path/to/script.py [args...]}"
shift 2
PYFILE="${1:?Usage: ./submit_multi_gpu.sh [-s SUFFIX] NUM_GPUS GPU_TYPE path/to/script.py [args...]}"
shift || true

# --- 3. Prepare Paths & Modules ---
REPO="$HOME/git/World-Model"
rel="${PYFILE#$REPO/}"
rel="${rel%.py}"
mod="${rel//\//.}"
name="$(basename "$PYFILE" .py)"
mkdir -p "logs"

# Format the suffix for the filename (add a dash if suffix exists)
# Syntax ${VAR:+val} means: if VAR is set and not null, use val.
JOB_SUFFIX="${SUFFIX:+-$SUFFIX}"
JOB_NAME="${name}-${NUM_GPUS}g${JOB_SUFFIX}"

# --- 4. Prepare Script Arguments ---
# Safely quote all remaining args for re-injection into --wrap
args=""
for a in "$@"; do
  args+=" $(printf '%q' "$a")"
done

# --- 5. Submit Job ---
sbatch \
  --job-name="${JOB_NAME}" \
  --output="logs/%x.out" \
  --partition=gpu \
  --gres="gpu:${GPU_TYPE}:${NUM_GPUS}" \
  --nodes=1 --ntasks-per-node="${NUM_GPUS}" \
  --cpus-per-task=8 \
  --mem=320G --time=8-00:00:00 \
  --export=ALL \
  --wrap "bash -lc '
    set -euo pipefail
    source ~/.bashrc
    shopt -s expand_aliases

    echo
    echo \"=============================\"
    echo \"🚀 Job: $JOB_NAME\"
    echo \"⚙️  Config: $name x${NUM_GPUS} (${GPU_TYPE})\"
    echo \"📅 Date: \$(date)\"
    echo \"=============================\"
    echo
    
    wm

    cd \"$REPO\"
    # Using the calculated module path
    torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} -m \"$mod\"$args
  '"