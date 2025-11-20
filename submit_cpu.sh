#!/usr/bin/env bash
set -euo pipefail

PYFILE="${1:?Usage: ./submit_cpu.sh path/to/script.py [args...]}"
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
  --job-name="$name" \
  --output="logs/%x.out" \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 \
  --mem=400G --time=3-00:00:00 \
  --export=ALL \
  --wrap "bash -lc '
    set -euo pipefail
    source ~/.bashrc
    shopt -s expand_aliases

    echo
    echo '============================='
    echo \"🚀 New run: $name (\$(date))\"
    echo '============================='
    echo
    wm

    cd \"$REPO\"
    python -m \"$mod\"$args
  '"