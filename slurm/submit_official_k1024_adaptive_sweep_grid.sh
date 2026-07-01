#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/official_k1024_adaptive_sweep.sbatch"

if [ "$#" -gt 0 ]; then
  PARAM_GRID="$(printf '%s\n' "$@")"
else
  PARAM_GRID="${PARAM_GRID:-$(cat <<'GRID'
0.5,1.0,10
0.7,1.0,10
0.9,1.0,10
GRID
)}"
fi

echo "$PARAM_GRID" | while IFS=, read -r u g r; do
  [ -n "$u" ] || continue
  case "$u" in \#*) continue ;; esac
  u="$(printf '%s' "$u" | xargs)"
  g="$(printf '%s' "$g" | xargs)"
  r="$(printf '%s' "$r" | xargs)"
  echo "submitting U=$u G=$g R=$r"
  sbatch --export=ALL,U="$u",G="$g",R="$r" "$TRAIN_SCRIPT"
done

