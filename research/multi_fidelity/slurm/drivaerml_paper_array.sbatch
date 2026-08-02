#!/bin/bash -l

set -euo pipefail

echo "STOP: this generic batch entry point is retired" >&2
echo "use drivaerml_statistics_array.sbatch or drivaerml_training_array.sbatch" >&2
exit 2
