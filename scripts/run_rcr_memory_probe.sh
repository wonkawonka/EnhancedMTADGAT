#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
output_root="${1:-runs/internal/rcr_memory_probe_20261024_v1}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/rcr-memory-matplotlib

# CPU-only; independent of the CUDA training queue. Avoid duplicate probes.
exec 9>/tmp/enhancedmtadgat_rcr_memory_probe.lock
flock -n 9
for chemistry in NCM LFP; do
  .venv/bin/python -u -m src.runners.run_rcr_memory_probe \
    --root "runs/internal/rcr_v1/${chemistry}/full" \
    --output "${output_root}/${chemistry}" \
    --ks 1 2 4 6 8 --seed 3407 --n-init 5 --max-iter 200 \
    --stability-seeds 3408 3409 --threads 4
done
