#!/usr/bin/env bash
set -euo pipefail

cd /home/chenmj/projects/EnhancedMTADGAT

.venv/bin/python -m src.runners.compare_experiments \
  --plan configs/internal/85_c3_restricted_state_public_quick_validation.json \
  --python .venv/bin/python \
  --batch-tag fresh_20260815

.venv/bin/python -m src.runners.compare_experiments \
  --plan configs/internal/86_c3_restricted_state_tsinghua_quick_validation.json \
  --python .venv/bin/python \
  --batch-tag fresh_20260815
