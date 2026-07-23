#!/usr/bin/env bash
set -u

repo="/home/chenmj/projects/EnhancedMTADGAT"
registry="$repo/runs/internal/20_condition_residual_physical_consistency_development/run_registry.json"
output_root="$repo/runs/internal/20_condition_residual_physical_consistency_development/output"
log="$repo/runs/internal/20_condition_residual_physical_consistency_development/monitor.log"

mkdir -p "$(dirname "$log")"

while true; do
    {
        date '+%Y-%m-%d %H:%M:%S %Z'
        if [[ -f "$registry" ]]; then
            done_count=$(find "$output_root" -mindepth 2 -maxdepth 2 -name metrics.json | wc -l)
            failed_count=$(rg -c '"status": "failed"' "$registry" || echo 0)
            echo "metrics_done=$done_count registry_failed=$failed_count"
        else
            done_count=0
            failed_count=0
            echo "metrics_done=0 registry=not_created"
        fi
        if pgrep -f 'src.runners.train_nc_battery' >/dev/null; then
            echo 'training_process=running'
        else
            echo 'training_process=not_found'
        fi
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
        echo
    } >> "$log"

    total=$((done_count + failed_count))
    if (( total >= 2 )); then
        break
    fi
    sleep 1200
done
