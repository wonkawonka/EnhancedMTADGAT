"""Run only the requested SWaT screening after the existing local matrix audit."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-local', default='runs/internal/mscm_anchor/mscm_rcr_anchor_20261107_s3407_v2')
    parser.add_argument('--output', default='runs/internal/mscm_simple_swat_20261107_s3407')
    args = parser.parse_args()
    local, output = Path(args.after_local), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status = output / 'queue_status.json'
    command = [sys.executable, '-u', '-m', 'src.runners.run_mscm_simple_swat', '--output', str(output),
               '--seed', '3407', '--pretrain-epochs', '20', '--adapt-epochs', '30',
               '--patience', '8', '--batch', '128', '--window', '100', '--stride', '10']

    def record(state, **values):
        write(status, dict(status=state, updated_at=datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=8))).isoformat(), **values))

    write(output / 'queue_manifest.json', dict(dataset='SWAT', after_local=str(local),
        command=command, arms=['A1', 'A2', 'A2-T', 'A3', 'A3-T', 'A5'], seed=3407,
        cache_mode='original cache implementation restored at user request',
        gpu_sequence='start after the newer RCR-anchored Memory request completes', no_kaggle_submission=True))
    try:
        record('waiting_for_local_completion', local_output=str(local))
        print('SWAT QUEUED AFTER LOCAL AUDIT', local, flush=True)
        while not (local / 'complete.json').exists():
            p = local / 'queue_status.json'
            if p.exists():
                state = json.loads(p.read_text()).get('status')
                if state in ['failed', 'training_failed', 'skipped_incomplete_training']:
                    raise RuntimeError(f'Local predecessor stopped ({state}); SWaT not started')
            time.sleep(10)
        predecessor = json.loads((local / 'complete.json').read_text())
        assert predecessor['requested_matrix_complete'] and len(predecessor['units']) == 7
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', MPLCONFIGDIR='/tmp/mscm-simple-matplotlib', PYTHONUNBUFFERED='1')
        with (output / 'training.log').open('a') as log:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            record('running_swat', pid=child.pid, command=command)
            result = child.wait()
        if result != 0 or not (output / 'complete.json').exists():
            raise RuntimeError(f'SWaT did not complete: returncode={result}; inspect training.log')
        record('complete', report='report/analysis/rcr/MSCM_SIMPLE_SWAT_20261107.md')
        print('SWAT QUEUE COMPLETE', output, flush=True)
    except Exception:
        record('failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
