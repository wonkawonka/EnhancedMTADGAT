"""Resume the recorded screening command in tmux, then perform CPU analysis."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    launch_path = root / 'launch.json'
    launch = json.loads(launch_path.read_text())
    status_path = root / 'finalization_status.json'
    history_path = root / 'resume_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(previous_launch=launch, resumed_at=now(),
                        host_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        reason='User requested remaining experiments; resume checkpoints without rerunning completed arms'))
    write_json(history_path, history)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MKL_NUM_THREADS='4', MPLCONFIGDIR='/tmp/mscm-simple-matplotlib', PYTHONUNBUFFERED='1')
    try:
        with (root / 'training.log').open('a') as log:
            log.write('\nRESUME_REMAINING_EXPERIMENTS ' + now() + '\n')
            log.flush()
            child = subprocess.Popen(launch['command'], env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            launch.update(pid=child.pid, resumed_at=now(), persistence='tmux mscm_simple_3407',
                          host_boot_id=history[-1]['host_boot_id'])
            write_json(launch_path, launch)
            write_json(status_path, dict(status='waiting_for_existing_training', parent_pid=child.pid,
                                        updated_at=now(), orchestration='tmux; checkpoint resume'))
            result = child.wait()
        if result != 0 or not (root / 'complete.json').exists():
            write_json(status_path, dict(status='training_failed', returncode=result, updated_at=now()))
            sys.exit(result if result > 0 else 1)
        write_json(status_path, dict(status='analyzing', updated_at=now(), new_training_started=False))
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with (root / 'analysis.log').open('a') as log:
            result = subprocess.run([sys.executable, '-u', '-m', 'src.runners.analyze_mscm_simple_screen',
                                     '--output', str(root), '--bootstrap', '500'],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        result.check_returncode()
        write_json(status_path, dict(status='complete', updated_at=now(),
                   report='report/analysis/rcr/MSCM_SIMPLE_SCREEN_20261025.md', new_training_started=False))
    except Exception:
        write_json(status_path, dict(status='failed', updated_at=now(), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
