"""Single-GPU, resumable RCR queue. Stops on errors; never alters test-selected settings."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.runners.run_rcr import atomic_json


def make_jobs(base):
    jobs = []
    python = str(PROJECT/'.venv/bin/python')

    def add(name, command, marker, directory):
        jobs.append(dict(name=name, command=[python]+command, marker=str(marker), directory=str(directory)))

    def backbone(chemistry, arm):
        root = base/chemistry/arm
        flags = {'full': [], 'without_feature_graph': ['--no-feature-graph'],
                 'graph_only': ['--no-temporal'], 'reconstruction_only': ['--no-feature-graph', '--no-temporal']}[arm]
        add(f'{chemistry}/{arm}/backbone', ['-u', '-m', 'src.runners.run_rcr', '--chemistry', chemistry,
            '--output', str(root), '--stage', 'residuals']+flags, root/'residuals/test/complete.json', root)
        return root

    def lite(root, mode='continuous'):
        add(f'{root.name}/lite/{mode}/{root.parent.name}', ['-u', '-m', 'src.runners.run_rcr_lite',
            '--root', str(root), '--condition', mode], root/f'lite_{mode}/complete.json', root)

    def net(root, mode='continuous', modulation='mlp'):
        name = f'net_{mode}_{modulation}'
        add(f'{root.parent.name}/{root.name}/{name}', ['-u', '-m', 'src.runners.run_rcr_relation',
            '--root', str(root), '--condition', mode, '--modulation', modulation], root/name/'complete.json', root)

    def evaluate(root, units):
        for unit in units:
            add(f'{root.parent.name}/{root.name}/evaluate/{unit}', ['-u', 'scripts/evaluate_rcr.py',
                '--root', str(root), '--unit', unit], root/unit/'evaluation_complete.json', root)

    # Main method on both settings first; ablations remain mandatory afterward.
    for chemistry in ['LFP', 'NCM']:
        root = backbone(chemistry, 'full')
        lite(root); net(root)
        evaluate(root, ['lite_continuous', 'net_continuous_mlp'])
    for chemistry in ['LFP', 'NCM']:
        root = base/chemistry/'full'
        for mode in ['none', 'cycle']:
            lite(root, mode)
            net(root, mode, 'none' if mode == 'none' else 'mlp')
        net(root, 'continuous', 'none'); net(root, 'continuous', 'linear')
        evaluate(root, ['lite_none', 'lite_cycle', 'net_none_none', 'net_cycle_mlp',
                        'net_continuous_none', 'net_continuous_linear'])
        root = backbone(chemistry, 'without_feature_graph')
        lite(root); net(root)
        evaluate(root, ['lite_continuous', 'net_continuous_mlp'])
        for arm in ['reconstruction_only', 'graph_only']:
            root = backbone(chemistry, arm)
            lite(root)
            evaluate(root, ['lite_continuous'])
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='runs/internal/rcr_v1')
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(PROJECT)
    base = Path(args.base)
    jobs = make_jobs(base)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    base.mkdir(parents=True, exist_ok=True)
    with (base/'queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(base/'queue_plan.json', jobs)
        if args.wait_pid:
            pidpath = Path(f'/proc/{args.wait_pid}/cmdline')
            expected = b'src.runners.run_rcr\x00'
            while pidpath.exists():
                try:
                    command = pidpath.read_bytes()
                except FileNotFoundError:
                    break
                if not command:
                    break
                if expected not in command or str(base/'LFP/full').encode() not in command:
                    raise RuntimeError('Waiting PID no longer belongs to intended LFP backbone; refusing unsafe continuation')
                atomic_json(base/'queue_status.json', dict(state='waiting_existing_backbone', pid=args.wait_pid,
                            supervisor_pid=os.getpid(), verified_at=time.time()))
                time.sleep(15)
        for index, job in enumerate(jobs):
            if Path(job['marker']).exists():
                print('SKIP COMPLETED', job['name'], flush=True)
                continue
            directory = Path(job['directory']); directory.mkdir(parents=True, exist_ok=True)
            logfile = directory/('queue_'+job['name'].replace('/', '_')+'.log')
            print('START', index+1, '/', len(jobs), job['name'], flush=True)
            with logfile.open('a') as handle:
                process = subprocess.Popen(job['command'], stdout=handle, stderr=subprocess.STDOUT,
                                           env={**os.environ, 'MPLCONFIGDIR': '/tmp/rcr-mpl', 'OPENBLAS_NUM_THREADS': '1'})
                atomic_json(base/'queue_status.json', dict(state='running', index=index, total=len(jobs),
                            job=job, pid=process.pid, supervisor_pid=os.getpid(), log=str(logfile)))
                code = process.wait()
            if code or not Path(job['marker']).exists():
                atomic_json(base/'queue_status.json', dict(state='failed', job=job, exit_code=code, log=str(logfile)))
                raise RuntimeError(f"Job failed: {job['name']} (exit={code}); inspect {logfile}")
        atomic_json(base/'queue_status.json', dict(state='rcr_jobs_finished', total=len(jobs),
                    note='Baseline comparison and final artifact audit still required; not goal completion'))
        print('RCR MODEL QUEUE COMPLETE; COMPARISON/AUDIT STILL REQUIRED', flush=True)


if __name__ == '__main__':
    main()
