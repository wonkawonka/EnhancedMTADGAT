"""Run one disjoint, owner-isolated shard of the unchanged priority-five MCR."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from src.runners.run_mcr_priority5 import SEEDS, DIRECT, seed_root, share_fixed_contexts
from src.runners.run_mcr_full_validation import HEADS, prepare_conditions, run_unit
from src.runners.run_rcr import atomic_json
from src.runners.run_rcr_memory_probe import digest


PLAN = 'configs/internal/mcr_priority5_kaggle4_20261025.json'


def link_directory(target, source):
    if target.is_symlink():
        assert target.resolve() == source.resolve()
    elif target.exists():
        raise RuntimeError(f'Expected new runtime directory: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve(), target_is_directory=True)


def setup_runtime(bundle, output, plan):
    temporary = Path('/kaggle/temp') if Path('/kaggle').exists() else Path('/tmp')
    runtime = temporary / ('runtime_' + output.name)
    runtime.mkdir(parents=True, exist_ok=True)
    frozen = bundle / 'payload' / plan['root']
    for name in ['model_one', 'cohorts']:
        link_directory(runtime / name, frozen / name)
    for seed in SEEDS:
        destination = seed_root(output, seed) / 'model_two'
        destination.mkdir(parents=True, exist_ok=True)
        link_directory(seed_root(runtime, seed) / 'model_two', destination)
        if seed != 3407:
            for name in ['model_one', 'cohorts']:
                link_directory(seed_root(runtime, seed) / name, frozen / name)
    return runtime


def worker(args, plan):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle CUDA GPU required')
    torch.set_num_threads(4)
    runtime = Path(args.runtime)
    if args.seed != 3407:
        share_fixed_contexts(runtime, seed_root(runtime, args.seed), args.unit)
    config = SimpleNamespace(epochs=plan['epochs'], max_epochs=plan['max_epochs'],
        patience=plan['patience'], batch=plan['batch'], seed=args.seed,
        cache=str(Path(args.bundle) / 'payload' / plan['cache']))
    run_unit(seed_root(runtime, args.seed), args.unit, config,
             HEADS if args.seed == 3407 else DIRECT)


def regularize_context_markers(output, unit):
    """Export JSON as regular files; large shared arrays stay canonical only."""
    for seed in SEEDS[1:]:
        folder = seed_root(output, seed) / 'model_two' / unit
        for path in folder.iterdir():
            if not path.is_symlink():
                continue
            if path.suffix == '.json':
                contents = path.read_bytes()
                path.unlink()
                path.write_bytes(contents)
            else:
                path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=PLAN)
    parser.add_argument('--shard', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--runtime')
    parser.add_argument('--unit')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if args.worker:
        worker(args, plan)
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU required')
    shard = plan['shards'][args.shard]
    bundle, output = Path(args.bundle).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = setup_runtime(bundle, output, plan)
    # source_for() intentionally retains its original relative source paths.
    project = Path.cwd()
    link_directory(project / 'runs', bundle / 'payload' / 'runs')
    sources = json.loads((bundle / 'frozen_manifest.json').read_text())
    state = dict(task=plan['task'], shard=args.shard, account=shard['account'],
        units=shard['units'], seeds=SEEDS, expected_heads=11*len(shard['units']),
        implementation=sources['code'], parent_signatures=sources['parents'],
        bundle_manifest_sha256=digest(bundle/'frozen_manifest.json'),
        plan=plan, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        status='running', completed=[], started_at=time.time(),
        parent_selection='fixed normal-validation best A4; no test selection',
        raw_target_masking=True, full_variable_LOO_training=True)
    marker = output / 'protocol_manifest.json'
    atomic_json(marker, state)
    config = SimpleNamespace(batch=plan['batch'], cache=str(bundle/'payload'/plan['cache']))
    jobs, active = [], {}
    try:
        for unit in shard['units']:
            # Compute once on GPU 0, then share exactly the same frozen arrays.
            prepare_conditions(runtime, unit, config)
            provenance = output / 'provenance' / unit
            provenance.mkdir(parents=True, exist_ok=True)
            for name in ['mapping.json', 'metadata.csv']:
                shutil.copy2(bundle/'payload'/plan['cache']/'inputs'/unit/name, provenance/name)
            jobs.extend((unit, seed) for seed in SEEDS)
        pending, active = list(jobs), {}
        devices = list(range(torch.cuda.device_count()))
        while pending or active:
            for device in devices:
                if device in active or not pending:
                    continue
                unit, seed = pending.pop(0)
                command = [sys.executable, '-u', '-m', 'src.runners.run_mcr_kaggle_shard',
                    '--plan', args.plan, '--shard', args.shard, '--bundle', str(bundle),
                    '--output', str(output), '--worker', '--runtime', str(runtime),
                    '--unit', unit, '--seed', str(seed)]
                log = output / f'{unit}_seed{seed}.log'
                handle = log.open('a')
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(device)})
                active[device] = (process, handle, unit, seed)
                print('MCR START', unit, seed, 'GPU', device, flush=True)
            for device, (process, handle, unit, seed) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[device]
                if code:
                    raise RuntimeError(f'{unit}/seed{seed} failed (exit {code}); see saved log')
                state['completed'].append(dict(unit=unit, seed=seed,
                    heads=list(HEADS if seed == 3407 else DIRECT)))
                atomic_json(marker, state)
                print('MCR COMPLETE', unit, seed, flush=True)
            if pending or active:
                time.sleep(5)
        for unit in shard['units']:
            regularize_context_markers(output, unit)
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        for process, handle, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
                process.wait()
            handle.close()
        raise
    finally:
        state['ended_at'] = time.time()
        atomic_json(marker, state)
    # A bounded inventory makes missing model/seed/score artifacts auditable.
    inventory = {}
    for path in output.rglob('*'):
        if path.is_file() and not path.is_symlink():
            inventory[path.relative_to(output).as_posix()] = dict(bytes=path.stat().st_size, sha256=digest(path))
    atomic_json(output/'artifact_inventory.json', inventory)
    print('MCR SHARD COMPLETE', args.shard, state['expected_heads'], flush=True)


if __name__ == '__main__':
    main()
