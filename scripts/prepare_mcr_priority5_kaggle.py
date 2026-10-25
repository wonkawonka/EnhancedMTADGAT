"""Prepare private data-only/code snapshots for the four priority-five shards.

No repository, credentials, old runs or unselected checkpoints enter a bundle.
The snapshot permits launch independently of a later Git push, while every
runtime Python file and every frozen input has a pinned SHA256.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[1]
PLAN = 'configs/internal/mcr_priority5_kaggle4_20261025.json'
PARTITIONS = ['train', 'validation', 'test']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dependencies(entries):
    seen, todo = set(), list(entries)
    while todo:
        name = todo.pop()
        if name in seen or not (PROJECT/name).is_file():
            continue
        seen.add(name)
        for node in ast.walk(ast.parse((PROJECT/name).read_text())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            for module in names:
                if module.split('.')[0] not in ['src', 'scripts']:
                    continue
                base = Path(*module.split('.'))
                for path in [base.with_suffix('.py'), base/'__init__.py']:
                    if (PROJECT/path).is_file():
                        todo.append(path.as_posix())
    return sorted(seen)


def source_path(unit):
    if unit in ['LFP', 'NCM']:
        return Path('runs/internal/rcr_v1')/unit/'full'
    return Path('runs/internal/rcr_memory_brand3_20261025_v1/sources')/f'fold{unit[-1]}'/'full'


def inputs_for(plan, units):
    files = {plan['root']+'/model_one_protocol.json'}
    for unit in units:
        parent = Path(plan['root'])/'model_one'/unit/'A4'
        if not (PROJECT/parent/'complete.json').exists():
            raise FileNotFoundError(f'A4 is not fully scored yet: {unit}')
        files.update((parent/name).as_posix() for name in ['training/best.pt',
            'training/complete.json', 'complete.json', 'calibration_signed.npz',
            'calibration_absolute.npz', 'train_memory_state.npy'])
        cohort = Path(plan['root'])/'cohorts'/unit
        files.update((cohort/name).as_posix() for name in ['fit_window_indices.npy',
            'calibration_window_indices.npy', 'normal_split.json'])
        source = source_path(unit)
        files.update((source/name).as_posix() for name in ['mapping.json', 'metadata.csv'])
        cache = Path(plan['cache'])/'inputs'/unit
        files.update((cache/name).as_posix() for name in ['mapping.json', 'metadata.csv', 'complete.json'])
        for part in PARTITIONS:
            files.add((cache/f'{part}_x.npy').as_posix())
            files.update((parent/f'{part}_{suffix}').as_posix() for suffix in
                ['signed_residual.npy', 'window_scores.npy', 'scores.csv', 'inference_complete.json'])
            files.update((source/'residuals'/part/name).as_posix() for name in ['condition.npy', 'references.npy'])
        assert json.loads((PROJECT/source/'mapping.json').read_text()) == json.loads((PROJECT/cache/'mapping.json').read_text())
    for name in files:
        if not (PROJECT/name).is_file():
            raise FileNotFoundError(name)
    protocol = json.loads((PROJECT/plan['root']/'model_one_protocol.json').read_text())
    for name, signature in protocol['implementations'].items():
        assert digest(PROJECT/name) == signature, f'Training source changed: {name}'
    return sorted(files)


def git_allowlist(code):
    extra = [__file__, PLAN, 'scripts/rcr_kaggle_account.py',
        'scripts/submit_mcr_priority5_kaggle.py',
        'scripts/audit_mcr_priority5_kaggle.py', 'scripts/watch_mcr_priority5_kaggle.py',
        'tests/test_mcr_kaggle_shard.py', 'tests/test_mcr_direct_context.py',
        'tests/test_mcr_full_validation.py', 'tests/test_mscm_full_validation.py',
        'tests/test_mcr_gat.py', 'tests/test_mscm_reconstruction.py',
        'report/system_design/MSCM_MCR_PRIORITY5_20261025.md',
        'report/system_design/MCR_PRIORITY5_KAGGLE4_20261025.md', '.gitignore']
    wanted = set(code)
    wanted.update(Path(p).relative_to(PROJECT).as_posix() if Path(p).is_absolute() else p for p in extra)
    changed = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=PROJECT, text=True)
    changes = {line[3:] for line in changed.splitlines()}
    selected = sorted(wanted & changes)
    path = PROJECT/'report/system_design/MCR_PRIORITY5_GIT_FILES_20261025.txt'
    selected.append(path.relative_to(PROJECT).as_posix())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(selected)+'\n')
    return path


def build(shard_name):
    plan = json.loads((PROJECT/PLAN).read_text())
    shard = plan['shards'][shard_name]
    inputs = inputs_for(plan, shard['units'])
    code = dependencies(['src/runners/run_mcr_kaggle_shard.py'])
    code += [PLAN]
    parent_signatures = {}
    for unit in shard['units']:
        parent = PROJECT/plan['root']/'model_one'/unit/'A4'
        parent_signatures[unit] = {name: digest(parent/name) for name in
            ['training/best.pt', 'calibration_signed.npz', 'calibration_absolute.npz', 'test_scores.csv']}
    code_records = {name: digest(PROJECT/name) for name in code}
    directory = PROJECT/'kaggle'/f'mcr_priority5_{shard_name}_20261025'
    data = directory/'inputs'
    data.mkdir(parents=True, exist_ok=True)
    manifest = dict(task=plan['task'], shard=shard_name, account=shard['account'],
        units=shard['units'], code=code_records, parents=parent_signatures, files={})
    archive = data/'mcr_frozen_bundle.tar.gz'
    with tarfile.open(archive.with_suffix('.tmp'), 'w:gz', compresslevel=1) as packed:
        for prefix, files in [('code', code), ('payload', inputs)]:
            for name in files:
                path = PROJECT/name
                key = prefix+'/'+name
                record = dict(sha256=digest(path), bytes=path.stat().st_size)
                manifest['files'][key] = record
                packed.add(path, arcname=key, recursive=False)
        raw = json.dumps(manifest, sort_keys=True, indent=2).encode()
        info = tarfile.TarInfo('frozen_manifest.json')
        info.size = len(raw)
        packed.addfile(info, io.BytesIO(raw))
    archive.with_suffix('.tmp').replace(archive)
    slug = f'mcr-priority5-{shard_name.replace("_", "-")}-inputs-20261025'
    dataset = shard['account']+'/'+slug
    (data/'dataset-metadata.json').write_text(json.dumps(dict(id=dataset,
        title=f'MCR priority5 {shard_name} frozen inputs 20261025',
        licenses=[dict(name='other')]), indent=2)+'\n')
    manifest_sha = hashlib.sha256(raw).hexdigest()
    bundle_sha = digest(archive)
    kernel = shard['account']+f'/mcr-priority5-{shard_name.replace("_", "-")}-20261025'
    metadata = dict(id=kernel, title=f'MCR priority5 {shard_name} 20261025', code_file='main.py',
        language='python', kernel_type='script', is_private=True, enable_gpu=True,
        enable_tpu=False, machine_shape='NvidiaTeslaT4', enable_internet=True,
        dataset_sources=[dataset], competition_sources=[], kernel_sources=[])
    (directory/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    main = f'''"""Private pinned MCR input/source snapshot: {shard_name}."""
import hashlib,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
destination=Path('/kaggle/temp/mcr_bundle_{shard_name}');destination.mkdir(parents=True,exist_ok=True)
archives=list(Path('/kaggle/input').rglob('mcr_frozen_bundle.tar.gz'))
if len(archives)==1:
 if sha(archives[0])!={bundle_sha!r}:raise RuntimeError('Frozen archive mismatch')
 with tarfile.open(archives[0],'r:gz') as archive:archive.extractall(destination,filter='data')
else:
 candidates=[p.parent for p in Path('/kaggle/input').rglob('frozen_manifest.json') if sha(p)=={manifest_sha!r}]
 if len(candidates)!=1:raise RuntimeError('Expected one pinned frozen input')
 shutil.copytree(candidates[0],destination,dirs_exist_ok=True)
if sha(destination/'frozen_manifest.json')!={manifest_sha!r}:raise RuntimeError('Manifest mismatch')
manifest=json.loads((destination/'frozen_manifest.json').read_text())
for name,record in manifest['files'].items():
 path=destination/name
 if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise RuntimeError('Frozen file mismatch: '+name)
code=destination/'code';os.chdir(code);sys.path.insert(0,str(code))
import torch
if not torch.cuda.is_available():raise RuntimeError('Kaggle GPU required')
print('GPU_PREFLIGHT',[(torch.cuda.get_device_name(i),torch.cuda.mem_get_info(i)) for i in range(torch.cuda.device_count())],flush=True)
subprocess.run([sys.executable,'-u','-m','src.runners.run_mcr_kaggle_shard','--shard',{shard_name!r},'--bundle',str(destination),'--output','/kaggle/working/artifacts/mcr_priority5_{shard_name}'],check=True)
import pandas as pd
output=Path('/kaggle/working/artifacts/mcr_priority5_{shard_name}')
inventory=json.loads((output/'artifact_inventory.json').read_text())
for path in output.rglob('metrics.csv'):
 target=path.with_suffix('.json')
 pd.read_csv(path,float_precision='round_trip').to_json(target,orient='records',indent=2,double_precision=15)
 inventory[target.relative_to(output).as_posix()]=dict(bytes=target.stat().st_size,sha256=sha(target))
(output/'artifact_inventory.json').write_text(json.dumps(inventory,indent=2))
'''
    (directory/'main.py').write_text(main)
    result = dict(shard=shard_name, account=shard['account'], kernel=kernel, dataset=dataset,
        units=shard['units'], expected_heads=11*len(shard['units']),
        input_sha256=bundle_sha, manifest_sha256=manifest_sha, input_bytes=archive.stat().st_size,
        directory=str(directory), source_mode='SHA256-pinned private source snapshot, no credentials',
        prepared_at=time.time(), parents=parent_signatures)
    (directory/'submission_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    all_code = dependencies(['src/runners/run_mcr_kaggle_shard.py',
        'src/runners/analyze_mscm_priority5.py', 'src/runners/watch_priority5.py'])
    git_allowlist(all_code)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', required=True, choices=['chen_a', 'chen_b', 'daisy_a', 'daisy_b'])
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            build(args.shard)
            return
        except FileNotFoundError as exc:
            if not args.wait:
                raise
            print('WAIT_FROZEN_PARENT', args.shard, str(exc), flush=True)
            time.sleep(30)


if __name__ == '__main__':
    main()
