"""Page version-pinned Kaggle Output and audit only the selected MCR artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.rcr_kaggle_account import authenticate_account
from scripts.prepare_mcr_priority5_kaggle import digest
from src.runners.run_mcr_full_validation import HEADS
from src.runners.run_mcr_priority5 import SEEDS, DIRECT, seed_root
from src.runners.run_rcr import atomic_json
from scripts.evaluate_rcr import metrics


def inventory(api, info, target):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    owner, slug = info['kernel'].split('/')
    files, token, pages = [], None, 0
    with api.build_kaggle_client() as client:
        service = client.kernels.kernels_api_client
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        assert service.get_kernel(request).metadata.current_version_number == info['version_number'], 'Submitted kernel version changed'
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug, request.page_size = owner, slug, 200
            if token:
                request.page_token = token
            response = service.list_kernel_session_output(request)
            files.extend(response.files)
            token, pages = response.next_page_token, pages+1
            if not token:
                break
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        assert service.get_kernel(request).metadata.current_version_number == info['version_number']
    version_ids = set()
    for item in files:
        version_ids.update(re.findall(r'(?:script[-_]versions/|scriptVersionId=|script_version_id=|/kf/)(\d+)', unquote(item.url)))
    record = dict(kernel=info['kernel'], version_number=info['version_number'],
        scriptVersionId=None, page_size=200, pages=pages, paged_to_end=True,
        files=[f.file_name for f in files], status='inventory_only', missing=[])
    atomic_json(target/'output_inventory.json', record)
    assert len(version_ids) == 1, f'Expected unique scriptVersionId: {version_ids}'
    record['scriptVersionId'] = next(iter(version_ids))
    if info.get('scriptVersionId'):
        assert info['scriptVersionId'] == record['scriptVersionId']
    manifests = [f for f in files if f.file_name.startswith('artifacts/') and f.file_name.endswith('/protocol_manifest.json')]
    assert len(manifests) == 1, 'Expected one actual artifacts protocol manifest'
    record['artifact_prefix'] = str(Path(manifests[0].file_name).parent)
    atomic_json(target/'output_inventory.json', record)
    return files, record


def download(files, target, record):
    prefix = record['artifact_prefix']+'/'
    selected = [f for f in files if f.file_name.startswith(prefix)]
    def one(item):
        relative = Path(item.file_name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe Output path')
        path = target/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        temporary = path.with_suffix(path.suffix+'.download')
        with requests.get(item.url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open('wb') as handle:
                for block in response.iter_content(4*1024*1024):
                    handle.write(block)
        temporary.replace(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, selected))
    record['downloaded_files'] = len(selected)


def verify(root, info, record):
    manifest = json.loads((root/'protocol_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['account'] == info['account'] and manifest['shard'] == info['shard']
    assert manifest['bundle_manifest_sha256'] == info['manifest_sha256']
    assert manifest['parent_signatures'] == info['parents']
    assert manifest['expected_heads'] == info['expected_heads']
    artifact_hashes = json.loads((root/'artifact_inventory.json').read_text())
    for name, stored in artifact_hashes.items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root/relative
        assert path.is_file() and path.stat().st_size == stored['bytes'] and digest(path) == stored['sha256'], name
    verified = []
    for unit in info['units']:
        mapping = json.loads((root/'provenance'/unit/'mapping.json').read_text())
        for seed in SEEDS:
            folder = seed_root(root,seed)/'model_two'/unit
            needed = ['complete.json','metrics.csv','metrics.json','thresholds.json',
                'conditions_complete.json','prediction_sets.csv','error_overlap.csv']
            needed += [f'{p}_scores.csv' for p in ['train','validation','test']]
            for name in needed:
                if not (folder/name).is_file():
                    record['missing'].append(f'{unit}/{seed}/{name}')
            if record['missing']:
                continue
            completed = json.loads((folder/'complete.json').read_text())
            assert completed['model_one_scores_exact'] and completed['full_target_training']
            context = json.loads((folder/'conditions_complete.json').read_text())
            assert context['raw_target_context_max_error'] < 1e-6
            heads = HEADS if seed == 3407 else DIRECT
            assert set(completed['heads']) == set(heads)
            for head in heads:
                directory = folder/head
                for name in ['training/best.pt','training/last.pt','training/complete.json',
                    'complete.json','head_protocol.json','mask_audit.json','normal_relation_error_scale.npy']:
                    assert (directory/name).is_file(), f'{unit}/{seed}/{head}/{name}'
                assert json.loads((directory/'mask_audit.json').read_text())['trained_residual_target_perturbation_error'] < 1e-6
                for part in ['train','validation','test']:
                    error = np.load(directory/f'{part}_variable_mae.npy',mmap_mode='r')
                    scores = np.load(directory/f'{part}_window_scores.npy',mmap_mode='r')
                    assert error.shape == (mapping['windows'][part],7)
                    assert scores.shape == (mapping['windows'][part],)
                    assert np.isfinite(error).all() and np.isfinite(scores).all()
                    scale = np.load(directory/'normal_relation_error_scale.npy')
                    np.testing.assert_array_equal((error/scale).mean(1), scores)
                verified.append(dict(unit=unit,seed=seed,head=head))
            table = pd.read_csv(folder/'metrics.csv',float_precision='round_trip')
            test = pd.read_csv(folder/'test_scores.csv',float_precision='round_trip')
            for row in table.itertuples():
                for key, value in metrics(test.sample_label,test[row.method],row.threshold).items():
                    assert abs(value-getattr(row,key)) < 1e-12
    assert not record['missing'], record['missing']
    assert len(verified) == info['expected_heads']
    record.update(status='complete', verified_heads=verified,
        implementations=manifest['implementation'], parent_signatures=manifest['parent_signatures'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard',required=True,choices=['chen_a','chen_b','daisy_a','daisy_b'])
    p.add_argument('--status-only',action='store_true')
    args = p.parse_args()
    path = PROJECT/'kaggle'/f'mcr_priority5_{args.shard}_20261025/submission_manifest.json'
    if not path.exists():
        print('WAIT_PARENT',args.shard)
        return
    info = json.loads(path.read_text())
    if not info.get('submitted'):
        print('WAIT_SUBMISSION',args.shard)
        return
    api = authenticate_account(info['account'])
    status = str(api.kernels_status(info['kernel']))
    info['latest_status'] = status
    atomic_json(path,info)
    print('KAGGLE_STATUS',args.shard,status,flush=True)
    if args.status_only or not re.search(r'\bCOMPLETE\b',status):
        return
    target = PROJECT/'runs/kaggle_downloads'/f'mcr_priority5_{args.shard}_v{info["version_number"]}'
    target.mkdir(parents=True,exist_ok=True)
    audit = target/'output_inventory.json'
    if audit.exists() and json.loads(audit.read_text()).get('status') == 'complete':
        return
    files,record = inventory(api,info,target)
    info['scriptVersionId'] = record['scriptVersionId']
    atomic_json(path,info)
    try:
        download(files,target,record)
        verify(target/record['artifact_prefix'],info,record)
    except Exception as exc:
        record.update(status='incomplete',error=repr(exc))
        raise
    finally:
        atomic_json(audit,record)
    print('KAGGLE_AUDITED_COMPLETE',args.shard,len(record['verified_heads']),flush=True)


if __name__ == '__main__':
    main()
