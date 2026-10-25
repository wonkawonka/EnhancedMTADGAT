"""Submit prepared private MCR shards using each dataset owner's credentials."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.rcr_kaggle_account import authenticate_account
from src.runners.run_rcr import atomic_json


def cli(account, *arguments):
    return subprocess.run([sys.executable, 'scripts/rcr_kaggle_account.py',
        '--account', account, '--', *arguments], cwd=PROJECT, check=True)


def submit(name, wait=False):
    directory = PROJECT/'kaggle'/f'mcr_priority5_{name}_20261025'
    submission = directory/'submission_manifest.json'
    while not submission.exists():
        if not wait:
            raise FileNotFoundError(submission)
        print('WAIT_PREPARED_SHARD', name, flush=True)
        time.sleep(30)
    info = json.loads(submission.read_text())
    if info.get('submitted'):
        print('ALREADY_SUBMITTED', info['kernel'], flush=True)
        return
    account = info['account']
    assert info['dataset'].split('/')[0] == info['kernel'].split('/')[0] == account
    api = authenticate_account(account)
    quota = api.quota_view()
    q = quota.gpu_quota
    remaining = max(0., (q.total_time_allowed-q.time_used-q.time_reserved).total_seconds()/3600)
    info['gpu_quota_at_submission'] = dict(used_hours=q.time_used.total_seconds()/3600,
        remaining_hours=remaining, total_hours=q.total_time_allowed.total_seconds()/3600,
        refresh_at=str(quota.quota_refresh_time))
    if not remaining:
        info.update(submitted=False, blocked='GPU quota exhausted')
        atomic_json(submission, info)
        raise RuntimeError(f'GPU quota exhausted for {account}')
    if not info.get('dataset_uploaded'):
        cli(account, 'datasets', 'create', '-p', str(directory/'inputs'), '--keep-tabular', '--quiet')
        info.update(dataset_uploaded=True, dataset_uploaded_at=time.time())
        atomic_json(submission, info)
    # Dataset ingestion may take time. Only push once the owner can read version 1.
    for attempt in range(90):
        try:
            details = json.loads(api.dataset_status(info['dataset'], format='json'))
            if details['status'] not in ['ready', 'complete']:
                raise RuntimeError(f'Dataset ingestion status: {details["status"]}')
            info['dataset_version'] = details['current_version_number']
            break
        except Exception:
            if attempt == 89:
                raise
            time.sleep(10)
    cli(account, 'kernels', 'push', '-p', str(directory))
    info.update(submitted=True, submitted_at=time.time())
    # Status retains the actual submitted version separately from future reruns.
    status = api.kernels_status(info['kernel'])
    info['initial_status'] = str(status)
    atomic_json(submission, info)
    with api.build_kaggle_client() as client:
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = info['kernel'].split('/')
        response = client.kernels.kernels_api_client.get_kernel(request)
        info['version_number'] = response.metadata.current_version_number
        info['kernel_id'] = response.metadata.id
    atomic_json(submission, info)
    print('SUBMITTED', info['kernel'], 'HEADS', info['expected_heads'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard', required=True, choices=['chen_a','chen_b','daisy_a','daisy_b'])
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    submit(args.shard, args.wait)


if __name__ == '__main__':
    main()
