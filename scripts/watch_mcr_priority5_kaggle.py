"""Monitor four kernels, audit their Output, then import a complete MCR matrix."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.prepare_mcr_priority5_kaggle import PLAN,digest
from src.runners.run_mcr_priority5 import SEEDS,DIRECT,seed_root,share_fixed_contexts
from src.runners.run_mcr_full_validation import HEADS
from src.runners.run_rcr import atomic_json


def read_status(plan):
    tasks = {}
    for name in plan['shards']:
        submission = PROJECT/'kaggle'/f'mcr_priority5_{name}_20261025/submission_manifest.json'
        record = dict(status='waiting_frozen_parent',units=plan['shards'][name]['units'])
        if submission.exists():
            info = json.loads(submission.read_text())
            record.update(kernel=info['kernel'],submitted=info.get('submitted',False),
                status=info.get('latest_status',info.get('initial_status','uploading_or_submitting')),
                version_number=info.get('version_number'),scriptVersionId=info.get('scriptVersionId'))
            if info.get('version_number'):
                directory = PROJECT/'runs/kaggle_downloads'/f'mcr_priority5_{name}_v{info["version_number"]}'
                if (directory/'output_inventory.json').exists():
                    audit = json.loads((directory/'output_inventory.json').read_text())
                    record.update(audit_status=audit['status'],download=str(directory),
                        scriptVersionId=audit.get('scriptVersionId'),
                        verified_heads=len(audit.get('verified_heads',[])))
        tasks[name] = record
    return tasks


def import_verified(plan,tasks):
    import pandas as pd
    root = PROJECT/plan['root']
    units = json.loads((root/'model_one_protocol.json').read_text())['units']
    provenance,implementations,parents = {},{},{}
    for name, record in tasks.items():
        assert record['audit_status'] == 'complete'
        download = Path(record['download'])
        audit = json.loads((download/'output_inventory.json').read_text())
        assert audit['paged_to_end'] and audit['scriptVersionId']
        source = download/audit['artifact_prefix']
        for path, signature in audit['implementations'].items():
            assert digest(PROJECT/path) == signature, f'Local/executed source differs: {path}'
            assert implementations.get(path,signature) == signature
            implementations[path] = signature
        for unit, signatures in audit['parent_signatures'].items():
            assert unit not in parents,'Duplicate unit across shards'
            for path, signature in signatures.items():
                assert digest(root/'model_one'/unit/'A4'/path) == signature
            parents[unit] = signatures
            for seed in SEEDS:
                src = seed_root(source,seed)/'model_two'/unit
                dst = seed_root(root,seed)/'model_two'/unit
                shutil.copytree(src,dst,dirs_exist_ok=True)
            for seed in SEEDS[1:]:
                share_fixed_contexts(root,seed_root(root,seed),unit)
        provenance[name] = {k:audit[k] for k in ['kernel','version_number','scriptVersionId','artifact_prefix','pages','paged_to_end']}
    assert set(parents) == set(units)
    protocol = dict(task='MCR-PRIORITY5-DIRECT-MEMORY-3SEEDS-20261025-v1',
        units=units,seeds=SEEDS,heads_seed3407={k:list(v) for k,v in HEADS.items()},
        extra_seed_heads={k:list(v) for k,v in DIRECT.items()},epochs=plan['epochs'],
        max_epochs=plan['max_epochs'],patience=plan['patience'],batch=plan['batch'],
        parent_selection='continued A4 normal-validation best, same parent for all head seeds',
        raw_target_masking=True,full_variable_LOO_training=True,independent_calibration=False,
        seed_variation='relation head only; frozen shared reconstructor/Memory/calibration/context',
        parent_signatures=parents,implementations=implementations,test_selection=False,
        execution='four owner-isolated Kaggle tasks',kaggle_provenance=provenance)
    atomic_json(root/'model_two_protocol.json',protocol)
    frames=[pd.read_csv(seed_root(root,s)/'model_two'/u/'metrics.csv',float_precision='round_trip').assign(seed=s)
        for s in SEEDS for u in units]
    pd.concat(frames,ignore_index=True).to_csv(root/'model_two_metrics.csv',index=False)
    atomic_json(root/'model_two_complete.json',dict(units=units,seeds=SEEDS,
        neural_heads=11*len(units),fixed_model_one=True,execution='kaggle',
        all_four_versions_fully_audited=True,kaggle_provenance=provenance))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    plan=json.loads((PROJECT/PLAN).read_text())
    root=PROJECT/plan['root']
    processes={}
    while True:
        for name,(process,handle) in list(processes.items()):
            if process.poll() is not None:
                handle.close();del processes[name]
        tasks=read_status(plan)
        for name,record in tasks.items():
            if not record.get('submitted') or record.get('audit_status')=='complete' or name in processes:
                continue
            log=(root/f'kaggle_{name}_audit.log').open('a')
            process=subprocess.Popen([sys.executable,'-u','scripts/audit_mcr_priority5_kaggle.py',
                '--shard',name],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
            processes[name]=(process,log)
        complete=all(r.get('audit_status')=='complete' for r in tasks.values())
        state=dict(updated_at=time.time(),execution='kaggle_four_shards',tasks=tasks,
            all_four_audited=complete,imported=(root/'model_two_complete.json').exists())
        atomic_json(root/'kaggle_pipeline_status.json',state)
        print(json.dumps({k:v.get('audit_status',v.get('status')) for k,v in tasks.items()}),flush=True)
        if complete:
            if not (root/'model_two_complete.json').exists():
                import_verified(plan,tasks)
            print('ALL FOUR KAGGLE SHARDS VERIFIED AND IMPORTED',flush=True)
            return
        if args.once:
            for process,handle in processes.values():
                process.wait();handle.close()
            return
        time.sleep(60)


if __name__=='__main__':
    main()
