"""Actual-Memory B3/B4 three-seed checks on the frozen continued A4."""
import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from src.runners.run_mscm_priority5 import DEFAULT_ROOT
from src.runners.run_mcr_full_validation import HEADS,run_unit
from src.runners.run_rcr import atomic_json
from src.runners.run_rcr_memory_probe import digest
from src.runners.probe_scale_relation_evidence import read


SEEDS=[3407,3408,3409]
DIRECT={k:v for k,v in HEADS.items() if k in ['B3_memory_gat','B4_memory_mlp']}


def seed_root(root,seed):
    return root if seed==3407 else root/f'relation_seed{seed}'


def share_fixed_contexts(root,target,unit):
    target.mkdir(parents=True,exist_ok=True)
    for folder in ['model_one','cohorts']:
        link=target/folder
        if not link.exists():link.symlink_to((root/folder).resolve(),target_is_directory=True)
    output=target/'model_two'/unit;output.mkdir(parents=True,exist_ok=True)
    source=root/'model_two'/unit
    files=['conditions_complete.json','normal_context_memory_prior.npz','normal_memory_mean.npy',
        'normal_fit_neutral.npy','normal_signed_diagonal_scale.npy']
    for partition in ['train','validation','test']:
        files += [f'{partition}_{suffix}' for suffix in ['conditions_complete.json','direct_context.npy',
            'prior_context.npy','signed_z.npy','signed_absolute_mean.npy']]
    for name in files:
        link=output/name
        if not link.exists():link.symlink_to((source/name).resolve())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=DEFAULT_ROOT)
    parser.add_argument('--cache',default='runs/internal/mscm_mcr_development_20261025_v2')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--max-epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=8)
    parser.add_argument('--batch',type=int,default=256)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    while not (root/'model_one_analysis_complete.json').exists():
        if not args.wait:raise RuntimeError('Finish and audit continued model one first')
        time.sleep(10)
    if not torch.cuda.is_available():raise RuntimeError('CUDA outside sandbox required')
    torch.set_num_threads(4)
    one=json.loads((root/'model_one_protocol.json').read_text());units=one['units']
    files=[__file__,'src/runners/run_mcr_full_validation.py','src/runners/full_validation_support.py',
        'src/models/mcr_direct_context.py','src/models/mcr_gat.py','src/models/mscm_reconstruction.py','src/models/rcr.py']
    protocol=dict(task='MCR-PRIORITY5-DIRECT-MEMORY-3SEEDS-20261025-v1',units=units,seeds=SEEDS,
        heads_seed3407={k:list(v) for k,v in HEADS.items()},extra_seed_heads={k:list(v) for k,v in DIRECT.items()},
        epochs=args.epochs,max_epochs=args.max_epochs,patience=args.patience,batch=args.batch,
        parent_selection='continued A4 normal-validation best, same parent for all head seeds',
        raw_target_masking=True,full_variable_LOO_training=True,independent_calibration=False,
        seed_variation='relation head only; frozen shared reconstructor/Memory/calibration/context',
        parent_signatures={u:{f:digest(root/'model_one'/u/'A4'/f) for f in [
            'training/best.pt','calibration_signed.npz','calibration_absolute.npz','test_scores.csv']} for u in units},
        implementations={p:digest(p) for p in files},test_selection=False)
    marker=root/'model_two_protocol.json'
    if marker.exists():assert json.loads(marker.read_text())==protocol
    else:atomic_json(marker,protocol)
    snapshots=root/'model_two_implementation';snapshots.mkdir(exist_ok=True)
    for p in files:shutil.copy2(p,snapshots/Path(p).name)
    with open('/tmp/enhancedmtadgat_mscm_gpu0.lock','a+') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for unit in units:
            for seed in SEEDS:
                target=seed_root(root,seed)
                if seed!=3407:share_fixed_contexts(root,target,unit)
                if (target/'model_two'/unit/'complete.json').exists():continue
                config=SimpleNamespace(**vars(args),seed=seed)
                run_unit(target,unit,config,HEADS if seed==3407 else DIRECT)
                print('PRIORITY5 RELATION UNIT COMPLETE',unit,seed,flush=True)
    for path,signature in protocol['implementations'].items():assert digest(path)==signature
    for unit in units:
        for name,signature in protocol['parent_signatures'][unit].items():
            assert digest(root/'model_one'/unit/'A4'/name)==signature
    frames=[read(seed_root(root,s)/'model_two'/u/'metrics.csv').assign(seed=s) for s in SEEDS for u in units]
    pd.concat(frames,ignore_index=True).to_csv(root/'model_two_metrics.csv',index=False)
    atomic_json(root/'model_two_complete.json',dict(units=units,seeds=SEEDS,
        neural_heads=len(units)*(len(HEADS)+2*len(DIRECT)),fixed_model_one=True))
    print('PRIORITY5 MODEL TWO COMPLETE',flush=True)


if __name__=='__main__':main()
