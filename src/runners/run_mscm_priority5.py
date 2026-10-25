"""The user's five-priority round: continue original A4, compare three fusions."""
import argparse
import fcntl
import gc
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from src.models.mscm_reconstruction import MSCMReconstructor
from src.runners.full_validation_support import UNITS,train_network
from src.runners.run_mscm_development import InputArrays,initialize_banks,infer_moments,evaluate_model,set_seed
from src.runners.run_local_residual_models import source_for
from src.runners.run_rcr import atomic_json,atomic_checkpoint,state_hash
from src.runners.run_rcr_memory_probe import digest


DEFAULT_ROOT='runs/internal/mscm_mcr_priority5_20261025_v1'
ARMS={'A4':('A4','bounded'),'A4_pure':('A4','pure'),'A4_direct':('A4','direct'),
      'A1':('A1','bounded'),'A6':('A6','bounded')}


def import_continuation(model,source,output,config):
    if (output/'last.pt').exists() or (output/'complete.json').exists():return
    output.mkdir(parents=True,exist_ok=True)
    model.configure_stage('memory',False)
    old=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(old['model'])
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.0001)
    optimizer.load_state_dict(old['optimizer'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=3,
        threshold=1e-4,threshold_mode='rel',min_lr=1e-6)
    history=[]
    for row in old['history']:
        row=dict(row)
        row['teacher_memory_loss']=row.pop('teacher_memory_mse',row.get('teacher_memory_loss',0.))
        row['source']='original_short_run'
        history.append(row)
    saved=dict(model=old['model'],optimizer=old['optimizer'],scheduler=scheduler.state_dict(),
        epoch=old['epoch'],best=old['best'],anchor=old['best'],best_epoch=old['best_epoch'],
        stale=old['stale'],history=history,limit=config.epochs,rng=old['rng'],cuda_rng=old['cuda_rng'])
    shutil.copy2(source/'best.pt',output/'best.pt')
    atomic_checkpoint(output/'last.pt',saved)
    initial=json.loads((source/'initial.json').read_text())
    initial.update(continued_from=str(source),old_last_sha256=digest(source/'last.pt'),
        old_best_sha256=digest(source/'best.pt'),resume_epoch=old['epoch']+1,
        optimizer_state_reused=True,normal_validation_scheduler_added_at_continuation=True)
    atomic_json(output/'initial.json',initial)


def mark_original_calibration(training):
    path=training/'complete.json';record=json.loads(path.read_text())
    record.update(calibration_vins_seen=True,independent_calibration=False,
        calibration_protocol='original normal training cohort, unchanged from short experiment')
    atomic_json(path,record)


def run_unit(root,core,unit,config):
    source,_=source_for(unit)
    mapping=json.loads((source/'mapping.json').read_text())
    assert mapping==json.loads((core/'inputs'/unit/'mapping.json').read_text())
    arrays=InputArrays(core/'inputs'/unit)
    split=root/'cohorts'/unit;split.mkdir(parents=True,exist_ok=True)
    n=len(arrays.arrays['train']);indices=np.arange(n)
    np.save(split/'fit_window_indices.npy',indices);np.save(split/'calibration_window_indices.npy',indices)
    atomic_json(split/'normal_split.json',dict(original_cohort_unchanged=True,independent_calibration=False,
        fit_windows=n,calibration_windows=n,source_mapping_sha256=digest(source/'mapping.json')))
    output_root=root/'model_one'/unit;output_root.mkdir(parents=True,exist_ok=True)
    teacher_state=torch.load(core/unit/'pretrain_multi/best.pt',map_location='cpu',weights_only=True)
    for name,(variant,fusion) in ARMS.items():
        output=output_root/name;output.mkdir(exist_ok=True)
        if (output/'complete.json').exists() and (name!='A4' or (output_root/'A4_final/complete.json').exists()):continue
        set_seed(config.seed)
        teacher=MSCMReconstructor('A1',patch=8).cuda().eval();teacher.load_state_dict(teacher_state)
        for p in teacher.parameters():p.requires_grad_(False)
        model=MSCMReconstructor(variant,patch=8,gamma=.1,fusion=fusion).cuda()
        missing,unexpected=model.load_state_dict(teacher_state,strict=False)
        assert not unexpected and all(k.startswith('memories.') for k in missing)
        if name in ['A4','A1','A6']:
            import_continuation(model,core/unit/name/'training',output/'training',config)
        elif not (output/'training/last.pt').exists():
            initialize_banks(model,teacher,arrays,output,config)
        atomic_json(output/'arm.json',dict(name=name,variant=variant,fusion=fusion,patch=8,gamma=.1,
            memory_weight=.01,finetune_encoder=False,teacher_sha256=digest(core/unit/'pretrain_multi/best.pt'),
            original_training_cohort=True,continued_from_old=name in ['A4','A1','A6']))
        model=train_network(model,arrays,output/'training',config,'memory',teacher=teacher,memory_weight=.01)
        mark_original_calibration(output/'training')
        last=torch.load(output/'training/last.pt',map_location='cpu',weights_only=False)
        atomic_checkpoint(output/'training/final.pt',last['model'])
        infer_moments(model,arrays,source,output,mapping,config)
        evaluate_model(unit,name,arrays,source,output,mapping)
        if name=='A4':
            final=output_root/'A4_final';final.mkdir(exist_ok=True)
            if not (final/'complete.json').exists():
                model.load_state_dict(last['model'])
                infer_moments(model,arrays,source,final,mapping,config)
                evaluate_model(unit,'A4_final',arrays,source,final,mapping)
                record=json.loads((final/'complete.json').read_text())
                record.update(checkpoint='last training epoch, diagnostic only',
                    model_hash=state_hash(model),used_for_parent_selection=False)
                atomic_json(final/'complete.json',record)
        del model,teacher,last;torch.cuda.empty_cache()
        print('PRIORITY5 RECONSTRUCTION COMPLETE',unit,name,flush=True)
    atomic_json(output_root/'complete.json',dict(unit=unit,arms=list(ARMS),original_cohort=True,
        A4_final_is_diagnostic_only=True))
    del arrays,teacher_state;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=DEFAULT_ROOT)
    parser.add_argument('--core',default='runs/internal/mscm_mcr_development_20261025_v2')
    parser.add_argument('--units',nargs='+',default=UNITS)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--max-epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=8)
    parser.add_argument('--batch',type=int,default=256)
    parser.add_argument('--seed',type=int,default=3407)
    args=parser.parse_args();args.patch=8;args.gamma=.1;args.fusion='bounded'
    args.memory_weight=.01;args.finetune_encoder=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA outside sandbox required')
    torch.set_num_threads(4)
    root,core=Path(args.root),Path(args.core);root.mkdir(parents=True,exist_ok=True)
    files=[__file__,'src/runners/full_validation_support.py','src/runners/run_mscm_development.py',
        'src/models/mscm_reconstruction.py','src/models/rcr.py']
    protocol=dict(task='MSCM-MCR-PRIORITY5-20261025-v1',units=args.units,arms={k:list(v) for k,v in ARMS.items()},
        core=str(core),cache=str(core),epochs=args.epochs,max_epochs=args.max_epochs,patience=args.patience,
        batch=args.batch,seed=args.seed,original_cohort_unchanged=True,teacher_frozen=True,
        independent_calibration=False,hyperparameter_search=False,encoder_finetuning=False,
        best_checkpoint='normal validation only',final_checkpoint='diagnostic only, never select by test',
        gpu=torch.cuda.get_device_name(0),implementations={p:digest(p) for p in files},
        source_parents={u:{a:digest(core/u/a/'training/last.pt') for a in ['A4','A1','A6']} for u in args.units},
        test_selection=False)
    marker=root/'model_one_protocol.json'
    if marker.exists():assert json.loads(marker.read_text())==protocol
    else:atomic_json(marker,protocol)
    snapshots=root/'model_one_implementation';snapshots.mkdir(exist_ok=True)
    for p in files:shutil.copy2(p,snapshots/Path(p).name)
    with open('/tmp/enhancedmtadgat_mscm_gpu0.lock','a+') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for unit in args.units:
            if not (root/'model_one'/unit/'complete.json').exists():run_unit(root,core,unit,args)
    for p,signature in protocol['implementations'].items():assert digest(p)==signature
    frames=[pd.read_csv(root/'model_one'/u/a/'metrics.csv',float_precision='round_trip').assign(arm=a)
        for u in args.units for a in list(ARMS)+['A4_final']]
    pd.concat(frames,ignore_index=True).to_csv(root/'model_one_metrics.csv',index=False)
    atomic_json(root/'model_one_complete.json',dict(units=args.units,arms=list(ARMS),
        original_cohort=True,extra_final_checkpoint_evaluated=True))
    print('PRIORITY5 MODEL ONE COMPLETE',flush=True)


if __name__=='__main__':main()
