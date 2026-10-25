"""Complete MSCM controls with VIN-held-out calibration and longer training."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from src.models.mscm_reconstruction import MSCMReconstructor, FixedRobustResidualCalibration
from src.runners.run_local_residual_models import source_for
from src.runners.run_mscm_development import InputArrays, initialize_banks, infer_moments, set_seed
from src.runners.run_rcr import atomic_json
from src.runners.run_rcr_memory_probe import digest
from src.runners.run_rcr_calibration_sensitivity import make_calibration
from src.runners.full_validation_support import (UNITS, PARTITIONS, TrainingSubset, normal_split,
    train_network, ranked_frames, evaluate_frames)


DEFAULT_ROOT = 'runs/internal/mscm_mcr_full_20261025_v1'
ARMS = {
    'A0': dict(variant='A0'), 'A1': dict(variant='A1'), 'A2': dict(variant='A2'),
    'A4': dict(variant='A4'), 'A6': dict(variant='A6'),
    'A4_pure': dict(variant='A4', fusion='pure'),
    'A4_direct': dict(variant='A4', fusion='direct'),
    'A4_lambda0': dict(variant='A4', memory_weight=0.),
    'A4_lambda0p1': dict(variant='A4', memory_weight=.1),
    'A4_patch4': dict(variant='A4', patch=4),
    'A4_patch16': dict(variant='A4', patch=16),
    'A4_finetune': dict(variant='A4', finetune_encoder=True),
}
SCORES = ['raw','cal_abs','cal_signed','rcr_legacy','fit_cal_abs','fit_cal_signed','fit_rcr_legacy']


def evaluate_model(unit, output, source, mapping, fit_indices, cal_indices):
    train = np.load(output/'train_moments.npy',mmap_mode='r')
    q = np.load(source/'residuals/train/condition.npy',mmap_mode='r')
    calibrations = {}
    for label, indices in [('heldout',cal_indices),('fit',fit_indices)]:
        m = np.asarray(train[indices],dtype=np.float64); controls = np.asarray(q[indices],dtype=np.float64)
        signed = FixedRobustResidualCalibration().fit(controls,m[:,0],m[:,1])
        absolute = FixedRobustResidualCalibration().fit(controls,m[:,2],m[:,3])
        rms = np.sqrt(np.maximum(m[:,1]+np.square(m[:,0]-signed.location(controls)),0))
        legacy,_ = make_calibration(controls,signed.q_mean,signed.q_std,signed.coef,rms,.05)
        suffix = '' if label=='heldout' else '_fit'
        signed.save(output/f'calibration_signed{suffix}.npz')
        absolute.save(output/f'calibration_absolute{suffix}.npz')
        legacy.save(output/f'calibration_legacy{suffix}.npz')
        calibrations[label] = signed,absolute,legacy
    windows = {}
    for partition in PARTITIONS:
        m = np.load(output/f'{partition}_moments.npy',mmap_mode='r')
        controls = np.asarray(np.load(source/'residuals'/partition/'condition.npy',mmap_mode='r'),dtype=np.float64)
        columns = [(m[:,1]+m[:,0]**2).mean(1)]
        for label in ['heldout','fit']:
            signed,absolute,legacy = calibrations[label]
            mu,scale = legacy.parameters(controls)
            columns += [absolute.energy(controls,m[:,2],m[:,3]), signed.energy(controls,m[:,0],m[:,1]),
                ((m[:,1]+np.square(m[:,0]-mu))/scale**2).mean(1)]
        windows[partition] = np.column_stack(columns)
        assert np.isfinite(windows[partition]).all()
        np.save(output/f'{partition}_window_scores.npy',windows[partition])
    frames,snippets = ranked_frames(source,mapping,windows,SCORES)
    evaluate_frames(unit,output,frames,SCORES,snippets)
    atomic_json(output/'complete.json',dict(unit=unit,scores=SCORES,primary_score='cal_abs',
        calibration_windows=len(cal_indices),reconstruction_fit_windows=len(fit_indices),
        fit_calibration_control='same frozen network; only calibration fitting VINs differ',
        independent_calibration=True,full_scoring=True,test_selection=False))


def infer_and_evaluate(model, arrays, source, output, mapping, unit, fit, cal, config):
    infer_moments(model,arrays,source,output,mapping,config)
    evaluate_model(unit,output,source,mapping,fit,cal)


def run_unit(root, cache, unit, config):
    source,_ = source_for(unit)
    split = root/'cohorts'/unit
    fit,cal,mapping = normal_split(source,split,config.seed)
    assert (cache/'inputs'/unit/'complete.json').exists()
    assert json.loads((cache/'inputs'/unit/'mapping.json').read_text())==mapping
    arrays = InputArrays(cache/'inputs'/unit)
    fitting = TrainingSubset(arrays,fit)
    unit_root = root/'model_one'/unit; unit_root.mkdir(parents=True,exist_ok=True)
    set_seed(config.seed)
    base = MSCMReconstructor('A0').cuda()
    base = train_network(base,fitting,unit_root/'backbone_training',config,'backbone',memory_weight=0.)
    base_state = {k:v.detach().cpu().clone() for k,v in base.backbone.state_dict().items()}
    reference = unit_root/'RCR_reference';reference.mkdir(exist_ok=True)
    if not (reference/'complete.json').exists():
        infer_and_evaluate(base,arrays,source,reference,mapping,unit,fit,cal,config)
    del base;torch.cuda.empty_cache()
    teacher_states = {}
    for name,variant in [('single','A0'),('multi','A1')]:
        set_seed(config.seed)
        teacher = MSCMReconstructor(variant).cuda();teacher.warm_start(base_state)
        teacher = train_network(teacher,fitting,unit_root/f'pretrain_{name}',config,'pretrain',memory_weight=0.)
        teacher_states[name] = {k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
        del teacher;torch.cuda.empty_cache()
    for name,overrides in ARMS.items():
        output = unit_root/name;output.mkdir(exist_ok=True)
        if (output/'complete.json').exists(): continue
        options = dict(patch=8,gamma=.1,fusion='bounded',memory_weight=.01,finetune_encoder=False)
        options.update(overrides)
        variant = options.pop('variant')
        args = SimpleNamespace(**vars(config), **{k:v for k,v in options.items() if k not in vars(config)})
        for k,v in options.items():setattr(args,k,v)
        set_seed(config.seed)
        kind = 'single' if variant in ['A0','A2'] else 'multi'
        teacher = MSCMReconstructor('A0' if kind=='single' else 'A1',patch=args.patch).cuda().eval()
        teacher.load_state_dict(teacher_states[kind])
        for p in teacher.parameters():p.requires_grad_(False)
        model = MSCMReconstructor(variant,patch=args.patch,gamma=args.gamma,fusion=args.fusion).cuda()
        missing,unexpected = model.load_state_dict(teacher.state_dict(),strict=False)
        assert not unexpected and all(k.startswith('memories.') for k in missing)
        atomic_json(output/'arm.json',dict(name=name,variant=variant,**options,
            cohort_sha256=digest(split/'normal_split.json')))
        if model.use_memory and not (output/'training/last.pt').exists():
            initialize_banks(model,teacher,fitting,output,args)
            local=np.load(output/'memory_initialization_normal_window_indices.npy')
            np.save(output/'memory_initialization_global_window_indices.npy',fit[local])
        model = train_network(model,fitting,output/'training',args,'memory',teacher=teacher,
            memory_weight=args.memory_weight,finetune=args.finetune_encoder)
        infer_and_evaluate(model,arrays,source,output,mapping,unit,fit,cal,args)
        del model,teacher;torch.cuda.empty_cache()
        print('FULL MSCM ARM COMPLETE',unit,name,flush=True)
    atomic_json(unit_root/'complete.json',dict(unit=unit,arms=list(ARMS),independent_calibration=True))
    del arrays,fitting,teacher_states,base_state;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=DEFAULT_ROOT)
    parser.add_argument('--cache',default='runs/internal/mscm_mcr_development_20261025_v2')
    parser.add_argument('--units',nargs='+',default=UNITS)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--max-epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=8)
    parser.add_argument('--batch',type=int,default=256)
    parser.add_argument('--seed',type=int,default=3407)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Run CUDA outside sandbox')
    assert args.max_epochs>=args.epochs>=args.patience
    torch.set_num_threads(4)
    root,cache=Path(args.output),Path(args.cache)
    root.mkdir(parents=True,exist_ok=True)
    files=[__file__,'src/runners/full_validation_support.py','src/models/mscm_reconstruction.py',
        'src/models/rcr.py','src/runners/run_mscm_development.py']
    protocol=dict(task='MSCM-MCR-FULL-20261025-v1',units=args.units,arms=ARMS,
        epochs=args.epochs,max_epochs=args.max_epochs,patience=args.patience,batch=args.batch,seed=args.seed,
        extension='at50: best in last8 and normal validation improvement over last8 >0.1%; extend to100',
        optimizer='AdamW1e-3,ReduceLROnPlateau factor0.5 patience3 floor1e-6',
        reconstruction_fit='80% normal training VINs, all their windows',
        calibration='remaining20% normal training VINs, never used by backbone/teacher/reconstructor',
        threshold='original independent normal validation P95/P99',
        pretrained_weights_reused=False,cache=str(cache),gpu=torch.cuda.get_device_name(0),
        precision='float32',scores=SCORES,primary_score='cal_abs',
        implementations={p:digest(p) for p in files},test_selection=False)
    marker=root/'model_one_protocol.json'
    if marker.exists():assert json.loads(marker.read_text())==protocol,'Changed protocol: use a new output'
    else:atomic_json(marker,protocol)
    import shutil
    snapshots=root/'model_one_implementation';snapshots.mkdir(exist_ok=True)
    for p in files:shutil.copy2(p,snapshots/Path(p).name)
    with open('/tmp/enhancedmtadgat_mscm_gpu0.lock','a+') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for unit in args.units:
            if not (root/'model_one'/unit/'complete.json').exists():run_unit(root,cache,unit,args)
    for path,signature in protocol['implementations'].items():assert digest(path)==signature
    frames=[pd.read_csv(root/'model_one'/u/a/'metrics.csv',float_precision='round_trip').assign(arm=a)
        for u in args.units for a in ['RCR_reference']+list(ARMS)]
    pd.concat(frames,ignore_index=True).to_csv(root/'model_one_metrics.csv',index=False)
    atomic_json(root/'model_one_complete.json',dict(units=args.units,arms=list(ARMS),
        trained_arms=len(args.units)*len(ARMS),A3_alias='A1',A5_alias='A4/raw'))
    print('FULL MSCM COMPLETE',flush=True)


if __name__=='__main__':main()
