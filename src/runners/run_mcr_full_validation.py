"""Full-target MCR validation with actual masked Memory and matched controls."""
import argparse
import fcntl
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from src.models.mscm_reconstruction import MSCMReconstructor, FixedRobustResidualCalibration
from src.models.mcr_gat import NormalContextMemoryPrior, MCRSequencePredictor, OriginalRCRLOOPredictor
from src.models.mcr_direct_context import DirectMaskedMemoryContext
from src.models.rcr import EmpiricalCDF
from src.runners.full_validation_support import (UNITS,PARTITIONS,IndexView,train_network,
    ranked_frames,evaluate_frames)
from src.runners.run_mscm_full_validation import DEFAULT_ROOT
from src.runners.run_mscm_development import InputArrays,set_seed
from src.runners.run_local_residual_models import source_for
from src.runners.run_rcr import atomic_json,state_hash
from src.runners.run_rcr_memory_probe import digest
from src.runners.probe_scale_relation_evidence import read,error_overlap


HEADS={
    'B1_original':('original','direct'), 'B2_gat':('gat','direct'),
    'B3_memory_gat':('memory_gat','direct'), 'B4_memory_mlp':('memory_mlp','direct'),
    'B3_prior_gat':('memory_gat','prior'), 'B4_prior_mlp':('memory_mlp','prior'),
    'B3_q_only_gat':('q_only_gat','prior'),
}


def prepare_conditions(root,unit,config):
    output=root/'model_two'/unit;output.mkdir(parents=True,exist_ok=True)
    parent=root/'model_one'/unit/'A4';source,_=source_for(unit)
    fit=np.load(root/'cohorts'/unit/'fit_window_indices.npy')
    cal=np.load(root/'cohorts'/unit/'calibration_window_indices.npy')
    signature=dict(parent=digest(parent/'training/best.pt'),
        calibration=digest(parent/'calibration_signed.npz'),split=digest(root/'cohorts'/unit/'normal_split.json'))
    marker=output/'conditions_complete.json'
    if marker.exists():
        assert json.loads(marker.read_text())['signature']==signature
        return
    qtrain=np.load(source/'residuals/train/condition.npy',mmap_mode='r')
    state=np.load(parent/'train_memory_state.npy',mmap_mode='r')
    prior=NormalContextMemoryPrior.fit(qtrain[fit],state[fit])
    prior.save(output/'normal_context_memory_prior.npz')
    np.save(output/'normal_memory_mean.npy',np.mean(state[fit],axis=0,dtype=np.float64))
    raw=InputArrays(Path(config.cache)/'inputs'/unit)
    total=np.zeros(7);count=0
    for lo in range(0,len(fit),1024):
        values=np.asarray(raw.arrays['train'][fit[lo:lo+1024]],dtype=np.float64)
        total+=values.sum((0,1));count+=values.shape[0]*values.shape[1]
    neutral=total/count
    np.save(output/'normal_fit_neutral.npy',neutral)
    model=MSCMReconstructor('A4').cuda().eval()
    model.load_state_dict(torch.load(parent/'training/best.pt',map_location=next(model.parameters()).device,weights_only=True))
    context=DirectMaskedMemoryContext(model,neutral,prior.mean,prior.std,prior.low,prior.high).cuda().eval()
    prior=prior.cuda().eval()
    audit=context.audit(raw.get('validation',slice(0,4)))
    signed=FixedRobustResidualCalibration.load(parent/'calibration_signed.npz')
    with torch.inference_mode():
        for partition in PARTITIONS:
            part_marker=output/f'{partition}_conditions_complete.json'
            if part_marker.exists():
                assert json.loads(part_marker.read_text())['signature']==signature
                continue
            q=np.load(source/'residuals'/partition/'condition.npy',mmap_mode='r')
            r=np.load(parent/f'{partition}_signed_residual.npy',mmap_mode='r')
            inherited=np.load(parent/f'{partition}_window_scores.npy',mmap_mode='r')
            n=len(q)
            direct=np.lib.format.open_memmap(output/f'{partition}_direct_context.npy',mode='w+',dtype='float32',shape=(n,7,32))
            predicted=np.lib.format.open_memmap(output/f'{partition}_prior_context.npy',mode='w+',dtype='float32',shape=(n,7,32))
            z=np.lib.format.open_memmap(output/f'{partition}_signed_z.npy',mode='w+',dtype='float32',shape=r.shape)
            absolute=np.lib.format.open_memmap(output/f'{partition}_signed_absolute_mean.npy',mode='w+',dtype='float64',shape=(n,7))
            for lo in range(0,n,config.batch):
                stop=lo+config.batch
                x=raw.get(partition,slice(lo,stop))
                controls=torch.from_numpy(np.array(q[lo:stop])).cuda()
                direct_value=context(x,controls);prior_value=prior(controls)
                torch.testing.assert_close(direct_value[...,:8],prior_value[...,:8],rtol=0,atol=0)
                direct[lo:stop]=direct_value.cpu().numpy();predicted[lo:stop]=prior_value.cpu().numpy()
                block=signed.transform(np.asarray(q[lo:stop],dtype=np.float64),np.asarray(r[lo:stop],dtype=np.float64))
                np.testing.assert_allclose(np.square(block).mean((1,2)),inherited[lo:stop,2],rtol=1e-9,atol=1e-8)
                z[lo:stop]=block
                absolute[lo:stop]=np.abs(block.astype('float32').astype('float64')).mean(1)
            for array in [direct,predicted,z,absolute]:array.flush()
            atomic_json(part_marker,dict(signature=signature,windows=n,all_raw_targets_masked=True))
            print('FULL MEMORY CONTEXT CACHED',unit,partition,n,flush=True)
    absolute=np.load(output/'train_signed_absolute_mean.npy',mmap_mode='r')
    np.save(output/'normal_signed_diagonal_scale.npy',np.maximum(np.median(absolute[cal],axis=0),1e-5))
    atomic_json(marker,dict(signature=signature,**audit,
        direct_context='actual scale-wise Memory retrieval, target raw input replaced by fixed normal-fit mean',
        prior_context='normal-fit q to log occupancy regression',
        calibration_vins_excluded_from_context_fitting=not bool(np.intersect1d(fit,cal).size)))
    del context,model,prior,raw;gc.collect();torch.cuda.empty_cache()


class RelationArrays:
    def __init__(self,output,fit_indices):
        self.output=output;self.fit=np.asarray(fit_indices)
        self.z={p:np.load(output/f'{p}_signed_z.npy',mmap_mode='r') for p in PARTITIONS}
        self.arrays=dict(self.z);self.arrays['train']=IndexView(self.z['train'],self.fit)
        self.resident={};free,_=torch.cuda.mem_get_info()
        if sum(self.z[p].nbytes for p in ['train','validation'])<.4*free:
            for p in ['train','validation']:self.resident[p]=torch.from_numpy(np.array(self.z[p])).cuda()
        self.set_context('direct')

    def set_context(self,name):
        self.c={p:np.load(self.output/f'{p}_{name}_context.npy',mmap_mode='r') for p in PARTITIONS}

    def get_full(self,partition,index):
        if partition in self.resident:
            gpu_index=torch.from_numpy(index).cuda() if isinstance(index,np.ndarray) else index
            z=self.resident[partition][gpu_index]
        else:z=torch.from_numpy(np.array(self.z[partition][index])).cuda()
        return z,torch.from_numpy(np.array(self.c[partition][index])).cuda()

    def get(self,partition,index):
        return self.get_full(partition,self.fit[index] if partition=='train' else index)


def infer(model,arrays,output,cal_indices,config,name):
    model.eval();batch=64 if name=='B1_original' else config.batch
    for partition in PARTITIONS:
        marker=output/f'{partition}_inference_complete.json'
        if marker.exists():
            assert json.loads(marker.read_text())['head_hash']==state_hash(model);continue
        n=len(arrays.z[partition])
        errors=np.lib.format.open_memmap(output/f'{partition}_variable_mae.npy',mode='w+',dtype='float64',shape=(n,7))
        with torch.inference_mode():
            for lo in range(0,n,batch):
                z,c=arrays.get_full(partition,slice(lo,lo+batch))
                errors[lo:lo+len(z)]=(z-model(z,c)).abs().double().mean(1).cpu().numpy()
        errors.flush();assert np.isfinite(errors).all()
        atomic_json(marker,dict(head_hash=state_hash(model),windows=n,full_targets=True))
        print('FULL RELATION INFERRED',name,partition,n,flush=True)
    error=np.load(output/'train_variable_mae.npy',mmap_mode='r')
    scale=np.maximum(np.median(error[cal_indices],axis=0),1e-5)
    np.save(output/'normal_relation_error_scale.npy',scale)
    for partition in PARTITIONS:
        e=np.load(output/f'{partition}_variable_mae.npy',mmap_mode='r')
        np.save(output/f'{partition}_window_scores.npy',(e/scale).mean(1))
    independent=not bool(np.intersect1d(arrays.fit,cal_indices).size)
    atomic_json(output/'score_protocol.json',dict(scale='normal calibration window MAE median by variable',
        reconstruction_training_vins_excluded=independent,test_fitting=False))


def audit_head(model,arrays,output):
    z,c=arrays.get('validation',slice(0,4));model.eval();error=0.
    with torch.inference_mode():
        before=model(z,c)
        for j in range(7):
            changed=z.clone();changed[:,:,j]+=10
            error=max(error,float((model(changed,c)[:,:,j]-before[:,:,j]).abs().max()))
    assert error<1e-6
    atomic_json(output/'mask_audit.json',dict(trained_residual_target_perturbation_error=error))


def evaluate_unit(root,unit,heads=None):
    heads=HEADS if heads is None else heads
    source,_=source_for(unit);output=root/'model_two'/unit;parent=root/'model_one'/unit/'A4'
    mapping=json.loads((source/'mapping.json').read_text())
    scale=np.load(output/'normal_signed_diagonal_scale.npy')
    relation_scale=np.load(output/'B3_memory_gat/normal_relation_error_scale.npy')
    names=list(heads)+['Signed_L1','B3_scaled_marginal']
    windows={}
    for p in PARTITIONS:
        absolute=np.load(output/f'{p}_signed_absolute_mean.npy',mmap_mode='r')
        windows[p]=np.column_stack([np.load(output/h/f'{p}_window_scores.npy',mmap_mode='r') for h in heads]+
            [(absolute/scale).mean(1),(absolute/relation_scale).mean(1)])
    frames,snippets=ranked_frames(source,mapping,windows,names)
    base_names=['B0','B0_raw','B0_signed','B0_legacy']
    for p in PARTITIONS:
        base=read(parent/f'{p}_scores.csv').rename(columns={'cal_abs':'B0','raw':'B0_raw',
            'cal_signed':'B0_signed','rcr_legacy':'B0_legacy'})
        keys=['vin','sample_label'] if unit.startswith('Brand3') else ['sample_id','vin','sample_label']
        frames[p]=base[keys+base_names].merge(frames[p][keys+names],on=keys,validate='one_to_one')
        np.testing.assert_array_equal(frames[p].B0.to_numpy(),base.B0.to_numpy())
    names=base_names+names
    normalizers={};fusion=[]
    for method,tag in [('B0','B0'),('B3_memory_gat','B3'),('B3_prior_gat','Prior'),('B3_q_only_gat','Q'),
                       ('B3_scaled_marginal','Diagonal')]:
        if method not in frames['validation']:continue
        validation=frames['validation'][method].to_numpy()
        cdf=EmpiricalCDF().fit(validation);median=float(np.median(validation))
        mad=max(float(1.4826*np.median(np.abs(validation-median))),1e-8)
        normalizers[tag]=dict(median=median,mad=mad)
        np.save(output/f'{tag}_normal_validation_CDF.npy',cdf.reference)
        for p in PARTITIONS:
            frames[p][tag+'_ECDF']=cdf.transform(frames[p][method].to_numpy())
            frames[p][tag+'_MAD']=(frames[p][method]-median)/mad
        fusion += [tag+'_ECDF',tag+'_MAD']
    for tag,prefix in [('B3','B5'),('Prior','B5_prior'),('Q','B5_q'),('Diagonal','B5_diagonal')]:
        if tag not in normalizers:continue
        for mode in ['ECDF','MAD']:
            name=prefix+'_'+mode
            for p in PARTITIONS:frames[p][name]=.5*(frames[p]['B0_'+mode]+frames[p][tag+'_'+mode])
            fusion.append(name)
    names+=fusion
    rows=evaluate_frames(unit,output,frames,names,snippets)
    atomic_json(output/'fusion_protocol.json',dict(normalizers=normalizers,alpha=[0,.5,1],
        primary='B5_ECDF',stage='after existing Top5 aggregation',test_selection=False))
    thresholds=json.loads((output/'thresholds.json').read_text())
    overlaps=[];objects=[];test=frames['test']
    for q in [.95,.99]:
        bt=thresholds['B0'][str(q)]
        for method in ['B3_memory_gat','B4_memory_mlp','B3_prior_gat','B3_q_only_gat','B5_ECDF']:
            if method not in test:continue
            rt=thresholds[method][str(q)]
            overlaps.append(error_overlap(test,'B0',method,bt,rt,unit,q))
            table=test[[c for c in ['sample_id','vin','sample_label'] if c in test]].copy()
            table['q']=q;table['method']=method
            table['base_score']=test.B0;table['relation_score']=test[method]
            table['base_threshold']=bt;table['relation_threshold']=rt
            table['base_alarm']=test.B0>=bt;table['relation_alarm']=test[method]>=rt
            objects.append(table)
    pd.DataFrame(overlaps).to_csv(output/'error_overlap.csv',index=False)
    pd.concat(objects,ignore_index=True).to_csv(output/'prediction_sets.csv',index=False)
    independent=json.loads((output/'conditions_complete.json').read_text())['calibration_vins_excluded_from_context_fitting']
    atomic_json(output/'complete.json',dict(unit=unit,heads=list(heads),score_names=names,
        model_one_scores_exact=True,full_target_training=True,independent_calibration=independent))
    return rows


def run_unit(root,unit,config,heads=None):
    heads=HEADS if heads is None else heads
    output=root/'model_two'/unit;prepare_conditions(root,unit,config)
    fit=np.load(root/'cohorts'/unit/'fit_window_indices.npy')
    cal=np.load(root/'cohorts'/unit/'calibration_window_indices.npy')
    arrays=RelationArrays(output,fit);length=arrays.z['train'].shape[1]
    set_seed(config.seed)
    shared={k:v.clone() for k,v in MCRSequencePredictor('memory_gat',length).state_dict().items()}
    for name,(mode,context) in heads.items():
        path=output/name;path.mkdir(exist_ok=True)
        if (path/'complete.json').exists():continue
        arrays.set_context(context);set_seed(config.seed)
        model=OriginalRCRLOOPredictor() if mode=='original' else MCRSequencePredictor(mode,length)
        if mode!='original':
            compatible={k:v for k,v in shared.items() if k in model.state_dict() and model.state_dict()[k].shape==v.shape}
            model.load_state_dict(compatible,strict=False)
        model=model.cuda()
        if mode=='q_only_gat':
            with torch.no_grad():model.memory_constant.copy_(torch.as_tensor(np.load(output/'normal_memory_mean.npy'),device=next(model.parameters()).device,dtype=torch.float32))
        args=SimpleNamespace(**vars(config));args.microbatch=64 if mode=='original' else config.batch
        atomic_json(path/'head_protocol.json',dict(mode=mode,context=context,effective_batch=config.batch,
            microbatch=args.microbatch,all_variables_each_window=True,initial_parameters=sum(p.numel() for p in model.parameters())))
        model=train_network(model,arrays,path/'training',args,'relation',relation=True,memory_weight=0.)
        training_record=json.loads((path/'training/complete.json').read_text())
        training_record['calibration_vins_seen']=bool(np.intersect1d(fit,cal).size)
        atomic_json(path/'training/complete.json',training_record)
        audit_head(model,arrays,path)
        infer(model,arrays,path,cal,args,name)
        atomic_json(path/'complete.json',dict(head=name,model_hash=state_hash(model),full_data=True))
        del model;torch.cuda.empty_cache()
        print('FULL MCR HEAD COMPLETE',unit,name,flush=True)
    evaluate_unit(root,unit,heads)
    del arrays,shared;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=DEFAULT_ROOT)
    parser.add_argument('--cache',default='runs/internal/mscm_mcr_development_20261025_v2')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--max-epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=8)
    parser.add_argument('--batch',type=int,default=256)
    parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--wait-for-model-one',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    while not (root/'model_one_analysis_complete.json').exists():
        if not args.wait_for_model_one:raise RuntimeError('Complete model-one training and analysis first')
        time.sleep(10)
    if not torch.cuda.is_available():raise RuntimeError('CUDA outside sandbox required')
    torch.set_num_threads(4)
    one=json.loads((root/'model_one_protocol.json').read_text())
    units=one['units']
    files=[__file__,'src/runners/full_validation_support.py','src/models/mcr_direct_context.py',
        'src/models/mcr_gat.py','src/models/mscm_reconstruction.py','src/models/rcr.py']
    protocol=dict(task='MCR-FULL-DIRECT-MEMORY-20261025-v1',units=units,heads={k:list(v) for k,v in HEADS.items()},
        epochs=args.epochs,max_epochs=args.max_epochs,patience=args.patience,batch=args.batch,seed=args.seed,
        training='all variables and all time points per window, microbatch64 accumulated for B1',
        model_one_protocol_sha256=digest(root/'model_one_protocol.json'),
        parent_signatures={u:{f:digest(root/'model_one'/u/'A4'/f) for f in [
            'training/best.pt','calibration_signed.npz','calibration_absolute.npz','test_scores.csv']} for u in units},
        implementations={p:digest(p) for p in files},direct_raw_target_masked=True,
        calibration='disjoint normal training VIN holdout',threshold='original normal validation P95/P99',
        gpu=torch.cuda.get_device_name(0),test_selection=False)
    marker=root/'model_two_protocol.json'
    if marker.exists():assert json.loads(marker.read_text())==protocol
    else:atomic_json(marker,protocol)
    import shutil
    snapshots=root/'model_two_implementation';snapshots.mkdir(exist_ok=True)
    for p in files:shutil.copy2(p,snapshots/Path(p).name)
    with open('/tmp/enhancedmtadgat_mscm_gpu0.lock','a+') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for unit in units:
            if not (root/'model_two'/unit/'complete.json').exists():run_unit(root,unit,args)
    for path,signature in protocol['implementations'].items():assert digest(path)==signature
    frames=[read(root/'model_two'/u/'metrics.csv') for u in units]
    pd.concat(frames,ignore_index=True).to_csv(root/'model_two_metrics.csv',index=False)
    atomic_json(root/'model_two_complete.json',dict(units=units,heads=list(HEADS),neural_heads=len(units)*len(HEADS)))
    print('FULL MCR COMPLETE',flush=True)


if __name__=='__main__':main()
