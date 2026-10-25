"""MCR-GAT B1-B5 after completed/frozen MSCM model-one evaluation (CUDA)."""
import argparse
import fcntl
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from src.models.mscm_reconstruction import FixedRobustResidualCalibration
from src.models.mcr_gat import NormalContextMemoryPrior,MCRSequencePredictor,OriginalRCRLOOPredictor
from src.models.rcr import EmpiricalCDF
from src.runners.run_mscm_development import UNITS,set_seed
from src.runners.run_local_residual_models import source_for
from src.runners.run_rcr import atomic_json,atomic_checkpoint,state_hash
from src.runners.run_rcr_memory_probe import digest
from src.runners.run_rcr_calibration_sensitivity import aggregate
from src.runners.probe_scale_relation_evidence import rank_frame,read,error_overlap
from scripts.evaluate_rcr import metrics
from sklearn.metrics import roc_curve


HEADS={'B1_original':'original','B2_gat':'gat','B3_memory_gat':'memory_gat','B4_memory_mlp':'memory_mlp',
       'B3_q_only_gat':'q_only_gat'}
TASK='MCR-GAT-DEVELOPMENT-20261025-v1'


def prepare_signed_z(core,unit,output):
    source,_=source_for(unit)
    parent=core/unit/'A4'
    signed=FixedRobustResidualCalibration.load(parent/'calibration_signed.npz')
    signature=dict(parent_training_sha256=digest(parent/'training/complete.json'),
        calibration_signed_sha256=digest(parent/'calibration_signed.npz'))
    marker=output/'signed_z_complete.json'
    if marker.exists():
        assert json.loads(marker.read_text())==signature
        return
    for partition in ['train','validation','test']:
        r=np.load(parent/f'{partition}_signed_residual.npy',mmap_mode='r')
        q=np.load(source/'residuals'/partition/'condition.npy',mmap_mode='r')
        inherited=np.load(parent/f'{partition}_window_scores.npy',mmap_mode='r')
        z=np.lib.format.open_memmap(output/f'{partition}_signed_z.npy',mode='w+',dtype='float32',shape=r.shape)
        diagonal=np.lib.format.open_memmap(output/f'{partition}_signed_absolute_mean.npy',mode='w+',dtype='float64',shape=(len(r),7))
        for lo in range(0,len(r),4096):
            block=signed.transform(np.asarray(q[lo:lo+4096],dtype=np.float64),np.asarray(r[lo:lo+4096],dtype=np.float64))
            assert np.isfinite(block).all()
            np.testing.assert_allclose(np.square(block).mean((1,2)),inherited[lo:lo+len(block),2],rtol=1e-9,atol=1e-8)
            z[lo:lo+len(block)]=block
            diagonal[lo:lo+len(block)]=np.abs(block.astype('float32').astype('float64')).mean(1)
        z.flush()
        diagonal.flush()
    diagonal_scale=np.maximum(np.median(np.load(output/'train_signed_absolute_mean.npy',mmap_mode='r'),axis=0),1e-5)
    np.save(output/'normal_signed_diagonal_scale.npy',diagonal_scale)
    for partition in ['train','validation','test']:
        diagonal=np.load(output/f'{partition}_signed_absolute_mean.npy',mmap_mode='r')
        np.save(output/f'{partition}_signed_diagonal_score.npy',(diagonal/diagonal_scale).mean(1))
    atomic_json(marker,signature)


class RelationArrays:
    def __init__(self,output,source):
        self.z={s:np.load(output/f'{s}_signed_z.npy',mmap_mode='r') for s in ['train','validation','test']}
        self.q={s:np.load(source/'residuals'/s/'condition.npy',mmap_mode='r') for s in self.z}
        self.resident={}
        needed=sum(self.z[s].nbytes+self.q[s].nbytes for s in ['train','validation'])
        free,_=torch.cuda.mem_get_info()
        if needed<.45*free:
            for s in ['train','validation']:
                self.resident[s]=(torch.from_numpy(np.array(self.z[s])).cuda(),torch.from_numpy(np.array(self.q[s])).cuda())
        print('MCR GPU INPUT CACHE',list(self.resident),needed,flush=True)

    def get(self,partition,index):
        if partition in self.resident:
            if isinstance(index,np.ndarray):index=torch.from_numpy(index).cuda()
            z,q=self.resident[partition]
            return z[index],q[index]
        return (torch.from_numpy(np.array(self.z[partition][index])).cuda(),
                torch.from_numpy(np.array(self.q[partition][index])).cuda())


def build_head(name,length,shared=None):
    model=OriginalRCRLOOPredictor() if HEADS[name]=='original' else MCRSequencePredictor(HEADS[name],length)
    if shared is not None and HEADS[name]!='original':
        compatible={k:v for k,v in shared.items() if k in model.state_dict() and model.state_dict()[k].shape==v.shape}
        model.load_state_dict(compatible,strict=False)
    return model.cuda()


def validate(model,prior,arrays,batch):
    model.eval();total,count=0.,0
    with torch.inference_mode():
        for lo in range(0,len(arrays.z['validation']),batch):
            z,q=arrays.get('validation',slice(lo,lo+batch))
            prediction=model(z,prior(q))
            total+=float(F.huber_loss(prediction,z,reduction='sum'));count+=z.numel()
    return total/count


def train(model,prior,arrays,output,name,args):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'training_complete.json').exists():
        model.load_state_dict(torch.load(output/'best.pt',map_location='cuda',weights_only=True))
        assert state_hash(model)==json.loads((output/'training_complete.json').read_text())['head_hash']
        return model.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    eval_batch=min(args.batch,64) if name=='B1_original' else args.batch
    history,start,stale,best_epoch=[],1,0,0
    if (output/'last.pt').exists():
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        history,start,stale,best_epoch,best=saved['history'],saved['epoch']+1,saved['stale'],saved['best_epoch'],saved['best']
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        best=validate(model,prior,arrays,eval_batch)
        atomic_checkpoint(output/'best.pt',model.state_dict())
        atomic_json(output/'initial.json',dict(head_hash=state_hash(model),parameters=sum(p.numel() for p in model.parameters()),normal_validation_loo_huber=best))
    n=len(arrays.z['train'])
    for epoch in range(start,args.epochs+1):
        if stale>=args.patience:break
        model.train();clock=time.monotonic();total,count=0.,0
        order=np.random.default_rng(args.seed+epoch).permutation(n)
        targets=np.random.default_rng(args.seed+10000+epoch).integers(0,7,n)
        for step,lo in enumerate(range(0,n,args.batch),1):
            index=order[lo:lo+args.batch]
            z,q=arrays.get('train',index)
            target=torch.from_numpy(targets[index]).cuda()
            optimizer.zero_grad(set_to_none=True)
            predicted=model(z,prior(q),targets=target)
            expected=z.gather(-1,target[:,None,None].expand(-1,z.shape[1],1)).squeeze(-1)
            loss=F.huber_loss(predicted,expected)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite normal-only MCR training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=float(loss.detach())*len(z);count+=len(z)
            if step%1000==0:print(json.dumps(dict(event='MCR_BATCH',head=name,epoch=epoch,batch=step,batches=int(np.ceil(n/args.batch)),loss=total/count)),flush=True)
        val=validate(model,prior,arrays,eval_batch)
        if val<best:
            best,stale,best_epoch=val,0,epoch;atomic_checkpoint(output/'best.pt',model.state_dict())
        else:stale+=1
        row=dict(epoch=epoch,normal_training_sampled_loo_huber=total/count,normal_validation_all_loo_huber=val,
            best_epoch=best_epoch,seconds=time.monotonic()-clock)
        history.append(row);pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        atomic_checkpoint(output/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
            best=best,best_epoch=best_epoch,stale=stale,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        print(json.dumps(dict(event='MCR_EPOCH',path=str(output),**row)),flush=True)
    model.load_state_dict(torch.load(output/'best.pt',map_location='cuda',weights_only=True))
    atomic_json(output/'training_complete.json',dict(head_hash=state_hash(model),best_epoch=best_epoch,
        epochs_executed=len(history),normal_validation_loo_huber=best,normal_training_only=True,
        training_target_schedule='uniform one whole variable per window per epoch, identical across heads; unbiased full LOO objective',
        validation='all variables, all time points, full normal validation cohort'))
    return model.eval()


def audit_mask(model,prior,arrays,output,name):
    z,q=arrays.get('validation',slice(0,8));c=prior(q)
    model.eval();error=0.
    with torch.inference_mode():
        original=model(z,c)
        for j in range(7):
            changed=z.clone();changed[:,:,j]+=10
            error=max(error,float((model(changed,c)[:,:,j]-original[:,:,j]).abs().max()))
    assert error<1e-6,'Target leakage in trained relation predictor'
    diagnostics=dict(trained_target_perturbation_max_abs_error=error,
        prior_excludes_current_statistics_for_current_target=True,prior_excludes_SOC_statistics_for_SOC_target=True)
    if name=='B3_memory_gat':
        before=state_hash(model)
        with torch.inference_mode():
            prediction,attention,bias=model(z,c,details=True)
            neutral=c.clone();neutral[:,:,8:]=c[:,:,8:].mean((0,1))
            no_mem,neutral_attention,neutral_bias=model(z,neutral,details=True)
            saved=model.template_u.clone();model.template_u.zero_()
            no_edges=model(z,c);model.template_u.copy_(saved)
        assert state_hash(model)==before
        diagnostics.update(memory_condition_attention_max_delta=float((attention-neutral_attention).abs().max()),
            memory_condition_bias_max_delta=float((bias-neutral_bias).abs().max()),
            memory_condition_prediction_mse=float((prediction-no_mem).square().mean()),
            graph_bias_only_prediction_mse=float((prediction-no_edges).square().mean()),
            interventions_restore_parameters_exactly=True)
    atomic_json(output/'mask_and_condition_audit.json',diagnostics)


def infer(model,prior,arrays,output,name,args):
    model.eval();batch=min(args.batch,64) if name=='B1_original' else args.batch
    for partition in ['train','validation','test']:
        marker=output/f'{partition}_inference_complete.json'
        if marker.exists():
            assert json.loads(marker.read_text())['head_hash']==state_hash(model)
            continue
        n=len(arrays.z[partition])
        errors=np.lib.format.open_memmap(output/f'{partition}_variable_mae.npy',mode='w+',dtype='float64',shape=(n,7))
        with torch.inference_mode():
            for lo in range(0,n,batch):
                z,q=arrays.get(partition,slice(lo,lo+batch))
                predicted=model(z,prior(q))
                errors[lo:lo+len(z)]=(z-predicted).abs().double().mean(1).cpu().numpy()
        errors.flush()
        assert np.isfinite(errors).all()
        atomic_json(marker,dict(head_hash=state_hash(model),windows=n,all_variables_and_time_points=True))
        print('MCR INFERRED',name,partition,n,flush=True)
    train_error=np.load(output/'train_variable_mae.npy',mmap_mode='r')
    scale=np.maximum(np.median(train_error,axis=0),1e-5)
    np.save(output/'normal_relation_error_scale.npy',scale)
    for partition in ['train','validation','test']:
        value=np.load(output/f'{partition}_variable_mae.npy',mmap_mode='r')
        score=(value/scale).mean(1)
        np.save(output/f'{partition}_window_scores.npy',score)
    atomic_json(output/'score_protocol.json',dict(scale='median normal training per-window MAE by variable; floor1e-5',
        score='time/variable mean absolute signed-calibrated-residual prediction error / fixed normal error scale',test_fitting=False))


def evaluate_unit(core,root,unit):
    source,_=source_for(unit);output=root/unit;parent=core/unit/'A4'
    mapping=json.loads((source/'mapping.json').read_text())
    windows=list(HEADS)+['B0_signed_L1']
    frames={}
    for partition in ['train','validation','test']:
        values=np.column_stack([np.load(output/name/f'{partition}_window_scores.npy',mmap_mode='r') for name in HEADS]+
            [np.load(output/f'{partition}_signed_diagonal_score.npy',mmap_mode='r')])
        np.save(output/f'{partition}_window_scores.npy',values)
        refs=np.load(source/'residuals'/partition/'references.npy',mmap_mode='r')
        snippets=aggregate(values,refs,mapping['sample_ids'][partition],windows)
        inherited=read(parent/f'{partition}_snippet_scores.csv').rename(columns={
            'raw':'B0_raw','cal_abs':'B0','cal_signed':'B0_signed','rcr_legacy':'B0_legacy'})
        snippets=inherited.merge(snippets,on='sample_id',validate='one_to_one',suffixes=('','_relation'))
        snippets.to_csv(output/f'{partition}_snippet_scores.csv',index=False)
        relation_rank=rank_frame(snippets,windows,unit.startswith('Brand3'))
        base=read(parent/f'{partition}_scores.csv').rename(columns={
            'raw':'B0_raw','cal_abs':'B0','cal_signed':'B0_signed','rcr_legacy':'B0_legacy'})
        keys=['vin','sample_label'] if unit.startswith('Brand3') else ['sample_id','vin','sample_label']
        frames[partition]=base.merge(relation_rank[keys+windows],on=keys,validate='one_to_one')
        np.testing.assert_array_equal(frames[partition].B0.to_numpy(),base.B0.to_numpy())
    normal=frames['validation']
    assert normal.sample_label.eq(0).all()
    cdfs={name:EmpiricalCDF().fit(normal[name]) for name in ['B0','B3_memory_gat']}
    medians={name:float(np.median(normal[name])) for name in cdfs}
    mads={name:max(float(1.4826*np.median(np.abs(normal[name]-medians[name]))),1e-8) for name in cdfs}
    np.savez(output/'normal_validation_CDFs.npz',**{n:c.reference for n,c in cdfs.items()})
    atomic_json(output/'normal_validation_fusion.json',dict(median=medians,MAD_scale=mads,
        normalizers_fit_on='normal validation ranking objects',alpha=[0,.5,1],primary='B5_ECDF_0p5',
        fusion_stage='after existing window/snippet/vehicle Top5 aggregation',test_fitting=False))
    for partition,frame in frames.items():
        u={n:c.transform(frame[n].to_numpy()) for n,c in cdfs.items()}
        continuous={n:(frame[n].to_numpy()-medians[n])/mads[n] for n in cdfs}
        frame['B0_ECDF'],frame['B3_ECDF']=u['B0'],u['B3_memory_gat']
        frame['B5_ECDF_0p5']=.5*(u['B0']+u['B3_memory_gat'])
        frame['B0_MAD'],frame['B3_MAD']=continuous['B0'],continuous['B3_memory_gat']
        frame['B5_MAD_0p5']=.5*(continuous['B0']+continuous['B3_memory_gat'])
        frame.to_csv(output/f'{partition}_scores.csv',index=False)
    names=['B0','B0_raw','B0_signed','B0_legacy']+windows+['B0_ECDF','B3_ECDF','B5_ECDF_0p5','B0_MAD','B3_MAD','B5_MAD_0p5']
    thresholds,rows={},[]
    for name in names:
        thresholds[name]={}
        for q in [.95,.99]:
            threshold=float(frames['validation'][name].quantile(q));thresholds[name][str(q)]=threshold
            test=frames['test'];fpr,tpr,_=roc_curve(test.sample_label,test[name])
            rows.append(dict(unit=unit,method=name,q=q,threshold=threshold,
                validation_fpr=float(np.mean(frames['validation'][name]>=threshold)),
                recall_at_test_fpr05=float(tpr[fpr<=.05].max(initial=0.)),
                **metrics(test.sample_label,test[name],threshold)))
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False);atomic_json(output/'thresholds.json',thresholds)
    overlap,predictions=[],[]
    for q in [.95,.99]:
        for base in ['B0','B0_signed_L1','B2_gat','B4_memory_mlp']:
            bt,rt=thresholds[base][str(q)],thresholds['B3_memory_gat'][str(q)]
            overlap.append(error_overlap(frames['test'],base,'B3_memory_gat',bt,rt,unit,q))
            f=frames['test'][[c for c in ['sample_id','vin','sample_label','fault_type','cycle_kind'] if c in frames['test']]].copy()
            f['q'],f['base']=q,base
            f['base_score'],f['relation_score']=frames['test'][base],frames['test'].B3_memory_gat
            f['base_alarm'],f['relation_alarm']=f.base_score>=bt,f.relation_score>=rt
            f['set']=np.select([f.base_alarm&f.relation_alarm,f.base_alarm&~f.relation_alarm,~f.base_alarm&f.relation_alarm],
                ['both','base_only','relation_only'],default='neither')
            predictions.append(f)
    pd.DataFrame(overlap).to_csv(output/'error_overlap.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(output/'prediction_sets.csv',index=False)
    atomic_json(output/'complete.json',dict(unit=unit,parent_model='A4',parent_score_exact=True,
        window_score_names=windows,methods=names,heads=list(HEADS),model_one_frozen=True,test_selection=False))
    print('MCR UNIT COMPLETE',unit,flush=True)


def run_unit(core,root,unit,args):
    output=root/unit;output.mkdir(parents=True,exist_ok=True)
    source,_=source_for(unit)
    prepare_signed_z(core,unit,output)
    if not (output/'normal_context_memory_prior.npz').exists():
        q=np.load(source/'residuals/train/condition.npy',mmap_mode='r')
        state=np.load(core/unit/'A4/train_memory_state.npy',mmap_mode='r')
        prior=NormalContextMemoryPrior.fit(q,state)
        prior.save(output/'normal_context_memory_prior.npz')
        atomic_json(output/'context_prior_protocol.json',dict(source='normal training A4 Memory occupancy',
            predictors='current/SOC statistics only; exclude predicted control from its own context',
            fit='fixed ridge1e-3 regression of normal log occupancy; scale-wise softmax',
            full_test_response_memory_used=False,test_fitting=False))
    prior=NormalContextMemoryPrior.load(output/'normal_context_memory_prior.npz').cuda().eval()
    arrays=RelationArrays(output,source)
    length=arrays.z['train'].shape[1]
    set_seed(args.seed)
    shared={k:v.clone() for k,v in MCRSequencePredictor('memory_gat',length).state_dict().items()}
    for name in HEADS:
        path=output/name
        if (path/'complete.json').exists():continue
        set_seed(args.seed);model=build_head(name,length,shared)
        if name=='B3_q_only_gat':
            state=np.load(core/unit/'A4/train_memory_state.npy',mmap_mode='r')
            with torch.no_grad():model.memory_constant.copy_(torch.as_tensor(np.mean(state,axis=0,dtype=np.float64),device='cuda',dtype=torch.float32))
        model=train(model,prior,arrays,path,name,args)
        audit_mask(model,prior,arrays,path,name)
        infer(model,prior,arrays,path,name,args)
        atomic_json(path/'complete.json',dict(head=name,head_hash=state_hash(model),full_data=True,model_one_frozen=True))
        del model;torch.cuda.empty_cache()
    evaluate_unit(core,root,unit)
    del prior,arrays,shared;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',default='runs/internal/mscm_mcr_development_20261025_v2')
    parser.add_argument('--output',default='runs/internal/mcr_gat_development_20261025_v1')
    parser.add_argument('--units',nargs='+',default=UNITS)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--batch',type=int,default=256)
    parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--wait-for-core',action='store_true')
    args=parser.parse_args()
    core,root=Path(args.core),Path(args.output)
    while not (core/'model_one_analysis_complete.json').exists():
        if not args.wait_for_core:raise RuntimeError('Complete/freeze model-one full evaluation first')
        time.sleep(10)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; run outside sandbox')
    torch.set_num_threads(4)
    root.mkdir(parents=True,exist_ok=True)
    protocol=dict(task=TASK,core=str(core),units=args.units,epochs=args.epochs,patience=args.patience,
        batch=args.batch,seed=args.seed,heads=HEADS,TCN=dict(layers=2,kernel=3,hidden=32,temporal_pool_bins=4),
        GAT=dict(heads=2,templates=4,rank=4,query='target identity only',aggregation='single star; self excluded'),
        context='normal training Memory occupancy projected from current/SOC, target control excluded',
        full_multivariate_test_memory=False,loss='signed calibrated residual sequence LOO Huber',
        extra_condition_control='q-only GAT has identical parameters to B3; Memory coordinates fixed to normal training mean occupancy',
        training_mask='one uniformly sampled variable per window per epoch; identical schedule in all heads',
        validation_and_scoring='all targets, all time points, full cohort',
        normal_relation_scale='median normal training window MAE by variable, floor1e-5',
        fusion='normal validation ranking-object ECDF and continuous median/MAD; alpha0/0.5/1',
        primary_fusion='B5_ECDF_0p5',primary_threshold=.95,thresholds=[.95,.99],
        parent_selection='preset A4, unchanged across datasets; no test best-parent selection',
        core_protocol_sha256=digest(core/'protocol_manifest.json'),
        parent_signatures={u:dict(checkpoint=digest(core/u/'A4/training/best.pt'),
            calibration=digest(core/u/'A4/calibration_signed.npz'),absolute_calibration=digest(core/u/'A4/calibration_absolute.npz'),
            test_scores=digest(core/u/'A4/test_scores.csv'),validation_scores=digest(core/u/'A4/validation_scores.csv')) for u in args.units},
        implementations={p:digest(p) for p in [__file__,'src/models/mcr_gat.py','src/models/rcr.py']},
        device=torch.cuda.get_device_name(0),precision='float32',test_selection=False)
    marker=root/'protocol_manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Protocol changed; use new output')
    atomic_json(marker,protocol)
    import shutil
    snapshots=root/'implementation_snapshot';snapshots.mkdir(exist_ok=True)
    for p in protocol['implementations']:shutil.copy2(p,snapshots/Path(p).name)
    with open('/tmp/enhancedmtadgat_mscm_gpu0.lock','a+') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for unit in args.units:
            if not (root/unit/'complete.json').exists():run_unit(core,root,unit,args)
            completed=[read(root/u/'metrics.csv') for u in args.units if (root/u/'complete.json').exists()]
            if completed:pd.concat(completed,ignore_index=True).to_csv(root/'all_completed_metrics.csv',index=False)
    atomic_json(root/'complete.json',dict(units=args.units,heads=list(HEADS),model_one_frozen=True,test_selection=False))
    print('MCR-GAT DEVELOPMENT COMPLETE',flush=True)


if __name__=='__main__':main()
