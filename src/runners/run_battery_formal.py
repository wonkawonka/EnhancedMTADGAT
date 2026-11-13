"""Approved battery comparison/ablation cells over frozen prepared inputs.

One process executes one declared cell. Kaggle orchestration lives in notebooks.
Run --list to inspect cells; --smoke exercises methods without formal data.
"""
import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,f1_score
from threadpoolctl import threadpool_limits
from src.experiment_preflight import Preflight,sha256
from src.experiments.battery_data import Bundle,write_json,window_starts,check,PARTS
from src.experiments.artifacts import Efficiency,package_cell
from src.experiments import reference_models as ref
from src.models.omni_anomaly_port import TF1Adam,clip_gradients_per_parameter

STOP=False

def request_stop(signum, frame):
    global STOP
    STOP=True


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def checkpoint(path, value):
    tmp=Path(str(path)+'.tmp');torch.save(value,tmp);tmp.replace(path)


def top_mean(values):
    n=max(1,math.ceil(len(values)*.05))
    return float(np.partition(values,len(values)-n)[-n:].mean())


def object_scores(snippets, mapping, dataset):
    if dataset!='Brand1':
        return snippets,np.asarray(mapping['labels']),mapping['sample_ids']
    groups={}
    for i,(entity,label) in enumerate(zip(mapping['entities'],mapping['labels'])):
        groups.setdefault(entity,[]).append(i)
    values=[];labels=[];ids=[]
    for entity,indices in groups.items():
        labs=np.asarray(mapping['labels'])[indices]
        check(np.unique(labs).size==1,'Inconsistent entity labels')
        values.append(top_mean(snippets[indices]));labels.append(labs[0]);ids.append(entity)
    return np.asarray(values),np.asarray(labels),ids


def evaluate(values, validation, labels):
    threshold=float(np.quantile(validation,.95,method='linear'));pred=values>=threshold
    return dict(AP=float(average_precision_score(labels,values)),AUROC=float(roc_auc_score(labels,values)),
                Precision=float(precision_score(labels,pred,zero_division=0)),Recall=float(recall_score(labels,pred,zero_division=0)),
                F1=float(f1_score(labels,pred,zero_division=0)),FPR=float(pred[labels==0].mean()),threshold=threshold,
                validation_FPR=float((validation>=threshold).mean()),test_objects=len(labels),fault_objects=int(labels.sum()))


def source_manifest(plan):
    root=ref.ROOT
    paths=['src/runners/run_battery_formal.py','src/experiments/reference_models.py','src/experiments/battery_data.py',
           'src/experiments/artifacts.py','src/experiments/battery_smoke.py','src/models/rcr.py','src/models/omni_anomaly_port.py',
           'src/models/unified_external_baselines.py','src/runners/run_rcr_revalidation.py','src/experiment_preflight.py']
    hashes={p:sha256(root/p) for p in paths}
    provenance=json.loads((ref.SOURCE/'provenance.json').read_text())
    for group,record in provenance.items():
        check(plan['reference_versions'][group]==(record.get('commit') or record.get('ref')),'Unexpected reference revision')
        for name,expected in record['files'].items():
            check(sha256(ref.SOURCE/group/name)==expected,'Modified pinned reference: '+group+'/'+name)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=root,text=True,capture_output=True)
    return dict(git_commit=commit.stdout.strip() if commit.returncode==0 else None,files=hashes,reference_versions=plan['reference_versions'])


def optimizer_for(method, model, cfg, stage):
    pars=model.encoder.parameters() if method=='svdd' and stage=='main' else model.parameters()
    if method=='omni':return TF1Adam(pars,lr=cfg['lr'])
    cls=torch.optim.AdamW if cfg['optimizer']=='AdamW' else torch.optim.Adam
    return cls(pars,lr=cfg['lr'],weight_decay=cfg['weight_decay'],betas=(.9,.999),eps=1e-8,amsgrad=False)


def stage_better(method, values, previous):
    if previous is None:return True
    if method=='at':return all(a<=b for a,b in zip(values,previous))
    return values[0]<previous[0]


def train_model(model,method,cfg,policy,bundle,out,device,meter,gate,identity,max_seconds):
    global STOP
    train_positions,_=window_starts(bundle.lengths['train'],cfg['span'],cfg['train_stride'])
    val_positions,_=window_starts(bundle.lengths['validation'],cfg['span'],cfg['validation_stride'])
    dtype=np.float64 if cfg['precision']=='float64' else np.float32
    kl=ref.kl_for(method) if method in ('at','dc') else None
    stages=[('pretrain',5),('main',30)] if method=='svdd' else [('main',cfg['formal_epochs'])]
    history=[];center=None;resume=None;started=time.monotonic()
    if (out/'latest.pt').exists():
        resume=torch.load(out/'latest.pt',map_location=device,weights_only=False)
        check(resume['identity']==identity,'Recovery belongs to another protocol/code/data')
        model.load_state_dict(resume['model']);history=resume['history'];center=resume['center']
    last_checkpoint=time.monotonic();last_progress=0
    for si,(stage,epochs) in enumerate(stages):
        if resume is not None and si<resume['stage_index']:continue
        if method=='svdd' and stage=='main' and center is None:
            model.eval();total=torch.zeros(32,device=device,dtype=torch.float64);count=0
            with meter.phase('SVDD_center_fit'),torch.no_grad():
                for lo in range(0,len(train_positions),cfg['batch']):
                    x=torch.as_tensor(bundle.batch('train',train_positions[lo:lo+cfg['batch']],cfg['span'],dtype),device=device)
                    h=model.encoder(x);total+=h.double().sum(0);count+=len(x)
            center=(total/count).float();small=center.abs()<.1
            center[small & (center<0)]=-.1;center[small & (center>=0)]=.1
            np.save(out/'center.npy',center.cpu().numpy())
        optimizer=optimizer_for(method,model,cfg,stage)
        model.zero_grad(set_to_none=True)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=.9) if method=='tranad' else None
        control={'best':None,'counter':0,'selected_epoch':None}
        start_epoch=1;start_step=0;loss_sum=0.;seen=0
        if resume is not None and si==resume['stage_index']:
            optimizer.load_state_dict(resume['optimizer'])
            if scheduler is not None:scheduler.load_state_dict(resume['scheduler'])
            control=resume['control'];start_epoch=resume['epoch'];start_step=resume['next_step'];loss_sum=resume['loss_sum'];seen=resume['seen']
            restore_rng(resume['rng'])
        def save_latest(epoch,next_step,loss_sum,seen):
            with meter.phase('checkpoint_IO',stage=stage):
                checkpoint(out/'latest.pt',dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict() if scheduler else None,rng=rng_state(),stage_index=si,stage=stage,
                    epoch=epoch,next_step=next_step,loss_sum=loss_sum,seen=seen,history=history,control=control,center=center))
        for epoch in range(start_epoch,epochs+1):
            positions=train_positions
            if cfg['shuffle']:positions=positions[np.random.default_rng(cfg['seed']+epoch-1+si*1000).permutation(len(positions))]
            omitted=0
            if method=='omni':omitted=len(positions)%cfg['batch'];positions=positions[:len(positions)-omitted]
            nb=math.ceil(len(positions)/cfg['batch']);model.train()
            seen_before=seen;step_before=start_step
            with meter.phase('train',stage=stage,epoch=epoch,windows=len(positions)-seen,updates=nb-start_step,omitted_tail_windows=omitted,
                             omitted_tail_starts=train_positions[np.random.default_rng(cfg['seed']+epoch-1).permutation(len(train_positions))][-omitted:].tolist() if omitted else []) as phase:
                for step in range(start_step,nb):
                    ids=positions[step*cfg['batch']:(step+1)*cfg['batch']]
                    x=torch.as_tensor(bundle.batch('train',ids,cfg['span'],dtype),device=device)
                    optimizer.zero_grad(set_to_none=True)
                    terms=ref.terms(method,model,x,epoch,kl,center,stage)
                    check(all(torch.isfinite(v).item() for v in terms),'Nonfinite training loss')
                    for j,loss in enumerate(terms):loss.backward(retain_graph=j<len(terms)-1)
                    gradients=[p.grad for p in model.parameters() if p.grad is not None]
                    check(gradients and all(torch.isfinite(g).all().item() for g in gradients),'Nonfinite/absent gradients')
                    if step==0 and epoch==1:
                        gate.check('first_batch_'+stage,any(torch.count_nonzero(g).item()>0 for g in gradients),
                                   {'loss':[v.item() for v in terms],'gradient_tensors':len(gradients)})
                    if method=='omni':clip_gradients_per_parameter(model,10.)
                    if method=='svdd' and stage=='main':torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),5.,error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum+=sum(v.item() for v in terms)*len(x);seen+=len(x)
                    if time.monotonic()-last_progress>30:
                        write_json(out/'progress.json',dict(stage=stage,epoch=epoch,step=step+1,total_steps=nb,loss=loss_sum/seen))
                        print(json.dumps({'cell':out.name,'phase':stage,'epoch':epoch,'step':step+1,'steps':nb,'loss':loss_sum/seen}),flush=True);last_progress=time.monotonic()
                    if STOP or (max_seconds and time.monotonic()-started>=max_seconds) or time.monotonic()-last_checkpoint>180:
                        phase['windows']=seen-seen_before;phase['updates']=step+1-step_before
                        save_latest(epoch,step+1,loss_sum,seen);last_checkpoint=time.monotonic()
                        if STOP or (max_seconds and time.monotonic()-started>=max_seconds):
                            write_json(out/'status.json',dict(status='paused_with_full_recovery',stage=stage,epoch=epoch,next_step=step+1))
                            return False,center
                phase['windows']=seen-seen_before;phase['updates']=nb-step_before
            model.eval();sums=None;count=0
            training_rng=rng_state()
            with meter.phase('validation',stage=stage,epoch=epoch,windows=len(val_positions)),torch.no_grad():
                generator=torch.Generator(device=device).manual_seed(cfg['seed']+100000)
                for lo in range(0,len(val_positions),cfg['validation_batch']):
                    x=torch.as_tensor(bundle.batch('validation',val_positions[lo:lo+cfg['validation_batch']],cfg['span'],dtype),device=device)
                    vals=ref.terms(method,model,x,epoch,kl,center,stage,generator)
                    check(all(torch.isfinite(v).item() for v in vals),'Nonfinite validation objective')
                    if sums is None:sums=np.zeros(len(vals))
                    sums+=np.asarray([v.item() for v in vals])*len(x);count+=len(x)
            restore_rng(training_rng)
            values=(sums/count).tolist()
            if method=='tranad':scheduler.step()
            elif method in ('at','dc'):
                for group in optimizer.param_groups:group['lr']=cfg['lr']*.5**(epoch-1)
            history.append(dict(stage=stage,epoch=epoch,train_loss=loss_sum/seen,validation=values,lr_next=optimizer.param_groups[0]['lr']))
            write_json(out/'history.json',history)
            final_only=method in ('tranad','dc','svdd')
            better=stage_better(method,values,control['best'])
            if better:control['best']=values;control['counter']=0
            else:control['counter']+=1
            if final_only or better:
                control['selected_epoch']=epoch
                if stage=='main':checkpoint(out/'selected.pt',model.state_dict())
            patience=policy.get('early_stopping_patience')
            next_epoch=epochs+1 if patience and control['counter']>=patience else epoch+1
            save_latest(next_epoch,0,0.,0);last_checkpoint=time.monotonic()
            write_json(out/'selection.json',dict(stage=stage,**control,selection=policy['selection']))
            start_step=0;loss_sum=0.;seen=0
            if next_epoch>epochs:break
        resume=None
    model.load_state_dict(torch.load(out/'selected.pt',map_location=device,weights_only=True));model.eval()
    return True,center


def calibrate(model,cfg,bundle,out,device,meter):
    positions,_=window_starts(bundle.lengths['train'],cfg['span'],cfg['train_stride'])
    q=np.empty((len(positions),8),dtype=np.float64);moments=np.empty((len(positions),2,7),dtype=np.float64)
    with meter.phase('RCR_calibration',windows=len(positions)),torch.no_grad():
        for lo in range(0,len(positions),cfg['batch']):
            x=torch.as_tensor(bundle.batch('train',positions[lo:lo+cfg['batch']],cfg['span']),device=device)
            r=(x-model(x)).cpu().numpy().astype(np.float64)
            q[lo:lo+len(x)]=ref.condition_features(x,1,2).cpu().numpy()
            moments[lo:lo+len(x),0]=r.mean(1);moments[lo:lo+len(x),1]=r.var(1)
        with threadpool_limits(4):cal=ref.fit_scale(q,moments,.05)
        cal.save(out/'calibration.npz')
    return cal


def score_part(method,model,cfg,bundle,part,out,device,meter,cal,center):
    positions,counts=window_starts(bundle.lengths[part],cfg['span'],cfg['inference_stride'])
    allpoint=method in ('at','dc','iforest')
    native_counts=bundle.lengths[part] if allpoint else counts
    boundaries=np.r_[0,np.cumsum(native_counts)]
    native=np.zeros(int(native_counts.sum()),dtype=np.float64)
    overlap=np.zeros(len(native),dtype=np.int16) if method in ('at','dc') else None
    dtype=np.float64 if cfg['precision']=='float64' else np.float32
    with meter.phase(part+'_scoring',windows=len(positions),snippets=len(counts)),torch.no_grad():
        if method=='iforest':
            for lo in range(0,len(native),65536):native[lo:lo+65536]=-model.score_samples(bundle.point_batch(part,lo,lo+65536))
        else:
            kl=ref.kl_for(method) if method in ('at','dc') else None
            generator=torch.Generator(device=device).manual_seed(cfg['seed']+(100000 if part=='validation' else 200000))
            last=time.monotonic()
            for lo in range(0,len(positions),cfg['inference_batch']):
                pos=positions[lo:lo+cfg['inference_batch']]
                x=torch.as_tensor(bundle.batch(part,pos,cfg['span'],dtype),device=device)
                scores=ref.scores(method,model,x,kl,cal,center,generator).cpu().numpy()
                check(np.isfinite(scores).all(),'Nonfinite native anomaly scores')
                if overlap is not None:
                    indices=pos[:,None]+np.arange(cfg['span'])[None,:]
                    np.add.at(native,indices,scores);np.add.at(overlap,indices,1)
                else:native[lo:lo+len(scores)]=scores
                if time.monotonic()-last>30:
                    print(json.dumps({'phase':part+'_score','windows':lo+len(x),'total':len(positions)}),flush=True);last=time.monotonic()
    if overlap is not None:
        check(np.all(overlap>0),'Unscored real point in AT/DC')
        native/=overlap
    with meter.phase(part+'_aggregation'):
        snippets=np.asarray([top_mean(native[a:b]) for a,b in zip(boundaries[:-1],boundaries[1:])])
        values,labels,ids=object_scores(snippets,bundle.mapping[part],cfg['dataset'])
        np.save(out/f'{part}_native.npy',native)
        np.savez_compressed(out/f'{part}_native_mapping.npz',boundaries=boundaries,window_starts=positions if not allpoint else np.empty(0,dtype=np.int64))
        np.savez_compressed(out/f'{part}_scores.npz',snippet_scores=snippets,object_scores=values,object_labels=labels,
                            object_ids=np.asarray(ids,dtype=str),sample_ids=np.asarray(bundle.mapping[part]['sample_ids'],dtype=str),
                            entities=np.asarray(bundle.mapping[part]['entities'],dtype=str),sample_labels=np.asarray(bundle.mapping[part]['labels']))
    return values,labels


def fit_iforest(cfg,bundle,out,meter):
    with meter.phase('IF_fit',points=len(bundle.raw['train'])):
        x=np.empty(bundle.raw['train'].shape,dtype=np.float32)
        for lo in range(0,len(x),1000000):x[lo:lo+1000000]=bundle.point_batch('train',lo,lo+1000000)
        forest=IsolationForest(n_estimators=200,max_samples=256,max_features=1.0,bootstrap=False,
                               contamination='auto',random_state=cfg['seed'],n_jobs=-1,warm_start=False).fit(x)
        joblib.dump(forest,out/'forest.joblib',compress=3)
        meter.record['forest_trees']=len(forest.estimators_)
        meter.record['forest_nodes']=sum(t.tree_.node_count for t in forest.estimators_)
    return forest


def smoke(out,device,dataset='LFP',experiments=None):
    from src.experiments.battery_smoke import run_smoke
    return run_smoke(out,device,dataset,experiments)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path('configs/external/battery_kaggle_formal_20261112_v1.json'))
    p.add_argument('--bundle',type=Path)
    p.add_argument('--output',type=Path,default=Path('artifacts'))
    p.add_argument('--dataset',choices=['LFP','NCM','Brand1'])
    p.add_argument('--experiment')
    p.add_argument('--list',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-experiments',help='Comma-separated approved experiment IDs; defaults to all in this dataset')
    p.add_argument('--smoke-evidence',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-seconds',type=float,default=0)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    if args.list:
        print(json.dumps(plan['cells'],ensure_ascii=False,indent=2));return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    device=torch.device(args.device)
    if args.smoke:
        smoke(args.output,device,args.dataset or 'LFP',args.smoke_experiments.split(',') if args.smoke_experiments else None);return
    check(plan['approved'] and plan['runnable'],'Unapproved plan')
    check(args.bundle is not None and args.dataset and args.experiment,'Explicit bundle/dataset/experiment required')
    job=next(j for j in plan['cells'] if j['dataset']==args.dataset and j['experiment']==args.experiment)
    method=job['method'];cfg=job['config'];policy=job['training_policy']
    if method=='iforest':device=torch.device('cpu')
    else:check(device.type=='cuda' and torch.cuda.is_available(),'CUDA required for formal neural training')
    out=args.output/args.dataset/args.experiment/f"seed{cfg['seed']}";out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        complete=json.loads((out/'complete.json').read_text())
        check(complete['plan_sha256']==sha256(args.plan),'Completed cell from different plan')
        for name,h in complete['artifacts_sha256'].items():check(sha256(out/name)==h,'Completed artifact changed: '+name)
        print('Already complete and verified:',out);return
    old_gate=json.loads((out/'preflight.json').read_text()) if (out/'preflight.json').exists() else None
    meter=Efficiency(out,device);gate=Preflight(out,args.plan,'approved formal cell')
    if old_gate:
        check(old_gate['plan_sha256']==sha256(args.plan),'Previous preflight belongs to another plan')
        gate.record=old_gate;gate.flush()
    try:
        source=source_manifest(plan)
        check(args.smoke_evidence is not None,'Run bounded smoke on actual hardware before formal cell')
        evidence=json.loads(args.smoke_evidence.read_text())
        check(evidence['status']=='passed' and evidence['files']==source['files'],'Missing/stale numerical smoke evidence')
        tested=evidence.get('experiments',{}).get(args.experiment,{})
        check(evidence.get('dataset')==args.dataset and tested.get('config')==cfg and tested.get('architecture_overrides')==job.get('architecture_overrides',{}),'Smoke must cover this actual dataset/model configuration')
        if method!='iforest':check(evidence['torch']==torch.__version__ and evidence['device']==str(device),'Smoke environment/device changed')
        gate.check('source_numerical_evidence',True,{'path':str(args.smoke_evidence),'sha256':sha256(args.smoke_evidence)})
        with meter.phase('input_load_verify'):
            bundle=Bundle(args.bundle,args.dataset,plan['data'][args.dataset],verify=True)
            expected=plan['bundle_manifests'][args.dataset]
            check(sha256(bundle.root/'bundle_manifest.json')==expected,'Unexpected data bundle revision')
        identity=dict(plan_sha256=sha256(args.plan),source=source,bundle_manifest_sha256=expected,job=job,
                      numerical_environment={'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda})
        write_json(out/'effective_config.json',job);write_json(out/'plan_snapshot.json',plan)
        write_json(out/'protocol_manifest.json',identity)
        gate.check('formal_plan_approved',plan['approved'] and plan['runnable'],plan['approval_record'])
        gate.check('frozen_input_and_config',True,{'counts':bundle.manifest['counts'],'points':bundle.manifest['points'],'config':cfg})
        seed_all(cfg['seed'])
        model=None;center=None;cal=None
        if method=='iforest':model=fit_iforest(cfg,bundle,out,meter)
        else:
            architecture=copy.deepcopy(plan['models'][method]);architecture.update(job.get('architecture_overrides',{}))
            if method=='svdd':architecture['window']=cfg['window']
            with meter.phase('model_initialize'):
                model,compatibility=ref.build(method,device,architecture)
                gate.check('reference_model_compatibility',True,compatibility)
                meter.record['parameters_total']=sum(t.numel() for t in model.parameters())
                meter.record['parameters_trainable']=sum(t.numel() for t in model.parameters() if t.requires_grad)
                init=hashlib.sha256()
                for name,tensor in model.state_dict().items():
                    init.update(name.encode());init.update(tensor.detach().cpu().numpy().tobytes())
                gate.check('fresh_initialization_seed',torch.initial_seed()==cfg['seed'],{'seed':cfg['seed'],'initial_state_sha256':init.hexdigest()})
                if method=='svdd':
                    meter.record['pretrain_parameters_total']=meter.record['parameters_total']
                    meter.record['encoder_parameters']=sum(t.numel() for t in model.encoder.parameters())
                    meter.record['parameters_total']=meter.record['encoder_parameters']
                    meter.record['parameters_trainable']=meter.record['encoder_parameters']
            seed_all(cfg['seed'])
            if job['retrain']:
                ready,center=train_model(model,method,cfg,policy,bundle,out,device,meter,gate,identity,args.max_seconds)
                if not ready:
                    meter.close();package_cell(out,args.output/'archives',f'{args.dataset}_{args.experiment}_s{cfg["seed"]}');return 75
            else:
                parent=args.output/args.dataset/'R1_full'/f"seed{cfg['seed']}"
                check((parent/'complete.json').exists(),'R6 requires completed matching R1_full in output root')
                parent_manifest=json.loads((parent/'protocol_manifest.json').read_text())
                check(parent_manifest['plan_sha256']==identity['plan_sha256'] and parent_manifest['bundle_manifest_sha256']==expected and parent_manifest['source']==source,'R6 parent mismatch')
                model.load_state_dict(torch.load(parent/'selected.pt',map_location=device,weights_only=True))
                checkpoint(out/'selected.pt',model.state_dict())
                write_json(out/'parent.json',dict(path=str(parent),checkpoint_sha256=sha256(parent/'selected.pt')))
            model.eval()
            if method=='rcr' and job['retrain']:cal=calibrate(model,cfg,bundle,out,device,meter)
        val,_=score_part(method,model,cfg,bundle,'validation',out,device,meter,cal,center)
        values,labels=score_part(method,model,cfg,bundle,'test',out,device,meter,cal,center)
        with meter.phase('metrics'):
            result=evaluate(values,val,labels)
            write_json(out/'metrics.json',dict(**result,dataset=args.dataset,experiment=args.experiment,seed=cfg['seed'],evaluation_level='entity' if args.dataset=='Brand1' else 'snippet'))
        meter.close()
        gate.require(['formal_plan_approved','source_numerical_evidence','frozen_input_and_config']+
                     ([] if method=='iforest' else ['reference_model_compatibility','fresh_initialization_seed'])+
                     (['first_batch_main'] if job['retrain'] and method!='iforest' else [])+
                     (['first_batch_pretrain'] if method=='svdd' else []))
        completion=dict(status='complete',plan_sha256=sha256(args.plan),artifacts_sha256={f.name:sha256(f) for f in out.iterdir() if f.name in ('selected.pt','forest.joblib','center.npy','calibration.npz','metrics.json','validation_scores.npz','test_scores.npz','validation_native.npy','test_native.npy','validation_native_mapping.npz','test_native_mapping.npz')})
        write_json(out/'completion_candidate.json',dict(completion,status='metrics_ready_not_yet_archived'))
        write_json(out/'status.json',dict(status='archiving_verified_metrics'))
        packaging_started=time.monotonic()
        index=package_cell(out,args.output/'archives',f'{args.dataset}_{args.experiment}_s{cfg["seed"]}')
        write_json(out/'packaging.json',dict(seconds=time.monotonic()-packaging_started,archive_bytes=sum(x['bytes'] for x in index['archives'])))
        write_json(out/'complete.json',completion)
        write_json(out/'status.json',dict(status='complete'))
        try:
            package_cell(out,args.output/'archives',f'{args.dataset}_{args.experiment}_s{cfg["seed"]}',include_groups=['results'])
        except Exception:
            (out/'complete.json').unlink(missing_ok=True)
            raise
        print(json.dumps(result),flush=True)
    except Exception:
        (out/'error.txt').write_text(traceback.format_exc())
        write_json(out/'status.json',dict(status='failed',error=traceback.format_exc().splitlines()[-1]))
        meter.close()
        package_cell(out,args.output/'archives',f'{args.dataset}_{args.experiment}_s{cfg["seed"]}')
        raise
    return 0


if __name__=='__main__':
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    sys.exit(main())
