"""Full-cohort staged MSCM reconstruction mechanism experiments, CUDA only.

First complete model-one reconstruction and postprocessing; model two is a
separate subsequent stage. Existing source directories are never mutated.
"""
import argparse
import copy
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
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

from src.data.ch_battery_utils import load_ch_battery_mixed_cycle_research_split
from src.data.external_protocol import load_external_protocol_data
from src.models.rcr import condition_features
from src.models.mscm_reconstruction import (MSCMReconstructor, FixedRobustResidualCalibration,
    patchify)
from src.runners.run_unified_external_baselines import WindowDataset
from src.runners.run_rcr import atomic_json, atomic_checkpoint, state_hash
from src.runners.run_local_residual_models import source_for
from src.runners.run_rcr_memory_probe import digest
from src.runners.run_rcr_calibration_sensitivity import aggregate, make_calibration
from src.runners.probe_scale_relation_evidence import rank_frame, paired_bootstrap, read
from scripts.evaluate_rcr import metrics
from sklearn.metrics import roc_curve


TASK = 'MSCM-MCR-DEVELOPMENT-20261025-v2'
UNITS = [f'Brand3_fold{i}' for i in range(5)]+['NCM','LFP']
VARIANTS = ['A0','A1','A2','A4','A6']
SCORES = ['raw','cal_abs','cal_signed','rcr_legacy']


def prepare_inputs(unit, root, batch):
    source, _ = source_for(unit)
    output = root/'inputs'/unit
    output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((source/'mapping.json').read_text())
    if not (output/'complete.json').exists():
        brand = unit.startswith('Brand3')
        print('LOAD DATA', unit, flush=True)
        if brand:
            fold = int(unit[-1])
            data = load_external_protocol_data('Brand3',brand_fold=fold,brand_fold_seed=0,
                seed=3407,brand_split_protocol='strict_normal_validation',brand_normalization='minmax')
            sequences = {s:getattr(data,s+'_sequences') for s in ['train','validation','test']}
            ids = data.metadata['sample_ids']
            window, stride = 127,127
        else:
            data = load_ch_battery_mixed_cycle_research_split(chemistry=unit,seed=3407)
            ids = {s:sorted(k for k,v in data[s].items() if len(v)>=32) for s in ['train','validation','test']}
            sequences = {s:[data[s][k] for k in ids[s]] for s in ids}
            window, stride = 32,4
        assert ids == mapping['sample_ids'], 'Source sample cohort changed'
        metadata = pd.read_csv(source/'metadata.csv',dtype={'vin':str,'car':str},float_precision='round_trip')
        assert metadata.loc[metadata.partition.ne('test'),'sample_label'].eq(0).all()
        metadata.to_csv(output/'metadata.csv',index=False)
        for partition, seq in sequences.items():
            ds = WindowDataset(seq,window,stride)
            assert len(ds) == mapping['windows'][partition]
            values = np.lib.format.open_memmap(output/f'{partition}_x.npy',mode='w+',dtype='float32',shape=(len(ds),window,7))
            old_q = np.load(source/'residuals'/partition/'condition.npy',mmap_mode='r')
            refs = np.load(source/'residuals'/partition/'references.npy',mmap_mode='r')
            offset, max_error = 0,0.
            for x, sample, end in DataLoader(ds,batch_size=batch,shuffle=False):
                stop = offset+len(x)
                values[offset:stop] = x.numpy()
                np.testing.assert_array_equal(refs[offset:stop],np.column_stack((sample,end)))
                q = condition_features(x,1,2).numpy()
                max_error = max(max_error,float(np.max(np.abs(q-old_q[offset:stop]))))
                offset = stop
            assert offset == len(ds) and max_error < 2e-6
            values.flush()
            print('INPUT CACHE',unit,partition,len(ds),'q_error',max_error,flush=True)
        atomic_json(output/'mapping.json',mapping)
        atomic_json(output/'complete.json',dict(unit=unit,windows=mapping['windows'],
            source_mapping_sha256=digest(source/'mapping.json'),full_data=True))
        del data,sequences
        gc.collect()
    return output,mapping


class InputArrays:
    def __init__(self, root):
        self.arrays = {s:np.load(root/f'{s}_x.npy',mmap_mode='r') for s in ['train','validation','test']}
        self.resident = {}
        free,_ = torch.cuda.mem_get_info()
        needed = sum(self.arrays[s].nbytes for s in ['train','validation'])
        if needed < .4*free:
            for s in ['train','validation']:
                self.resident[s] = torch.from_numpy(np.array(self.arrays[s])).cuda()
        print('GPU INPUT CACHE',list(self.resident), 'bytes',needed,flush=True)

    def get(self, partition, index):
        if partition in self.resident:
            if isinstance(index,np.ndarray):
                index = torch.from_numpy(index).cuda()
            return self.resident[partition][index]
        return torch.from_numpy(np.array(self.arrays[partition][index])).cuda()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def validation_loss(model, inputs, batch):
    model.eval(); total,count = 0.,0
    with torch.inference_mode():
        for lo in range(0,len(inputs.arrays['validation']),batch):
            x = inputs.get('validation',slice(lo,lo+batch))
            loss = F.huber_loss(model(x),x)
            total += float(loss)*len(x); count += len(x)
    return total/count


def train_stage(model, teacher, inputs, output, stage, epochs, args, memory_weight=.01):
    output.mkdir(parents=True,exist_ok=True)
    model.configure_stage(stage,args.finetune_encoder)
    if (output/'complete.json').exists():
        model.load_state_dict(torch.load(output/'best.pt',map_location='cuda',weights_only=True))
        assert state_hash(model) == json.loads((output/'complete.json').read_text())['model_hash']
        return model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=.001,weight_decay=.0001)
    history,start,stale,best_epoch = [],1,0,0
    if (output/'last.pt').exists():
        saved = torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        history,start,stale,best_epoch,best = saved['history'],saved['epoch']+1,saved['stale'],saved['best_epoch'],saved['best']
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        best = validation_loss(model,inputs,args.batch)
        atomic_checkpoint(output/'best.pt',model.state_dict())
        atomic_json(output/'initial.json',dict(model_hash=state_hash(model),validation_huber=best,
            parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in params)))
    n = len(inputs.arrays['train'])
    for epoch in range(start,epochs+1):
        if stale >= args.patience: break
        clock = time.monotonic();model.train()
        if stage == 'memory': model.backbone.temporal.eval()
        # Identical epoch window permutations in every architecture/stage.
        order = np.random.default_rng(args.seed+epoch).permutation(n)
        rec_total,mem_total,count = 0.,0.,0
        for step,lo in enumerate(range(0,n,args.batch),1):
            x = inputs.get('train',order[lo:lo+args.batch])
            optimizer.zero_grad(set_to_none=True)
            targets = None
            if teacher is not None and model.use_memory and memory_weight:
                with torch.no_grad(): targets = teacher.patches(x)
            reconstruction,mem,_,_ = model(x,details=True,teacher_patches=targets)
            rec = F.huber_loss(reconstruction,x)
            loss = rec+memory_weight*mem
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite MSCM training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,5.);optimizer.step()
            rec_total += float(rec.detach())*len(x);mem_total += float(mem.detach())*len(x);count += len(x)
            if step % 1000 == 0:
                print(json.dumps(dict(stage=stage,unit=str(output.parent.name),epoch=epoch,batch=step,
                    batches=int(np.ceil(n/args.batch)),rec=rec_total/count,mem=mem_total/count)),flush=True)
        val = validation_loss(model,inputs,args.batch)
        if not np.isfinite(val): raise FloatingPointError('Nonfinite normal validation loss')
        if val < best:
            best,stale,best_epoch = val,0,epoch
            atomic_checkpoint(output/'best.pt',model.state_dict())
        else: stale += 1
        row = dict(epoch=epoch,reconstruction_huber=rec_total/count,teacher_memory_mse=mem_total/count,
            normal_validation_huber=val,best_epoch=best_epoch,seconds=time.monotonic()-clock)
        history.append(row);pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        atomic_checkpoint(output/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,history=history,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        print(json.dumps(dict(event='MSCM_EPOCH',path=str(output),**row)),flush=True)
        atomic_json(output/'progress.json',dict(stage=stage,**row))
    model.load_state_dict(torch.load(output/'best.pt',map_location='cuda',weights_only=True))
    atomic_json(output/'complete.json',dict(model_hash=state_hash(model),best_epoch=best_epoch,
        epochs_executed=len(history),normal_validation_huber=best,selection='normal_validation_huber',
        memory_weight=memory_weight,teacher_hash=state_hash(teacher) if teacher is not None else None,
        full_training_windows=n,test_labels_used=False))
    return model.eval()


def initialize_banks(model, teacher, inputs, output, args):
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(inputs.arrays['train']),min(4096,len(inputs.arrays['train'])),replace=False))
    np.save(output/'memory_initialization_normal_window_indices.npy',indices)
    collected = [[] for _ in model.kernels]
    with torch.inference_mode():
        for lo in range(0,len(indices),args.batch):
            for s,values in enumerate(teacher.patches(inputs.get('train',indices[lo:lo+args.batch]))):
                collected[s].append(values.flatten(0,1).cpu().numpy())
    for s,(parts,memory) in enumerate(zip(collected,model.memories)):
        values = np.concatenate(parts)
        selected = rng.choice(len(values),min(32768,len(values)),replace=False)
        memory.initialize(values[selected],args.seed+s)
    atomic_json(output/'memory_initialization.json',dict(normal_windows=len(indices),max_patch_vectors_per_scale=32768,
        normal_training_only=True,teacher_hash=state_hash(teacher),method='normal K-means' if model.variant!='A6' else 'normal PCA low-rank initialization'))


def infer_moments(model, inputs, source, output, mapping, args):
    for partition in ['train','validation','test']:
        done = output/f'{partition}_inference_complete.json'
        if done.exists():
            assert json.loads(done.read_text())['model_hash'] == state_hash(model)
            continue
        n = len(inputs.arrays[partition]); banks = len(model.memories)
        moments = np.lib.format.open_memmap(output/f'{partition}_moments.npy',mode='w+',dtype='float64',shape=(n,4,7))
        residual = np.lib.format.open_memmap(output/f'{partition}_signed_residual.npy',mode='w+',dtype='float32',shape=inputs.arrays[partition].shape) if model.variant=='A4' else None
        states = np.lib.format.open_memmap(output/f'{partition}_memory_state.npy',mode='w+',dtype='float32',shape=(n,banks*8)) if model.use_memory and model.variant!='A6' else None
        distance = np.lib.format.open_memmap(output/f'{partition}_memory_distance.npy',mode='w+',dtype='float32',shape=(n,banks)) if states is not None else None
        hard_counts = np.zeros((banks,8),dtype=np.int64)
        q_sum,q_square,ret_sum,ret_square = [np.zeros((banks,32)) for _ in range(4)]
        patch_count = np.zeros(banks,dtype=np.int64)
        model.eval()
        with torch.inference_mode():
            for lo in range(0,n,args.batch):
                x = inputs.get(partition,slice(lo,lo+args.batch))
                reconstructed,_,state,weights = model(x,details=True)
                r = (x-reconstructed).cpu().numpy().astype(np.float64)
                stop = lo+len(x)
                moments[lo:stop,0],moments[lo:stop,1] = r.mean(1),r.var(1)
                moments[lo:stop,2],moments[lo:stop,3] = np.abs(r).mean(1),np.abs(r).var(1)
                if residual is not None: residual[lo:stop] = r
                if states is not None:
                    states[lo:stop] = state.cpu().numpy()
                    queries = model.patches(x)
                    for s,(query,memory,w) in enumerate(zip(queries,model.memories,weights)):
                        q = (query-memory.center)/memory.scale
                        dist = (q[:,:,None]-memory.prototypes).square().mean(-1).sqrt().amin(-1)
                        distance[lo:stop,s] = dist.mean(1).cpu().numpy()
                        hard_counts[s] += torch.bincount(w.argmax(-1).flatten(),minlength=8).cpu().numpy()
                        retrieved = w@memory.prototypes
                        q_sum[s] += q.sum((0,1)).cpu().numpy();q_square[s] += q.square().sum((0,1)).cpu().numpy()
                        ret_sum[s] += retrieved.sum((0,1)).cpu().numpy();ret_square[s] += retrieved.square().sum((0,1)).cpu().numpy()
                        patch_count[s] += q.shape[0]*q.shape[1]
        moments.flush()
        if residual is not None: residual.flush()
        if states is not None:
            states.flush();distance.flush()
            records = []
            for s,memory in enumerate(model.memories):
                p = memory.prototypes.detach().cpu().numpy()
                cosine = p@p.T/np.maximum(np.linalg.norm(p,axis=1)[:,None]*np.linalg.norm(p,axis=1)[None],1e-12)
                off = ~np.eye(8,dtype=bool)
                soft = np.asarray(states[:,s*8:(s+1)*8]).mean(0)
                hard = hard_counts[s]/max(1,hard_counts[s].sum())
                records.append(dict(scale=s,kernel=model.kernels[s],soft_occupancy=soft.tolist(),hard_occupancy=hard.tolist(),
                    effective_soft_prototypes=float(np.exp(-(soft*np.log(np.maximum(soft,1e-30))).sum())),
                    hard_prototypes_above_one_percent=int((hard>=.01).sum()),
                    pairwise_cosine_mean=float(cosine[off].mean()),
                    pairwise_distance_mean=float(np.linalg.norm(p[:,None]-p[None],axis=-1)[off].mean()),
                    query_variance_mean=float((q_square[s]/patch_count[s]-(q_sum[s]/patch_count[s])**2).mean()),
                    retrieved_variance_mean=float((ret_square[s]/patch_count[s]-(ret_sum[s]/patch_count[s])**2).mean()),
                    nearest_distance_quantiles=np.quantile(distance[:,s],[.05,.5,.95,.99]).tolist()))
            atomic_json(output/f'{partition}_memory_diagnostics.json',records)
        atomic_json(done,dict(model_hash=state_hash(model),windows=n,full_data=True,signed_residual_saved=residual is not None))
        print('MSCM INFERRED',output.name,partition,n,flush=True)


def evaluate_model(unit, variant, inputs, source, output, mapping):
    train = np.load(output/'train_moments.npy',mmap_mode='r')
    qtrain = np.asarray(np.load(source/'residuals/train/condition.npy',mmap_mode='r'),dtype=np.float64)
    signed = FixedRobustResidualCalibration().fit(qtrain,train[:,0],train[:,1])
    absolute = FixedRobustResidualCalibration().fit(qtrain,train[:,2],train[:,3])
    signed.save(output/'calibration_signed.npz');absolute.save(output/'calibration_absolute.npz')
    rms = np.sqrt(np.maximum(train[:,1]+np.square(train[:,0]-signed.location(qtrain)),0))
    legacy,_ = make_calibration(qtrain,signed.q_mean,signed.q_std,signed.coef,rms,.05)
    legacy.save(output/'calibration_legacy.npz')
    metadata = pd.read_csv(source/'metadata.csv',dtype={'vin':str,'car':str},float_precision='round_trip').set_index('sample_id',verify_integrity=True)
    if unit.startswith('Brand3'): metadata['vin'] = metadata.car.astype(str)
    frames = {}
    for partition in ['train','validation','test']:
        m = np.load(output/f'{partition}_moments.npy',mmap_mode='r')
        q = np.asarray(np.load(source/'residuals'/partition/'condition.npy',mmap_mode='r'),dtype=np.float64)
        mu,scale = legacy.parameters(q)
        values = np.column_stack(((m[:,1]+m[:,0]**2).mean(1),absolute.energy(q,m[:,2],m[:,3]),
            signed.energy(q,m[:,0],m[:,1]),((m[:,1]+(m[:,0]-mu)**2)/scale**2).mean(1)))
        assert np.isfinite(values).all()
        np.save(output/f'{partition}_window_scores.npy',values)
        refs = np.load(source/'residuals'/partition/'references.npy',mmap_mode='r')
        snippets = aggregate(values,refs,mapping['sample_ids'][partition],SCORES)
        snippets = metadata.loc[mapping['sample_ids'][partition]].reset_index().merge(snippets,on='sample_id',validate='one_to_one')
        snippets.to_csv(output/f'{partition}_snippet_scores.csv',index=False)
        frames[partition] = rank_frame(snippets,SCORES,unit.startswith('Brand3'))
        frames[partition].to_csv(output/f'{partition}_scores.csv',index=False)
    rows,thresholds = [],{}
    test = frames['test']
    for name in SCORES:
        thresholds[name] = {}
        for q in [.95,.99]:
            threshold = float(frames['validation'][name].quantile(q)); thresholds[name][str(q)] = threshold
            fpr,tpr,_ = roc_curve(test.sample_label,test[name])
            rows.append(dict(unit=unit,variant=variant,score=name,q=q,threshold=threshold,
                recall_at_test_fpr05=float(tpr[fpr<=.05].max(initial=0.)),
                recall_at_test_fpr01=float(tpr[fpr<=.01].max(initial=0.)),
                **metrics(test.sample_label,test[name],threshold)))
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    atomic_json(output/'thresholds.json',thresholds)
    atomic_json(output/'complete.json',dict(unit=unit,variant=variant,normal_only_calibration=True,
        primary_score='cal_abs',full_data=True,test_labels_used_for_selection=False))
    print('MSCM UNIT COMPLETE',unit,variant,flush=True)


def run_unit(unit, root, args):
    source,_ = source_for(unit)
    input_root,mapping = prepare_inputs(unit,root,args.batch)
    arrays = InputArrays(input_root)
    pretrained = torch.load(source/'backbone_best.pt',map_location='cpu',weights_only=True)
    teachers = {}
    for kind,variant in [('single','A0'),('multi','A1')]:
        set_seed(args.seed)
        model = MSCMReconstructor(variant,patch=args.patch).cuda();model.warm_start(pretrained)
        model = train_stage(model,None,arrays,root/unit/('pretrain_'+kind),'pretrain',args.pretrain_epochs,args,0.)
        for p in model.parameters(): p.requires_grad_(False)
        teachers[kind] = model.eval()
    for variant in VARIANTS:
        output = root/unit/variant
        output.mkdir(parents=True,exist_ok=True)
        if (output/'complete.json').exists(): continue
        set_seed(args.seed)
        teacher = teachers['single' if variant in ['A0','A2'] else 'multi']
        model = MSCMReconstructor(variant,patch=args.patch,gamma=args.gamma,fusion=args.fusion).cuda()
        missing,unexpected = model.load_state_dict(teacher.state_dict(),strict=False)
        assert not unexpected and all(k.startswith('memories.') for k in missing)
        if model.use_memory and not (output/'training/last.pt').exists() and not (output/'training/complete.json').exists():
            initialize_banks(model,teacher,arrays,output,args)
        model = train_stage(model,teacher,arrays,output/'training','memory',args.memory_epochs,args,args.memory_weight)
        infer_moments(model,arrays,source,output,mapping,args)
        evaluate_model(unit,variant,arrays,source,output,mapping)
        del model;torch.cuda.empty_cache()
    del teachers,arrays,pretrained
    gc.collect();torch.cuda.empty_cache()
    atomic_json(root/unit/'complete.json',dict(unit=unit,variants=VARIANTS,A5_alias='A4/raw',model_one_frozen=True))


def summarize(root, units, repeats=500):
    frames = []
    for unit in units:
        for variant in VARIANTS:
            p = root/unit/variant
            if (p/'complete.json').exists(): frames.append(read(p/'metrics.csv'))
    if not frames: return
    metrics_ = pd.concat(frames,ignore_index=True)
    metrics_['dataset'] = np.where(metrics_.unit.str.startswith('Brand3'),'Brand3',metrics_.unit)
    metrics_.to_csv(root/'all_completed_metrics.csv',index=False)
    metrics_.groupby(['dataset','variant','score','q'])[['ap','auroc','f1','recall','fpr','recall_at_test_fpr05']].agg(['mean','std']).to_csv(root/'completed_macro_summary.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/internal/mscm_mcr_development_20261025_v2')
    parser.add_argument('--units',nargs='+',default=UNITS)
    parser.add_argument('--pretrain-epochs',type=int,default=5)
    parser.add_argument('--memory-epochs',type=int,default=5)
    parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--batch',type=int,default=256)
    parser.add_argument('--patch',type=int,default=8)
    parser.add_argument('--gamma',type=float,default=.1)
    parser.add_argument('--memory-weight',type=float,default=.01)
    parser.add_argument('--fusion',choices=['bounded','pure','direct'],default='bounded')
    parser.add_argument('--finetune-encoder',action='store_true')
    parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--summarize-only',action='store_true')
    args = parser.parse_args()
    root = Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:
        summarize(root,args.units);return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; run outside sandbox')
    torch.set_num_threads(4)
    settings = dict(task=TASK,units=args.units,pretrain_epochs=args.pretrain_epochs,memory_epochs=args.memory_epochs,
        patience=args.patience,batch=args.batch,seed=args.seed,patch=args.patch,stride=4,kernels=[3,7,15],slots=8,
        gamma=args.gamma,memory_weight=args.memory_weight,fusion=args.fusion,finetune_encoder=args.finetune_encoder,
        teacher='frozen stage-A normal representation',
        teacher_loss_reduction='sum_scales mean_batch_patch sum_features squared_normalized_error',
        stage_A_frozen='embedding, variable graph, token projection',
        stage_B_frozen='embedding, graph, temporal transformer, multiscale encoder unless explicit ablation',
        capacity_control='A6 low-rank ordinary bottleneck exactly matches A4 total and trainable parameter count',
        scores=SCORES,primary_score='cal_abs',thresholds=[.95,.99],full_training=True,full_evaluation=True,
        fit_calibration_on='same normal train cohort, no independent calibration holdout',
        max_training_epochs_per_arm=args.pretrain_epochs+args.memory_epochs,
        source_manifests={u:digest(source_for(u)[0]/'mapping.json') for u in args.units},
        implementations={p:digest(p) for p in [__file__,'src/models/mscm_reconstruction.py','src/models/rcr.py']},
        gpu=torch.cuda.get_device_name(0),precision='float32',test_selection=False,stage='model_one_core')
    p = root/'protocol_manifest.json'
    if p.exists() and json.loads(p.read_text()) != settings: raise ValueError('Protocol differs; use a new output directory')
    atomic_json(p,settings)
    snapshots = root/'implementation_snapshot';snapshots.mkdir(exist_ok=True)
    import shutil
    for path in settings['implementations']: shutil.copy2(path,snapshots/Path(path).name)
    with open('/tmp/enhancedmtadgat_mscm_gpu0.lock','a+') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for unit in args.units:
            if not (root/unit/'complete.json').exists(): run_unit(unit,root,args)
            summarize(root,args.units)
    atomic_json(root/'model_one_core_complete.json',dict(units=args.units,variants=VARIANTS,
        A3_alias='A1',A5_alias='A4/raw',model_two_started=False))
    print('MSCM MODEL ONE CORE COMPLETE',flush=True)


if __name__ == '__main__':
    main()
