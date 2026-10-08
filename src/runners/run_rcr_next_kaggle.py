"""Four fixed Kaggle shards; cache-preserving head experiments and vehicle evaluation."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import torch

from src.runners.run_rcr import atomic_json
from src.runners.run_rcr_relation import file_hash

CONFIGS = {'B1': ('full', 'mlp'), 'B2': ('transformer_only', 'mlp'),
           'B3': ('dense_mlp', 'mlp'), 'B4': ('transformer_only', 'none'),
           'B5': ('transformer_only', 'linear')}
LEGACY = {'B1': 'net_continuous_mlp', 'B2': 'net_ablation_transformer_only_continuous_mlp',
          'B3': 'net_ablation_dense_mlp_continuous_mlp'}


def archive(unit, root, output):
    relative = unit.relative_to(root)
    target = output/'units'/f'{relative.as_posix().replace("/", "__")}.tar.gz'
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.tmp')
    def keep(info):
        if Path(info.name).name in {'standardized_residual.npy', 'condition_standardized.npy'}:
            return None
        return info
    with tarfile.open(temp, 'w:gz', compresslevel=1) as packed:
        packed.add(unit, arcname=relative.as_posix(), filter=keep)
    temp.replace(target)
    record = dict(unit=str(relative), archive=target.name, sha256=file_hash(target), bytes=target.stat().st_size)
    atomic_json(target.with_suffix('.json'), record)
    return record


def score_json(unit):
    rows = pd.read_csv(unit/'metrics.csv').to_dict('records')
    atomic_json(unit/'metrics.json', rows)


def vehicle_evaluate(root, unit):
    from scripts.evaluate_rcr import metrics
    from src.runners.run_rcr_calibration_sensitivity import aggregate
    mapping = json.loads((root/'mapping.json').read_text())
    frames = {}
    names = ['raw', 'magnitude', 'relation', 'magnitude_percentile', 'relation_percentile', 'combined']
    for partition in ['validation', 'test']:
        scores = np.load(unit/partition/'scores.npy', mmap_mode='r')
        refs = np.load(root/'residuals'/partition/'references.npy', mmap_mode='r')
        snippet = aggregate(scores, refs, mapping['sample_ids'][partition], names)
        assert len(snippet) == len(mapping[partition+'_cars'])
        snippet['car'] = mapping[partition+'_cars']; snippet['label'] = mapping[partition+'_labels']
        vehicles = []
        for car, group in snippet.groupby('car'):
            assert group.label.nunique() == 1
            values = group[names].to_numpy()
            k = max(1, int(np.ceil(.05*len(values))))
            top = np.partition(values, -k, axis=0)[-k:].mean(0)
            vehicles.append(dict(car=str(car), label=int(group.label.iloc[0]), snippets=len(group),
                                 **dict(zip(names, top))))
        snippet.to_csv(unit/f'{partition}_snippet_scores.csv', index=False)
        frames[partition] = pd.DataFrame(vehicles)
        frames[partition].to_csv(unit/f'{partition}_vehicle_scores.csv', index=False)
    assert frames['validation'].label.eq(0).all()
    rows = []
    for name in names:
        for q in (.95, .99):
            threshold = float(frames['validation'][name].quantile(q))
            rows.append(dict(score=name, q=q, threshold=threshold,
                **metrics(frames['test'].label, frames['test'][name], threshold)))
    pd.DataFrame(rows).to_csv(unit/'metrics.csv', index=False)
    atomic_json(unit/'metrics.json', rows)
    atomic_json(unit/'evaluation_complete.json', dict(ranking_level='vehicle',
        threshold_source='normal_validation_vehicle', thresholds=[.95, .99],
        aggregation='snippet_top5pct_then_vehicle_top5pct', no_test_tuning=True))


def execute_unit(root, unit_name, config, seed, output, brand=False, reuse=False):
    unit = root/unit_name
    unit.mkdir(exist_ok=True)
    if not (unit/'complete.json').exists():
        if reuse:
            source = root/LEGACY[config]
            protocol = json.loads((source/'protocol.json').read_text())
            assert protocol['seed'] == seed == 3407 and protocol['epochs'] == 30
            assert protocol['condition'] == 'continuous' and protocol['modulation'] == 'mlp'
            assert (source/'complete.json').exists()
            shutil.copytree(source, unit, dirs_exist_ok=True)
            atomic_json(unit/'reuse.json', dict(source=source.name, immutable_scores=True,
                source_head_sha256=file_hash(source/'best.pt'), cache_sha_manifest='frozen_manifest.json'))
        else:
            architecture, modulation = CONFIGS[config]
            command = [sys.executable, '-u', '-m', 'src.runners.run_rcr_relation',
                '--root', str(root), '--architecture', architecture, '--modulation', modulation,
                '--condition', 'continuous', '--epochs', '30', '--seed', str(seed),
                '--device', 'cuda', '--output-unit', unit_name, '--no-save-errors']
            subprocess.run(command, check=True)
    if brand:
        vehicle_evaluate(root, unit)
    else:
        from scripts.evaluate_rcr import evaluate
        meta = pd.read_csv(root/'evaluation_metadata.csv')
        evaluate(root, unit, meta); score_json(unit)
    if not (unit/'evaluation_complete.json').exists():
        raise RuntimeError('Missing evaluated unit marker')
    record = dict(config=config, seed=seed, dataset='Brand3' if brand else root.parent.name,
                  reused=reuse, frozen_backbone_sha256=file_hash(root/'backbone_best.pt'))
    atomic_json(unit/'experiment.json', record)
    return archive(unit, root.parent.parent, output)


def linear_loo(root, output):
    """Normal-selected ridge leave-one-variable-out with point-level sufficient moments."""
    from src.models.rcr import EmpiricalCDF
    from src.models.rcr_fitting import slices
    from scripts.evaluate_rcr import evaluate
    lite = root/'lite_continuous'; unit = root/'linear_loo'; unit.mkdir(exist_ok=True)
    d = np.load(lite/'train/standardized_residual.npy', mmap_mode='r').shape[-1]
    qdim = np.load(lite/'train/condition_standardized.npy', mmap_mode='r').shape[-1]
    p = d-1+qdim+1
    xtx, xty, count = np.zeros((d,p,p)), np.zeros((d,p)), 0
    def blocks(partition):
        z = np.load(lite/partition/'standardized_residual.npy', mmap_mode='r')
        q = np.load(lite/partition/'condition_standardized.npy', mmap_mode='r')
        for ix in slices(len(z), 1024):
            a = np.asarray(z[ix], dtype='float64').reshape(-1,d)
            b = np.repeat(np.asarray(q[ix], dtype='float64'), z.shape[1], axis=0)
            yield ix, z[ix].shape, a, b
    def design(z, q, target):
        return np.column_stack([np.delete(z,target,axis=1),q,np.ones(len(z))])
    for _, _, z, q in blocks('train'):
        for i in range(d):
            x=design(z,q,i);xtx[i]+=x.T@x;xty[i]+=x.T@z[:,i]
        count+=len(z)
    candidates, losses = {}, {}
    penalty=np.eye(p);penalty[-1,-1]=0
    # Standardize predictors with normal-training moments. Otherwise the fixed
    # ridge grid over-penalizes tiny residual coordinates relative to unit-scale q.
    transformations=[]
    for i in range(d):
        gram=xtx[i]/count;mean=gram[-1,:-1]
        std=np.maximum(np.sqrt(np.maximum(np.diag(gram)[:-1]-mean*mean,0)),1e-8)
        transform=np.eye(p);transform[:-1,:-1]=np.diag(1/std)
        transform[-1,:-1]=-mean/std;transformations.append(transform)
    for lam in (1e-4,1e-2,1.):
        candidates[lam]=np.stack([transformations[i]@np.linalg.solve(
            transformations[i].T@(xtx[i]/count)@transformations[i]+lam*penalty,
            transformations[i].T@(xty[i]/count)) for i in range(d)])
        total,n=0.,0
        for _, _, z,q in blocks('validation'):
            for i in range(d):total+=float(np.square(z[:,i]-design(z,q,i)@candidates[lam][i]).sum())
            n+=z.size
        losses[lam]=total/n
    lam=min(losses,key=losses.get);coefficients=candidates[lam]
    atomic_json(unit/'selection.json',dict(lambdas=[1e-4,1e-2,1.],validation_loo=losses,
                                          selected=lam,predictors_standardized='normal_training_only',test_labels_used=False))
    np.savez(unit/'linear_coefficients.npz',coefficients=coefficients,lambda_selected=lam)
    cdf=None
    names=['raw','magnitude','relation','magnitude_percentile','relation_percentile','combined']
    for partition in ['train','validation','test']:
        source=np.load(lite/partition/'scores.npy',mmap_mode='r')
        directory=unit/partition;directory.mkdir(exist_ok=True)
        score=np.lib.format.open_memmap(directory/'scores.npy',mode='w+',dtype='float64',shape=source.shape)
        score[:]=source
        for ix,shape,z,q in blocks(partition):
            prediction=np.column_stack([design(z,q,i)@coefficients[i] for i in range(d)])
            score[ix,2]=np.square(z-prediction).reshape(shape).mean((1,2))
        if partition=='train':
            cdf=EmpiricalCDF().fit(score[:,2]);np.save(unit/'relation_reference.npy',cdf.reference)
        score[:,4]=cdf.transform(score[:,2]);score[:,5]=.5*(score[:,3]+score[:,4]);score.flush()
        assert np.isfinite(score).all()
        atomic_json(directory/'complete.json',dict(windows=len(score),score_names=names))
    atomic_json(unit/'complete.json',dict(normal_training_only=True,selected_lambda=lam))
    evaluate(root,unit,pd.read_csv(root/'evaluation_metadata.csv'));score_json(unit)
    return archive(unit,root.parent.parent,output)


def prepare_brand_fold(fold, bundle, work):
    from types import SimpleNamespace
    from src.data.external_protocol import load_external_protocol_data
    from src.data.nc_battery import FEATURE_NAMES
    from src.runners.run_unified_external_baselines import WindowDataset
    from src.runners.run_rcr import backbone_train, cache_residuals
    root=work/'Brand3'/f'fold{fold}'/'full';root.mkdir(parents=True,exist_ok=True)
    frozen=bundle/'brand_frozen'/f'fold{fold}'/'full'
    if frozen.is_dir():
        for name in ['backbone_best.pt','backbone_complete.json','mapping.json','protocol.json']:
            shutil.copy2(frozen/name,root/name)
    data=load_external_protocol_data('Brand3',brand_fold=fold,brand_fold_seed=0,seed=3407,
        brand_split_protocol='strict_normal_validation',brand_normalization='minmax')
    sequences=dict(train=data.train_sequences,validation=data.validation_sequences,test=data.test_sequences)
    datasets={p:WindowDataset(s,127,127,seed=3407) for p,s in sequences.items()}
    if not frozen.is_dir():
        mapping=dict(sample_ids={p:[f'{p}_{i:05d}' for i in range(len(s))] for p,s in sequences.items()},
            feature_columns=list(FEATURE_NAMES),control_indices=[1,2],
            windows={p:len(ds) for p,ds in datasets.items()},
            validation_cars=data.validation_entity_ids,test_cars=data.entity_ids,
            validation_labels=[int(y[0]) for y in data.validation_labels],
            test_labels=[int(y[0]) for y in data.test_labels])
        atomic_json(root/'mapping.json',mapping)
        atomic_json(root/'protocol.json',dict(dataset='Brand3',fold=fold,seed=3407,
            split='strict_normal_validation',normalization=data.metadata['normalization'],
            window=127,stride=127,feature_graph=True,temporal=True,epochs=50,patience=8,
            lr=.001,weight_decay=.0001,batch_size=128,smoke=False,
            epoch_selection='normal_validation_only',no_test_selection=True,
            backbone_provenance='retrained_once_then_shared_frozen; old remote output unavailable'))
    else:
        mapping=json.loads((root/'mapping.json').read_text())
    assert mapping['feature_columns']==list(FEATURE_NAMES)
    assert mapping['test_cars']==data.entity_ids and mapping['validation_cars']==data.validation_entity_ids
    assert mapping['test_labels']==[int(y[0]) for y in data.test_labels]
    assert all(len(ds)==mapping['windows'][p] for p,ds in datasets.items())
    assert not set(data.validation_entity_ids)&set(data.entity_ids)
    assert set(mapping['validation_labels'])=={0} and set(mapping['test_labels'])=={0,1}
    torch.manual_seed(3407);torch.cuda.manual_seed_all(3407);np.random.seed(3407)
    args=SimpleNamespace(seed=3407,no_feature_graph=False,no_temporal=False,epochs_backbone=50)
    model=backbone_train(args,root,datasets,7)
    cache_residuals(model,datasets,mapping['sample_ids'],(1,2),root)
    del model,data,datasets,sequences;gc.collect();torch.cuda.empty_cache()
    subprocess.run([sys.executable,'-u','-m','src.runners.run_rcr_lite','--root',str(root),'--condition','continuous'],check=True)
    return root


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shard',choices=['ch_a','ch_b','brand024','brand13'])
    p.add_argument('--bundle',type=Path)
    p.add_argument('--work',type=Path,default=Path('/kaggle/temp/rcr_next_work'))
    p.add_argument('--output',type=Path,default=Path('/kaggle/working/artifacts/rcr_next'))
    p.add_argument('--unit-root',type=Path)
    p.add_argument('--unit-name');p.add_argument('--config',choices=list(CONFIGS))
    p.add_argument('--seed',type=int);p.add_argument('--brand',action='store_true');p.add_argument('--reuse',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    if a.unit_root:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'unit_records').mkdir(exist_ok=True)
        r=execute_unit(a.unit_root,a.unit_name,a.config,a.seed,a.output,a.brand,a.reuse)
        atomic_json(a.output/'unit_records'/f'{a.unit_root.parent.name}_{a.unit_name}.json',r)
        return
    if not a.shard or not a.bundle or not torch.cuda.is_available():
        raise RuntimeError('Explicit shard, verified input bundle and CUDA required')
    a.output.mkdir(parents=True,exist_ok=True);a.work.mkdir(parents=True,exist_ok=True)
    shutil.copy2(a.bundle/'frozen_manifest.json',a.output/'frozen_manifest.json')
    state=dict(shard=a.shard,status='running',completed=[],started=time.time(),
        account='chenmjdaisy' if a.shard.startswith('ch_') else 'daisychen2',
        gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        source_hash=file_hash(Path(__file__)),backbone_seed=3407,head_seeds=[3407,3408,3409],
        primary_alpha=.5,thresholds=[.95,.99],no_test_selection=True)
    manifest=a.output/'protocol_manifest.json';atomic_json(manifest,state)
    jobs=[]
    try:
        if a.shard.startswith('ch_'):
            seeds=[3407,3408] if a.shard=='ch_a' else [3409]
            roots=[]
            for chemistry in ['LFP','NCM']:
                root=a.work/chemistry/'full';root.parent.mkdir(parents=True,exist_ok=True)
                checkpoint_root=a.bundle/'ch_checkpoints'/chemistry/'full'
                rebuild=checkpoint_root.is_dir()
                if rebuild:
                    raw=next((x.parent for x in Path('/kaggle/input').rglob('LFP')
                              if x.is_dir() and (x.parent/'NCM').is_dir()),None)
                    if raw is None:raise FileNotFoundError('Own-account CH raw LFP/NCM not mounted')
                    os.environ['MTAD_GAT_CH_BATTERY_ROOT']=str(raw)
                    root.mkdir(exist_ok=True)
                    for name in ['backbone_best.pt','backbone_complete.json','protocol.json','mapping.json','metadata.csv','evaluation_metadata.csv']:
                        shutil.copy2(checkpoint_root/name,root/name)
                    subprocess.run([sys.executable,'-u','-m','src.runners.run_rcr','--chemistry',chemistry,
                        '--output',str(root),'--stage','residuals','--seed','3407'],check=True)
                    subprocess.run([sys.executable,'-u','-m','src.runners.run_rcr_lite','--root',str(root),
                                    '--condition','continuous'],check=True)
                else:
                    shutil.copytree(a.bundle/'ch_frozen'/chemistry/'full',root,dirs_exist_ok=True,
                                    copy_function=os.link)
                # Exercise the real CH evaluation path before expensive head jobs.
                from scripts.evaluate_rcr import evaluate
                evaluate(root, root/'lite_continuous', pd.read_csv(root/'evaluation_metadata.csv'))
                score_json(root/'lite_continuous')
                hashes={}
                for partition in ['train','validation','test']:
                    for name in ['standardized_residual.npy','condition_standardized.npy']:
                        path=root/'lite_continuous'/partition/name
                        hashes[f'{partition}/{name}']=file_hash(path)
                atomic_json(root/'cache_hashes.json',dict(rebuilt_on_kaggle=rebuild,files=hashes))
                state.setdefault('cache_hashes',{})[chemistry]=hashes
                roots.append(root)
                for seed in seeds:
                    for config in CONFIGS:
                        jobs.append(dict(root=root,name=f'head_{config}_seed{seed}',config=config,seed=seed,
                                         brand=False,reuse=not rebuild and seed==3407 and config in LEGACY))
        else:
            raw=next((x.parent for x in Path('/kaggle/input').rglob('battery_brand3')
                      if (x/'label').is_dir() and (x/'data').is_dir()),None)
            if raw is None:raise FileNotFoundError('Own-account Brand3 raw dataset not mounted')
            os.environ['MTAD_GAT_TSINGHUA_EV_ROOT']=str(raw)
            folds=[0,2,4] if a.shard=='brand024' else [1,3]
            state['folds']=folds;roots=[]
            for fold in folds:
                root=prepare_brand_fold(fold,a.bundle,a.work);roots.append(root)
                vehicle_evaluate(root,root/'lite_continuous')
                state['completed'].append(archive(root/'lite_continuous',root.parent.parent,a.output))
                for seed in [3407,3408,3409]:
                    for config in ['B1','B4']:
                        jobs.append(dict(root=root,name=f'head_{config}_seed{seed}',config=config,seed=seed,brand=True,reuse=False))
        state['expected_units']=[dict(dataset='Brand3' if j['brand'] else j['root'].parent.name,
            fold=j['root'].parent.name if j['brand'] else None,unit=j['name'],config=j['config'],seed=j['seed']) for j in jobs]
        for root in roots:
            destination=a.output/'sources'/root.relative_to(a.work)
            destination.mkdir(parents=True,exist_ok=True)
            for name in ['mapping.json','protocol.json','backbone_complete.json','metadata.csv','evaluation_metadata.csv','cache_hashes.json']:
                if (root/name).exists():shutil.copy2(root/name,destination/name)
            atomic_json(destination/'frozen_hashes.json',dict(backbone=file_hash(root/'backbone_best.pt'),
                calibration=file_hash(root/'lite_continuous/calibration.npz')))
            for partition in ['train','validation','test']:
                d=destination/'residuals'/partition;d.mkdir(parents=True,exist_ok=True)
                for name in ['references.npy','complete.json']:shutil.copy2(root/'residuals'/partition/name,d/name)
        atomic_json(manifest,state)
        logs=a.output/'logs';logs.mkdir(exist_ok=True)
        active={};pending=list(jobs)
        while pending or active:
            for gpu in range(torch.cuda.device_count()):
                if gpu in active or not pending:continue
                job=pending.pop(0);key=f'{job["root"].parent.name}_{job["name"]}'
                log=logs/(key+'.log');handle=log.open('w')
                command=[sys.executable,'-u','-m','src.runners.run_rcr_next_kaggle',
                    '--unit-root',str(job['root']),'--unit-name',job['name'],'--config',job['config'],
                    '--seed',str(job['seed']),'--output',str(a.output)]
                if job['brand']:command.append('--brand')
                if job['reuse']:command.append('--reuse')
                process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'})
                active[gpu]=(process,handle,job,key)
                print('START',key,'GPU',gpu,flush=True)
            for gpu,(process,handle,job,key) in list(active.items()):
                code=process.poll()
                if code is None:continue
                handle.close();del active[gpu]
                if code:
                    raise RuntimeError(f'{key} failed exit={code}; see logs/{key}.log')
                record=json.loads((a.output/'unit_records'/f'{key}.json').read_text())
                state['completed'].append(record);atomic_json(manifest,state)
                print('DONE',key,flush=True)
                shutil.rmtree(job['root']/job['name'])
            if active:time.sleep(2)
        if a.shard=='ch_a':
            for root in roots:
                state['completed'].append(linear_loo(root,a.output))
                from scripts.evaluate_rcr import evaluate
                evaluate(root,root/'lite_continuous',pd.read_csv(root/'evaluation_metadata.csv'))
                score_json(root/'lite_continuous')
                state['completed'].append(archive(root/'lite_continuous',root.parent.parent,a.output))
        state['status']='complete'
    except Exception as exc:
        for process,handle,_,_ in locals().get('active',{}).values():
            process.terminate();handle.close()
        state['status'],state['error']='failed',repr(exc)
        raise
    finally:
        state['ended']=time.time();atomic_json(manifest,state)


if __name__=='__main__':main()
