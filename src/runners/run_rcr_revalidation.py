"""Immutable config-driven, fresh-training RCR replication/ablation queue.

Use --dry-run to validate and expand the plan without CUDA.  --execute must be
run outside the sandbox; --wait-predecessor checks every predecessor artifact.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from scripts.evaluate_rcr import metrics
from src.models.rcr import RCRBackbone, LinearResidualCalibration
from src.runners.full_validation_support import ranked_frames
from src.runners.run_local_residual_models import source_for
from src.runners.run_mscm_development import InputArrays
from src.runners.run_rcr import atomic_checkpoint, atomic_json, state_hash


DEFAULT_PLAN = 'configs/internal/rcr_revalidation_20261107_v2.json'
PARTITIONS = ('train', 'validation', 'test')
METRICS = ['ap', 'auroc', 'precision', 'recall', 'f1', 'fpr']
MODEL_ARGS = ['hidden', 'feature_graph', 'temporal', 'layers', 'heads', 'ffn_dim', 'dropout']


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def value_id(value):
    return str(value).replace('.', 'p')


def dataset_for(unit):
    return unit.split('_fold', 1)[0]


def plan_source(unit, plan):
    if 'source_paths' in plan['data']:
        return Path(plan['data']['source_paths'][unit]).resolve()
    return source_for(unit)[0]


def expand_plan(plan):
    """Expand true training jobs only; score ablations depend on R1 outputs."""
    assert plan['schema_version'] == 1
    assert plan['seeds'] == [3407, 3408, 3409]
    assert plan['units'] and len(plan['units']) == len(set(plan['units']))
    assert plan['data']['split_seed'] == 3407 and plan['data']['brand_fold_seed'] == 0
    assert plan['common_args']['fresh_initialization']
    assert plan['common_args']['epochs'] == 30 and plan['common_args']['patience'] == 8
    assert plan['scoring']['primary'] == 'scale'
    assert plan['scoring']['threshold_quantile'] == .95
    assert not plan['scoring']['test_based_selection'] and not plan['scoring']['cal_abs']
    assert all(len(a['values']) == 3 and a['default'] in a['values'] for a in plan['sensitivity']['training_axes'])
    floor = plan['sensitivity']['scoring_axis']
    assert len(floor['values']) in (1, 3) and floor['default'] == plan['scoring']['min_std']
    assert floor['default'] in floor['values']
    jobs = []
    for experiment in plan['experiments']:
        seeds = experiment.get('seeds', plan['seeds'])
        assert set(seeds).issubset(plan['seeds'])
        for seed in seeds:
            for unit in plan['units']:
                args = dict(plan['common_args'], **experiment['args'])
                jobs.append(dict(name=experiment['name'], stage=experiment['stage'], seed=seed,
                                 unit=unit, args=args))
    for axis in plan['sensitivity']['training_axes']:
        assert plan['common_args'][axis['parameter']] == axis['default']
        for value in axis['values']:
            if value == axis['default']: continue  # Reuse R1, never retrain a default.
            for seed in plan['sensitivity']['seeds']:
                for unit in plan['units']:
                    args = dict(plan['common_args']); args[axis['parameter']] = value
                    jobs.append(dict(name=axis['name']+'_'+value_id(value), stage='sensitivity',
                        seed=seed, unit=unit, args=args, axis=axis['parameter'], value=value))
    for job in jobs:
        job['id'] = f'{job["name"]}/{job["unit"]}/seed{job["seed"]}'
        assert job['args']['hidden'] % job['args']['heads'] == 0
    assert len(jobs) == len({j['id'] for j in jobs}) == plan['expected']['training_jobs']
    return jobs


def score_names(job, plan):
    names = ['scale']
    if job['name'] == 'R1_full':
        if any(item['score'] == 'no_calibration' and job['seed'] in item['seeds']
               for item in plan['evaluation_only']):
            names += ['no_calibration']
        names += ['scale_floor_'+value_id(v) for v in plan['sensitivity']['scoring_axis']['values']
                  if v != plan['scoring']['min_std']]
    return names


def preflight(plan, jobs):
    records = {}
    for unit in plan['units']:
        source = plan_source(unit, plan)
        inputs = Path(plan['data']['input_cache'])/unit
        mapping = json.loads((source/'mapping.json').read_text())
        assert json.loads((inputs/'mapping.json').read_text()) == mapping
        assert json.loads((inputs/'complete.json').read_text())['source_mapping_sha256'] == file_hash(source/'mapping.json')
        metadata = pd.read_csv(source/'metadata.csv', dtype={'vin':str,'car':str}).set_index('sample_id')
        for part in ['train','validation']:
            assert metadata.loc[mapping['sample_ids'][part]].sample_label.eq(0).all()
        for part in PARTITIONS:
            array = np.load(inputs/f'{part}_x.npy', mmap_mode='r')
            dataset = dataset_for(unit)
            assert array.shape == (mapping['windows'][part], plan['data']['windows'][dataset], 7)
            assert np.load(source/'residuals'/part/'references.npy',mmap_mode='r').shape == (len(array),2)
        records[unit] = dict(source=str(source), input=str(inputs),
            mapping_sha256=file_hash(source/'mapping.json'), metadata_sha256=file_hash(source/'metadata.csv'),
            normalization_source_sha256=file_hash(source/'protocol.json'), windows=mapping['windows'])
    expected_scores = sum(len(score_names(j,plan)) for j in jobs)
    assert expected_scores == plan['expected']['total_metric_cells']
    return records


def predecessor_ready(plan):
    if not plan['execution'].get('predecessor'):
        return True
    root = Path(plan['execution']['predecessor'])
    status = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    if status.get('status') == 'failed': raise RuntimeError('Predecessor failed; successor will not start')
    if not (root/'complete.json').exists() or status.get('status') != 'complete': return False
    count = 0
    for unit in plan['units']:
        for arm in ['RCR_anchor', 'decoder_retrain', 'capacity_control', 'memory_concat']:
            path = root/'units'/unit/arm
            if not (path/'complete.json').exists(): return False
            for name in ['metrics.csv','thresholds.json','test_scores.csv','test_snippet_scores.csv','test_window_scores.npy']:
                if not (path/name).exists(): raise RuntimeError(f'Incomplete predecessor artifact: {path/name}')
            if arm != 'RCR_anchor':
                if not (path/'training/complete.json').exists(): return False
                count += 1
    assert count == plan['execution']['predecessor_training_jobs']
    return True


def validate(model, arrays, batch):
    model.eval(); total, count = 0.,0
    with torch.inference_mode():
        for lo in range(0,len(arrays.arrays['validation']),batch):
            x = arrays.get('validation',slice(lo,lo+batch))
            total += float(F.huber_loss(model(x),x,reduction='sum'));count += x.numel()
    return total/count


def fit_network(model, arrays, job, output):
    """Fresh init, or resume this exact job including every RNG state."""
    output.mkdir(parents=True,exist_ok=True);args=job['args']
    optimizer = torch.optim.AdamW(model.parameters(),lr=args['lr'],weight_decay=args['weight_decay'])
    best, best_epoch, stale, start, history = float('inf'),0,0,1,[]
    if (output/'complete.json').exists():
        model.load_state_dict(torch.load(output/'best.pt',map_location='cuda',weights_only=True))
        assert state_hash(model) == json.loads((output/'complete.json').read_text())['model_hash']
        return model.eval()
    if (output/'last.pt').exists():
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best,best_epoch,stale,start,history=saved['best'],saved['best_epoch'],saved['stale'],saved['epoch']+1,saved['history']
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    else:
        atomic_json(output/'initial.json',dict(model_hash=state_hash(model),fresh_initialization=True,
            pretrained_weights_loaded=False,parameters=sum(p.numel() for p in model.parameters()),seed=job['seed']))
    n=len(arrays.arrays['train']);steps=0
    for epoch in range(start,args['epochs']+1):
        if stale>=args['patience']:break
        clock=time.monotonic();model.train();total,count=0.,0
        order=np.random.default_rng(job['seed']+epoch).permutation(n)
        for lo in range(0,n,args['batch_size']):
            x=arrays.get('train',order[lo:lo+args['batch_size']])
            optimizer.zero_grad(set_to_none=True);loss=F.huber_loss(model(x),x)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite training loss: {job["id"]}')
            loss.backward();optimizer.step();steps+=1
            total+=float(loss.detach())*len(x);count+=len(x)
            if (lo//args['batch_size']+1)%2000==0:
                print(json.dumps(dict(event='RCR_BATCH',job=job['id'],epoch=epoch,
                    windows=count,total=n,train_huber=total/count)),flush=True)
        current=validate(model,arrays,args['batch_size'])
        if not np.isfinite(current):raise FloatingPointError('Nonfinite validation loss')
        if current<best:
            best,best_epoch,stale=current,epoch,0;atomic_checkpoint(output/'best.pt',model.state_dict())
        else:stale+=1
        row=dict(epoch=epoch,train_huber=total/count,validation_huber=current,best_epoch=best_epoch,
            seconds=time.monotonic()-clock,optimizer_steps=int(np.ceil(n/args['batch_size'])),lr=args['lr'])
        history.append(row);pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        atomic_checkpoint(output/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            best=best,best_epoch=best_epoch,stale=stale,epoch=epoch,history=history,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state()))
        print(json.dumps(dict(event='RCR_REVALIDATION_EPOCH',job=job['id'],**row)),flush=True)
    if best_epoch==0:raise RuntimeError('No trained checkpoint selected')
    atomic_checkpoint(output/'final.pt',model.state_dict())
    model.load_state_dict(torch.load(output/'best.pt',map_location='cuda',weights_only=True))
    atomic_json(output/'complete.json',dict(model_hash=state_hash(model),best_epoch=best_epoch,
        validation_huber=best,epochs_executed=len(history),optimizer_steps=sum(r['optimizer_steps'] for r in history),
        fresh_initialization=True,pretrained_weights_loaded=False,
        checkpoint_selection='minimum normal-validation Huber',test_labels_used=False))
    return model.eval()


def infer_moments(model, arrays, output, batch):
    signature=state_hash(model)
    for part in PARTITIONS:
        marker=output/f'{part}_inference.json'
        if marker.exists():
            assert json.loads(marker.read_text())['model_hash']==signature
            continue
        n=len(arrays.arrays[part])
        moment=np.lib.format.open_memmap(output/f'{part}_moments.npy',mode='w+',dtype='float64',shape=(n,2,7))
        model.eval()
        with torch.inference_mode():
            for lo in range(0,n,batch):
                x=arrays.get(part,slice(lo,lo+batch));r=(x-model(x)).cpu().numpy().astype('float64')
                moment[lo:lo+len(x),0]=r.mean(1);moment[lo:lo+len(x),1]=r.var(1)
        moment.flush()
        atomic_json(marker,dict(model_hash=signature,windows=n,columns=['signed_residual_mean','signed_residual_variance']))
        print('RCR_INFERRED',output,part,n,flush=True)


def fit_scale(q, moments, floor):
    """Original OLS/log-RMS estimator from sufficient residual moments."""
    q=np.asarray(q,dtype='float64');cal=LinearResidualCalibration(floor)
    cal.q_mean=q.mean(0);cal.q_std=np.maximum(q.std(0),1e-8);design=cal._design(q)
    cal.mean_coef=np.linalg.lstsq(design,moments[:,0],rcond=None)[0]
    energy=np.maximum(moments[:,1]+np.square(moments[:,0]-design@cal.mean_coef),0)
    target=np.log(np.maximum(np.sqrt(energy),floor))
    cal.scale_coef=np.linalg.lstsq(design,target,rcond=None)[0]
    predicted=np.exp(np.clip(design@cal.scale_coef,-30,30))
    cal.scale_low=np.maximum(np.quantile(predicted,.05,axis=0),floor)
    cal.scale_high=np.maximum(np.quantile(predicted,.95,axis=0),cal.scale_low)
    return cal


def evaluate_job(job, plan, output, source):
    names=score_names(job,plan);floor=plan['scoring']['min_std']
    train=np.load(output/'train_moments.npy',mmap_mode='r')
    train_q=np.load(source/'residuals/train/condition.npy',mmap_mode='r')
    floors=[floor]+([v for v in plan['sensitivity']['scoring_axis']['values'] if v!=floor] if job['name']=='R1_full' else [])
    calibrations={v:fit_scale(train_q,train,v) for v in floors}
    for v,c in calibrations.items():c.save(output/f'calibration_{value_id(v)}.npz')
    windows={}
    for part in PARTITIONS:
        moments=np.load(output/f'{part}_moments.npy',mmap_mode='r')
        q=np.asarray(np.load(source/'residuals'/part/'condition.npy',mmap_mode='r'),dtype='float64')
        values=[]
        for name in names:
            if name=='no_calibration':value=(moments[:,1]+moments[:,0]**2).mean(1)
            else:
                f=floor if name=='scale' else float(name.removeprefix('scale_floor_').replace('p','.'))
                mu,sigma=calibrations[f].parameters(q)
                value=((moments[:,1]+np.square(moments[:,0]-mu))/np.square(sigma)).mean(1)
            if not np.isfinite(value).all():raise FloatingPointError('Nonfinite RCR scores')
            values.append(value)
        windows[part]=np.column_stack(values);np.save(output/f'{part}_window_scores.npy',windows[part])
    mapping=json.loads((source/'mapping.json').read_text())
    frames,snippets=ranked_frames(source,mapping,windows,names)
    assert frames['validation'].sample_label.eq(0).all()
    rows=[];thresholds={};q=plan['scoring']['threshold_quantile']
    for part in PARTITIONS:
        frames[part].to_csv(output/f'{part}_scores.csv',index=False)
        snippets[part].to_csv(output/f'{part}_snippet_scores.csv',index=False)
    for name in names:
        threshold=float(frames['validation'][name].quantile(q));thresholds[name]=dict(q=q,value=threshold)
        rows.append(dict(job=job['id'],name=job['name'],unit=job['unit'],seed=job['seed'],method=name,q=q,
            threshold=threshold,validation_fpr=float((frames['validation'][name]>=threshold).mean()),
            **metrics(frames['test'].sample_label,frames['test'][name],threshold)))
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    atomic_json(output/'metrics.json',rows);atomic_json(output/'thresholds.json',thresholds)
    atomic_json(output/'scoring_audit.json',dict(primary='scale',score_names=names,
        calibration_fitted_on='all original normal training windows',signed_residual=True,
        floor_values=floors,threshold_partition='normal validation',test_selection=False,
        mapping_sha256=file_hash(source/'mapping.json')))


def run_job(job,plan,root):
    output=root/'jobs'/job['id'];output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():return
    record=output/'job.json'
    if record.exists():assert json.loads(record.read_text())==job
    else:atomic_json(record,job)
    seed=job['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    args=job['args'];source=plan_source(job['unit'],plan)
    arrays=InputArrays(Path(plan['data']['input_cache'])/job['unit'])
    model=RCRBackbone(7,**{k:args[k] for k in MODEL_ARGS}).cuda()
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    model=fit_network(model,arrays,job,output/'training')
    infer_moments(model,arrays,output,args['batch_size'])
    evaluate_job(job,plan,output,source)
    atomic_json(output/'mapping.json',json.loads((source/'mapping.json').read_text()))
    atomic_json(output/'complete.json',dict(job=job['id'],fresh_training=True,score_names=score_names(job,plan),
        model_hash=state_hash(model),seconds=time.monotonic()-started,
        peak_cuda_allocated=int(torch.cuda.max_memory_allocated()),test_selection=False))
    del model,arrays;gc.collect();torch.cuda.empty_cache()


def collect(root,jobs):
    return [j for j in jobs if (root/'jobs'/j['id']/'complete.json').exists()]


def summarize(plan,root,jobs):
    datasets=list(dict.fromkeys(dataset_for(u) for u in plan['units']))
    completed=collect(root,jobs);rows=[]
    for j in completed:
        f=pd.read_csv(root/'jobs'/j['id']/'metrics.csv',float_precision='round_trip')
        f['dataset']=dataset_for(j['unit']);rows.append(f)
    report=Path(plan['execution']['report']);report.parent.mkdir(parents=True,exist_ok=True)
    lines=['# RCR规范复验：从零训练','',f'计划：`{plan["plan_name"]}`。', '',
        '所有网络从零训练；固定原数据划分、七变量、30轮上限、patience=8。'
        '唯一主评分为原RCR Scale，正常验证P95阈值；去校准仅为消融，尺度下限仅为参数敏感性。','',
        f'已完成训练单元：**{len(completed)}/{len(jobs)}**。尚未完整覆盖的数据集单元填“待补”。','',
        'Brand系列先按每个种子汇总五折，再计算三个种子的均值±样本标准差；跨折重复故障车不当作独立故障对象。'
        '本报告为固定旧划分上的开发复验，不将历史不同训练/校准协议当配对基线，不改论文最终四表。','',
        '| 方法 | '+' | '.join(d+' AP / AUROC / P / R / F1 / FPR' for d in datasets)+' |',
        '| --- | '+' | '.join('---' for _ in datasets)+' |']
    if rows:
        full=pd.concat(rows,ignore_index=True);full.to_csv(root/'all_metrics.csv',index=False)
        within=full.groupby(['dataset','name','method','seed'])[METRICS].mean()
        within.to_csv(root/'per_seed_metrics.csv')
        for name,method,label,only_seed in [('R1_full','scale','完整RCR（三种子）',None),
                                 ('R1_full','scale','完整RCR（3407，消融配对基线）',3407),
                                 ('R4_no_graph','scale','去变量图（3407）',3407),
                                 ('R5_no_temporal','scale','去Transformer（3407）',3407),
                                 ('R1_full','no_calibration','去校准（3407）',3407)]:
            cells=[]
            for dataset in datasets:
                expected_units=[u for u in plan['units'] if dataset_for(u)==dataset]
                selected=full.query('dataset == @dataset and name == @name and method == @method')
                if only_seed is not None:
                    selected=selected.query('seed == @only_seed')
                expected_seeds=plan['seeds'] if only_seed is None else [only_seed]
                if len(selected)!=len(expected_units)*len(expected_seeds):cells.append('待补');continue
                per_seed=selected.groupby('seed')[METRICS].mean()
                mean,std=per_seed.mean(),per_seed.std(ddof=1)
                cells.append(' / '.join((f'{mean[k]:.4f}±{std[k]:.4f}' if len(expected_seeds)>1
                                        else f'{mean[k]:.4f}') for k in METRICS))
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        sensitivity=[]
        for axis in plan['sensitivity']['training_axes']:
            for v in axis['values']:
                name='R1_full' if v==axis['default'] else axis['name']+'_'+value_id(v)
                f=full.query('name == @name and method == "scale" and seed == 3407').copy()
                f['axis'],f['value']=axis['parameter'],v;sensitivity.append(f)
        for v in plan['sensitivity']['scoring_axis']['values']:
            method='scale' if v==plan['scoring']['min_std'] else 'scale_floor_'+value_id(v)
            f=full.query('name == "R1_full" and method == @method').copy();f['axis'],f['value']='min_std',v;sensitivity.append(f)
        pd.concat(sensitivity,ignore_index=True).to_csv(root/'sensitivity_metrics.csv',index=False)
    else:
        for name in ['完整RCR','去变量图','去Transformer','去校准']:lines.append('| '+name+' | '+' | '.join('待补' for _ in datasets)+' |')
    lines+=['','完整逐种子/逐折分数、阈值、训练曲线、最佳/最终检查点、映射和源码快照保存在运行目录。',
             '参数敏感性报告全部预登记取值，不依据测试最大AP选择新默认值。','']
    temporary=report.with_suffix('.tmp');temporary.write_text('\n'.join(lines));temporary.replace(report)
    atomic_json(root/'summary.json',dict(plan=plan['plan_name'],completed_training=len(completed),
        expected_training=len(jobs),report=str(report),final_matrix_modified=False,updated_at=stamp()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default=DEFAULT_PLAN)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--dry-run',action='store_true');action.add_argument('--execute',action='store_true')
    parser.add_argument('--wait-predecessor',action='store_true')
    args=parser.parse_args();plan=json.loads(Path(args.plan).read_text());jobs=expand_plan(plan)
    records=preflight(plan,jobs)
    if args.dry_run:
        print(json.dumps(dict(plan=plan['plan_name'],training_jobs=len(jobs),
            stages={s:sum(j['stage']==s for j in jobs) for s in ['multiseed','ablation','sensitivity']},
            metric_cells=sum(len(score_names(j,plan)) for j in jobs),
            units={u:r['windows'] for u,r in records.items()},output=plan['execution']['output']),ensure_ascii=False,indent=2))
        return
    root=Path(plan['execution']['output']);root.mkdir(parents=True,exist_ok=True)
    sources=[__file__,'src/models/rcr.py','src/runners/full_validation_support.py',
             'src/runners/run_mscm_development.py','src/runners/run_local_residual_models.py',
             'src/runners/run_rcr_calibration_sensitivity.py','src/runners/probe_scale_relation_evidence.py']
    sources += plan['execution'].get('additional_sources', [])
    protocol=dict(plan=plan,plan_sha256=file_hash(args.plan),sources={str(p):file_hash(p) for p in sources},
                  cohorts=records,jobs=jobs)
    marker=root/'protocol_manifest.json'
    if marker.exists():assert json.loads(marker.read_text())==protocol,'Plan/source/cohort changed: use new output'
    else:
        atomic_json(marker,protocol);snap=root/'implementation';snap.mkdir(exist_ok=True)
        shutil.copy2(args.plan,snap/'plan.json')
        for p in sources:shutil.copy2(p,snap/Path(p).name)
        atomic_json(root/'expanded_jobs.json',jobs)
    with open('/tmp/enhancedmtadgat_rcr_revalidation.lock','a+') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while not predecessor_ready(plan):
                if not args.wait_predecessor:raise RuntimeError('Predecessor incomplete; use --wait-predecessor')
                atomic_json(root/'status.json',dict(status='waiting_predecessor',predecessor=plan['execution']['predecessor'],updated_at=stamp()))
                time.sleep(10)
            if not torch.cuda.is_available():raise RuntimeError('CUDA required outside sandbox')
            torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.use_deterministic_algorithms(True)
            atomic_json(root/'hardware.json',dict(gpu=torch.cuda.get_device_name(0),
                memory_bytes=torch.cuda.get_device_properties(0).total_memory,torch_version=torch.__version__,
                deterministic=True,tf32=False,started_at=stamp()))
            summarize(plan,root,jobs)
            for job in jobs:
                if (root/'jobs'/job['id']/'complete.json').exists():continue
                host=plan['execution'].get('host_disk_guard')
                if host and shutil.disk_usage(host['path']).free < host['min_free_bytes']:
                    raise RuntimeError('Host disk free space below configured safety reserve')
                atomic_json(root/'status.json',dict(status='running',active_job=job['id'],
                    completed_training=len(collect(root,jobs)),expected_training=len(jobs),updated_at=stamp()))
                run_job(job,plan,root);summarize(plan,root,jobs)
            for p,sha in protocol['sources'].items():assert file_hash(p)==sha,'Source changed during execution'
            atomic_json(root/'complete.json',dict(plan=plan['plan_name'],training_jobs=len(jobs),
                metric_cells=sum(len(score_names(j,plan)) for j in jobs),final_matrix_modified=False))
            atomic_json(root/'status.json',dict(status='complete',completed_training=len(jobs),updated_at=stamp()))
        except Exception:
            atomic_json(root/'status.json',dict(status='failed',traceback=traceback.format_exc(),updated_at=stamp()))
            raise


if __name__=='__main__':main()
