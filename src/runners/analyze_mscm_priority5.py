"""Audit and report the five-priority Memory continuation and 3-seed study."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/mscm-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.evaluate_rcr import metrics
from src.models.mscm_reconstruction import MSCMReconstructor
from src.runners.run_mscm_priority5 import DEFAULT_ROOT,ARMS
from src.runners.run_mcr_priority5 import SEEDS,DIRECT,seed_root
from src.runners.run_mcr_full_validation import HEADS
from src.runners.run_rcr import atomic_json
from src.runners.run_rcr_memory_probe import digest
from src.runners.run_local_residual_models import source_for
from src.runners.probe_scale_relation_evidence import read,paired_bootstrap
from src.runners.report_metric_format import bold_best_metrics


PRIOR=Path('runs/internal/rcr_local_two_models_20261025_v1')
ONE_PAIRS=[('A4_short_raw','A4_raw'),('A4_short_cal_abs','A4_cal_abs'),
    ('A1_raw','A4_raw'),('A6_raw','A4_raw'),('A1_cal_abs','A4_cal_abs'),('A6_cal_abs','A4_cal_abs'),
    ('A4_pure_raw','A4_raw'),('A4_direct_raw','A4_raw'),
    ('A4_pure_cal_abs','A4_cal_abs'),('A4_direct_cal_abs','A4_cal_abs'),
    ('A4_raw','A4_cal_abs'),('A4_raw','A4_rcr_legacy'),
    ('A4_raw','A4_final_raw'),('A4_cal_abs','A4_final_cal_abs'),
    ('RCR_reference','A4_raw'),('RCR_reference','A4_cal_abs')]
TWO_PAIRS=[('B4_memory_mlp','B3_memory_gat'),('B0','B3_memory_gat'),('Signed_L1','B3_memory_gat'),
    ('B0','B5_ECDF'),('B5_diagonal_ECDF','B5_ECDF')]
CONTEXT_PAIRS=[('B2_gat','B3_memory_gat'),('B3_prior_gat','B3_memory_gat'),('B3_q_only_gat','B3_memory_gat')]


def audit_metrics(output):
    test=read(output/'test_scores.csv');saved=read(output/'metrics.csv');error=0.
    for row in saved.itertuples():
        name=getattr(row,'score',getattr(row,'method',None))
        actual=metrics(test.sample_label,test[name],row.threshold)
        for key,value in actual.items():error=max(error,abs(value-getattr(row,key)))
    assert error<1e-12
    return error


def check_sources(protocol):
    checks={path:digest(path)==signature for path,signature in protocol['implementations'].items()}
    assert all(checks.values()),checks
    return checks


def reconstruction_frames(root,core,unit):
    keys=['vin','sample_label'] if unit.startswith('Brand3') else ['sample_id','vin','sample_label']
    frames={}
    for p in ['validation','test']:
        base=read(root/'model_one'/unit/'A4'/f'{p}_scores.csv')[keys].copy()
        for arm in list(ARMS)+['A4_final']:
            values=read(root/'model_one'/unit/arm/f'{p}_scores.csv')
            pd.testing.assert_frame_equal(base[keys],values[keys])
            for score in ['raw','cal_abs','rcr_legacy']:base[arm+'_'+score]=values[score].to_numpy()
        old=read(core/unit/'A4'/f'{p}_scores.csv')
        pd.testing.assert_frame_equal(base[keys],old[keys])
        for score in ['raw','cal_abs','rcr_legacy']:base['A4_short_'+score]=old[score].to_numpy()
        reference=read(PRIOR/unit/f'{p}_scores.csv')
        pd.testing.assert_frame_equal(base[keys],reference[keys])
        base['RCR_reference']=reference.A.to_numpy()
        frames[p]=base
    thresholds={name:{str(q):float(frames['validation'][name].quantile(q)) for q in [.95,.99]}
        for name in frames['validation'] if name not in keys}
    return frames,thresholds


def local_metrics(root,unit,output,frames,methods,thresholds):
    if unit.startswith('Brand3'):return
    meta=read(PRIOR/unit/'test_snippet_scores.csv')[['sample_id','operating_regime','fault_type']]
    frame=frames['test'].drop(columns=[c for c in ['operating_regime','fault_type'] if c in frames['test']]).merge(
        meta,on='sample_id',validate='one_to_one');rows=[]
    for state,group in frame.groupby('operating_regime'):
        normal=group[group.sample_label==0]
        for fault,affected in group[group.sample_label==1].groupby('fault_type'):
            if min(len(normal),len(affected))<10:continue
            both=pd.concat([normal,affected])
            for method in methods:
                rows.append(dict(unit=unit,state=state,fault_type=fault,method=method,
                    normal=len(normal),fault=len(affected),normal_vins=normal.vin.nunique(),fault_vins=affected.vin.nunique(),
                    **metrics(both.sample_label,both[method],thresholds[method]['0.95'])))
    pd.DataFrame(rows).to_csv(output,index=False)


def memory_probe(root,core,unit,arm):
    output=root/'model_one'/unit/arm
    marker=output/'routing_probe.json'
    if marker.exists():return json.loads(marker.read_text())
    mode={'A4':'bounded','A4_pure':'pure','A4_direct':'direct'}[arm]
    model=MSCMReconstructor('A4',fusion=mode).eval()
    model.load_state_dict(torch.load(output/'training/best.pt',map_location='cpu',weights_only=True))
    source,_=source_for(unit);mapping=json.loads((source/'mapping.json').read_text())
    metadata=read(source/'metadata.csv').set_index('sample_id')
    records=[];torch.set_num_threads(2);generator=torch.Generator().manual_seed(3407)
    with torch.inference_mode():
        for partition in ['train','test']:
            x=np.load(core/'inputs'/unit/f'{partition}_x.npy',mmap_mode='r')
            refs=np.load(source/'residuals'/partition/'references.npy',mmap_mode='r')
            labels=metadata.loc[mapping['sample_ids'][partition],'sample_label'].to_numpy()[refs[:,0].astype(int)]
            for label in np.unique(labels):
                candidates=np.flatnonzero(labels==label)
                indices=np.random.default_rng(3407).choice(candidates,min(1024,len(candidates)),replace=False)
                np.save(output/f'{partition}_label{label}_routing_indices.npy',indices)
                for s in range(3):
                    values=[]
                    for lo in range(0,len(indices),128):
                        h=model.patches(torch.from_numpy(np.array(x[indices[lo:lo+128]])))[s]
                        memory=model.memories[s];q=(h-memory.center)/memory.scale
                        p=memory.prototypes
                        weights=(q@p.T/(memory.temperature*np.sqrt(q.shape[-1]))).softmax(-1)
                        assigned=weights.argmax(-1)
                        perturbed=q+.01*torch.randn(q.shape,generator=generator)
                        changed=(perturbed@p.T/(memory.temperature*np.sqrt(q.shape[-1]))).argmax(-1)
                        distance=(q[:,:,None]-p).square().mean(-1).sqrt().amin(-1)
                        gap=weights.topk(2,-1).values.diff(dim=-1).abs().squeeze(-1)
                        switch=(assigned[:,1:]!=assigned[:,:-1]).float().mean(1) if assigned.shape[1]>1 else torch.zeros(len(q))
                        values.append(torch.stack((distance.mean(1),(assigned!=changed).float().mean(1),switch,gap.mean(1)),-1).numpy())
                    data=np.concatenate(values)
                    records.append(dict(unit=unit,arm=arm,partition=partition,label=int(label),scale=s,windows=len(indices),
                        mean_nearest_patch_distance=float(data[:,0].mean()),noise_flip_rate=float(data[:,1].mean()),
                        within_window_switch_rate=float(data[:,2].mean()),mean_top2_attention_gap=float(data[:,3].mean())))
    atomic_json(marker,records)
    return records


def write_table(lines,methods,table,datasets):
    lines+=['| 方法 | '+' | '.join(d+' AP / AUROC / F1' for d in datasets)+' |',
        '| --- | '+' | '.join('---' for _ in datasets)+' |']
    for method,label in methods:
        cells=[]
        for d in datasets:
            r=table.loc[d,method];cells.append(f'{r.ap:.4f} / {r.auroc:.4f} / {r.f1:.4f}')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')


def analyze_one(root,repeats):
    assert (root/'model_one_complete.json').exists()
    protocol=json.loads((root/'model_one_protocol.json').read_text());core=Path(protocol['core'])
    units=protocol['units'];audits=[];convergence=[];occupancy=[];routing=[];quantiles=[];amplitudes=[];all_rows=[]
    for unit,parents in protocol.get('source_parents',{}).items():
        for arm,signature in parents.items():assert digest(core/unit/arm/'training/last.pt')==signature
    entries={};datasets=[d for d in ['LFP','NCM','Brand3'] if any(u==d or u.startswith(d+'_') for u in units)]
    for unit in units:
        entries[unit]=reconstruction_frames(root,core,unit)
        for arm in list(ARMS)+['A4_final']:
            output=root/'model_one'/unit/arm
            audits.append(dict(unit=unit,arm=arm,max_metric_error=audit_metrics(output)))
            saved=read(output/'metrics.csv').assign(arm=arm,dataset='Brand3' if unit.startswith('Brand3') else unit)
            all_rows.append(saved)
            if arm!='A4_final':
                training=json.loads((output/'training/complete.json').read_text())
                convergence.append(dict(unit=unit,arm=arm,**training))
                for file in sorted((output/'training').glob('memory_epoch_*.json')):
                    row=json.loads(file.read_text())
                    for s,soft in enumerate(row['soft_occupancy']):
                        p=np.asarray(soft)
                        occupancy.append(dict(unit=unit,arm=arm,epoch=row['epoch'],scale=s,
                            effective_soft_prototypes=float(np.exp(-(p*np.log(np.maximum(p,1e-30))).sum())),
                            hard_active=sum(np.asarray(row['hard_occupancy'][s])>=.01)))
            for label,group in read(output/'test_scores.csv').groupby('sample_label'):
                for score in ['raw','cal_abs','rcr_legacy']:
                    quantiles.append(dict(unit=unit,arm=arm,label=int(label),score=score,count=len(group),
                        **{f'p{q}':float(group[score].quantile(q/100)) for q in [5,25,50,75,95,99]}))
            source,_=source_for(unit);mapping=json.loads((source/'mapping.json').read_text())
            metadata=read(source/'metadata.csv').set_index('sample_id')
            refs=np.load(source/'residuals/test/references.npy',mmap_mode='r')
            labels=metadata.loc[mapping['sample_ids']['test'],'sample_label'].to_numpy()[refs[:,0].astype(int)]
            moments=np.load(output/'test_moments.npy',mmap_mode='r')
            for label in np.unique(labels):
                ix=np.flatnonzero(labels==label)
                for j in range(7):
                    mae=moments[ix,2,j];rms=np.sqrt(moments[ix,1,j]+moments[ix,0,j]**2)
                    amplitudes.append(dict(unit=unit,arm=arm,label=int(label),variable=j,windows=len(ix),
                        mae_median=float(np.median(mae)),mae_p95=float(np.quantile(mae,.95)),
                        rms_median=float(np.median(rms)),rms_p95=float(np.quantile(rms,.95))))
            for p in ['train','validation','test']:
                scores=np.load(output/f'{p}_window_scores.npy',mmap_mode='r')
                assert np.isfinite(scores).all()
            if arm in ['A4','A4_pure','A4_direct']:routing+=memory_probe(root,core,unit,arm)
        a=json.loads((root/'model_one'/unit/'A4/training/initial.json').read_text())
        b=json.loads((root/'model_one'/unit/'A6/training/initial.json').read_text())
        assert a['parameters']==b['parameters'] and a['trainable_parameters']==b['trainable_parameters']
        local_metrics(root,unit,root/'model_one'/unit/'local_fault_regime_metrics.csv',entries[unit][0],
            ['A4_raw','A4_cal_abs','A4_pure_cal_abs','A4_direct_cal_abs'],entries[unit][1])
    cis=[]
    for dataset in datasets:
        group=[entries[u] for u in units if u==dataset or u.startswith(dataset+'_')]
        for q in [.95,.99]:cis+=paired_bootstrap(group,ONE_PAIRS,repeats,dataset,q)
    pd.DataFrame(cis).to_csv(root/'model_one_paired_intervals.csv',index=False)
    pd.DataFrame(convergence).to_csv(root/'model_one_convergence.csv',index=False)
    pd.DataFrame(occupancy).to_csv(root/'model_one_epoch_occupancy.csv',index=False)
    pd.DataFrame(routing).to_csv(root/'model_one_routing.csv',index=False)
    pd.DataFrame(quantiles).to_csv(root/'model_one_score_quantiles.csv',index=False)
    pd.DataFrame(amplitudes).to_csv(root/'model_one_variable_amplitudes.csv',index=False)
    all_metrics=pd.concat(all_rows,ignore_index=True);all_metrics.to_csv(root/'model_one_metrics.csv',index=False)
    table=all_metrics.query('q==.95').groupby(['dataset','arm','score'])[['ap','auroc','f1','recall','fpr']].mean()
    primary=table.xs('cal_abs',level='score')
    lines=['# 五项核心验证：模型一长训练与融合机制','',
        '保持原完整数据划分、阶段A教师与固定参数；A4由原第5轮优化器断点续训。纯Memory、直接0.5融合、有界补偿在同一教师上比较，A1/A6为匹配训练预算对照。正常验证选最佳；最后检查点只作诊断。','']
    write_table(lines,[(a,a+' / 固定稳健尺度校准') for a in ARMS]+[('A4_final','A4最后检查点 / 仅诊断')],primary,datasets)
    lines+=['','## 未校准重构误差','']
    write_table(lines,[(a,a+' / raw') for a in ARMS]+[('A4_final','A4最后检查点 / raw')],table.xs('raw',level='score'),datasets)
    lines+=['','## 原RCR尺度规则重新拟合','']
    write_table(lines,[(a,a+' / rcr_legacy') for a in ARMS]+[('A4_final','A4最后检查点 / rcr_legacy')],table.xs('rcr_legacy',level='score'),datasets)
    lines+=['','## 主要成对比较','',
        '| 数据集 | 对照 → 方法 | AUROC差值 [95% VIN配对区间] |','| --- | --- | --- |']
    for r in pd.DataFrame(cis).query('q==.95 and metric=="auroc"').itertuples():
        lines.append(f'| {r.dataset} | {r.base} → {r.contender} | {r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] |')
    lines+=['','## 训练与判读边界','',
        '最佳与最终参数分别保存在各组training/best.pt和training/final.pt。原第1–5轮曲线保留，续训记录每轮完整正常训练的原型占用；不能把最后测试成绩更高作为改选检查点的理由。',
        '初始50轮、按预设正常验证规则延长至100轮；触及硬上限不等于已经收敛，逐组训练轮数、停止原因、最优点及损失见model_one_convergence.csv。',
        'cal_abs与rcr_legacy分别在同一固定模型上重新拟合；完整raw/cal_abs/cal_signed/rcr_legacy结果见model_one_metrics.csv。模型one配对区间同时比较旧短预算A4、三种融合、A1/A6与校准前后，均按VIN成组计算。',
        f'{repeats}次分层VIN配对bootstrap；重复故障车跨折共享权重。区间固定模型与正常验证阈值，不包含重训练/阈值拟合不确定性。',
        'Brand3保持本地重训参考来源；原Kaggle Scale不是同一个检查点。本轮没有重新划分独立校准集，没有跑Patch/λ搜索或编码器微调。','']
    report=Path('report/analysis/rcr/MSCM_PRIORITY5_MODEL_ONE_20261025.md');report.write_text(bold_best_metrics('\n'.join(lines)))
    figures=Path('report/figures/mscm_mcr_priority5_20261025');figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(len(datasets),2,figsize=(12,3*len(datasets)),squeeze=False)
    for i,d in enumerate(datasets):
        for unit in [u for u in units if u==d or u.startswith(d+'_')]:
            history=read(root/'model_one'/unit/'A4/training/history.csv')
            axes[i,0].plot(history.epoch,history.normal_validation_huber,label=unit)
        axes[i,0].set(title=d+' normal validation',yscale='log',xlabel='epoch');axes[i,0].legend(fontsize=7)
        for arm in ['A4','A4_pure','A4_direct']:
            group=pd.DataFrame(occupancy)
            if len(group):
                group=group[(group.arm==arm)&group.unit.isin([u for u in units if u==d or u.startswith(d+'_')])]
                curve=group.groupby('epoch').effective_soft_prototypes.mean()
                axes[i,1].plot(curve.index,curve.values,label=arm)
        axes[i,1].set(title=d+' effective prototypes',xlabel='epoch',ylim=(0,8.2));axes[i,1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(figures/f'model_one_training.{ext}',dpi=170)
    plt.close(fig)
    atomic_json(root/'model_one_artifact_audit.json',dict(units=audits,code_unchanged=check_sources(protocol),
        original_training_cohort=True,bootstrap=repeats,all_metrics_recomputed=True))
    atomic_json(root/'model_one_analysis_complete.json',dict(report=str(report),units=units,bootstrap=repeats))
    print('PRIORITY5 MODEL ONE ANALYSIS COMPLETE',flush=True)


def analyze_two(root,repeats):
    assert (root/'model_two_complete.json').exists()
    protocol=json.loads((root/'model_two_protocol.json').read_text());units=protocol['units']
    datasets=[d for d in ['LFP','NCM','Brand3'] if any(u==d or u.startswith(d+'_') for u in units)]
    audits=[];all_rows=[];overlaps=[];intervals=[];context_intervals=[];entries={}
    for seed in SEEDS:
        for unit in units:
            directory=seed_root(root,seed)/'model_two'/unit
            marker=json.loads((directory/'complete.json').read_text())
            assert marker['model_one_scores_exact'] and marker['full_target_training']
            context=json.loads((directory/'conditions_complete.json').read_text())
            assert context['raw_target_context_max_error']<1e-6
            for head in HEADS if seed==3407 else DIRECT:
                head_dir=directory/head
                mask=json.loads((head_dir/'mask_audit.json').read_text())
                assert mask['trained_residual_target_perturbation_error']<1e-6
                training=json.loads((head_dir/'training/complete.json').read_text())
                for part in ['train','validation','test']:
                    e=np.load(head_dir/f'{part}_variable_mae.npy',mmap_mode='r')
                    scale=np.load(head_dir/'normal_relation_error_scale.npy')
                    score=np.load(head_dir/f'{part}_window_scores.npy',mmap_mode='r')
                    assert np.isfinite(e).all()
                    np.testing.assert_array_equal((e/scale).mean(1),score)
                audits.append(dict(unit=unit,seed=seed,head=head,**mask,**training))
            error=audit_metrics(directory)
            assert error<1e-12
            frames={p:read(directory/f'{p}_scores.csv') for p in ['validation','test']}
            for part in frames:
                parent=read(root/'model_one'/unit/'A4'/f'{part}_scores.csv')
                np.testing.assert_array_equal(frames[part].B0.to_numpy(),parent.cal_abs.to_numpy())
            thresholds=json.loads((directory/'thresholds.json').read_text())
            entries[seed,unit]=(frames,thresholds)
            all_rows.append(read(directory/'metrics.csv').assign(seed=seed,dataset='Brand3' if unit.startswith('Brand3') else unit))
            overlaps.append(read(directory/'error_overlap.csv').assign(seed=seed))
            local_metrics(root,unit,directory/'local_fault_regime_metrics.csv',frames,
                ['B0','B3_memory_gat','B4_memory_mlp','B5_ECDF'],thresholds)
    for dataset in datasets:
        group=[u for u in units if u==dataset or u.startswith(dataset+'_')]
        repeated=[entries[s,u] for s in SEEDS for u in group]
        for q in [.95,.99]:
            joint=paired_bootstrap(repeated,TWO_PAIRS,repeats,dataset,q)
            intervals += [dict(seed_scope='macro mean of3 paired head seeds',**r) for r in joint]
            context_intervals += paired_bootstrap([entries[3407,u] for u in group],CONTEXT_PAIRS,repeats,dataset,q)
    combined=pd.concat(all_rows,ignore_index=True)
    combined.to_csv(root/'model_two_metrics.csv',index=False)
    per_seed=combined.groupby(['dataset','seed','method','q'])[
        ['ap','auroc','precision','f1','recall','fpr','recall_at_test_fpr05','recall_at_test_fpr01']].mean().reset_index()
    per_seed.to_csv(root/'model_two_seed_means.csv',index=False)
    summary=per_seed.groupby(['dataset','method','q'])[
        ['ap','auroc','precision','f1','recall','fpr','recall_at_test_fpr05','recall_at_test_fpr01']].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    summary.to_csv(root/'model_two_seed_summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(root/'model_two_paired_intervals.csv',index=False)
    pd.DataFrame(context_intervals).to_csv(root/'memory_context_paired_intervals.csv',index=False)
    pd.DataFrame(audits).to_csv(root/'trained_relation_audits.csv',index=False)
    pd.concat(overlaps,ignore_index=True).to_csv(root/'relation_error_overlap.csv',index=False)
    for unit in units:
        for name,signature in protocol['parent_signatures'][unit].items():
            assert digest(root/'model_one'/unit/'A4'/name)==signature
    atomic_json(root/'model_two_artifact_audit.json',dict(code_unchanged=check_sources(protocol),
        model_one_unchanged=True,all_metrics_recomputed=True,raw_target_masking=True,
        residual_target_masking=True,full_relation_normalization_replay=True,
        head_seeds=SEEDS,bootstrap=repeats))
    table=per_seed.query('q==.95').groupby(['dataset','method'])[['ap','auroc','f1','recall','fpr']].mean()
    lines=['# 五项核心验证：实际Memory条件与B3/B4三随机种子','',
        'B3/B4使用当前样本实际Memory检索权重：逐目标将原始变量替换为正常训练全局均值，再重新编码检索；预测电流/SOC时移除自身工况统计。A4重构器、Memory和校准器固定，全部7个变量参与每个窗口每轮的LOO训练。','',
        'B3/B4及其融合结果按seed3407/3408/3409三seed平均；其他关系头为seed3407控制。先按每seed求Brand3五折均值，再跨seed平均。多seed只覆盖关系头，不代表整套重构模型的多seed。','']
    methods=[('B0','B0：固定长训练A4'),('B1_original','B1：原关系头，1 seed'),('B2_gat','B2：普通GAT，1 seed'),
        ('B3_memory_gat','B3：实际Memory GAT，3 seeds'),('B4_memory_mlp','B4：同条件MLP，3 seeds'),
        ('B3_prior_gat','先验条件GAT，1 seed'),('B3_q_only_gat','工况GAT，1 seed'),
        ('B5_ECDF','B5：ECDF等权融合，3 seeds'),('B5_MAD','MAD等权融合，3 seeds')]
    write_table(lines,methods,table,datasets)
    lines+=['','## B3与B4逐seed结果','',
        '| 数据集 | seed | B3 AP / AUROC / F1 | B4 AP / AUROC / F1 | B3−B4 AUROC |','| --- | --- | --- | --- | --- |']
    for d in datasets:
        for seed in SEEDS:
            a=per_seed.query('dataset==@d and seed==@seed and q==.95 and method=="B3_memory_gat"').iloc[0]
            b=per_seed.query('dataset==@d and seed==@seed and q==.95 and method=="B4_memory_mlp"').iloc[0]
            lines.append(f'| {d} | {seed} | {a.ap:.4f} / {a.auroc:.4f} / {a.f1:.4f} | {b.ap:.4f} / {b.auroc:.4f} / {b.f1:.4f} | {a.auroc-b.auroc:+.4f} |')
    lines+=['','## 三seed平均的成对区间','',
        '| 数据集 | 对照 → 方法 | AUROC差值 [95% VIN配对区间] |','| --- | --- | --- |']
    for r in pd.DataFrame(intervals).query('q==.95 and metric=="auroc"').itertuples():
        lines.append(f'| {r.dataset} | {r.base} → {r.contender} | {r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] |')
    lines+=['','## 同seed实际检索与先验/工况条件','',
        '| 数据集 | 对照 → 实际Memory GAT | AUROC差值 [95% VIN配对区间] |','| --- | --- | --- |']
    for r in pd.DataFrame(context_intervals).query('q==.95 and metric=="auroc"').itertuples():
        lines.append(f'| {r.dataset} | {r.base} → {r.contender} | {r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] |')
    lines+=['',f'{repeats}次标签分层VIN配对bootstrap，所有方法、seed、跨折重复VIN共用抽样权重。区间条件于已训练的3个头和固定正常验证阈值；不把三seed当独立故障队列，不包含完整Backbone重训练不确定性。',
        'seed均值/标准差及逐折原始结果见model_two_seed_summary.csv、model_two_seed_means.csv、model_two_metrics.csv。错漏检与独立检出见relation_error_overlap.csv，各seed/unit的prediction_sets.csv保留逐对象分数和阈值。',
        '实际检索优于先验或工况条件、GAT优于同条件MLP是不同问题，须分别比较；新增检出还必须同时看新增正常误报。','']
    report=Path('report/analysis/rcr/MCR_PRIORITY5_MODEL_TWO_20261025.md')
    report.write_text(bold_best_metrics('\n'.join(lines)))
    figures=Path('report/figures/mscm_mcr_priority5_20261025');figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,len(datasets),figsize=(5*len(datasets),4),squeeze=False)
    for ax,d in zip(axes[0],datasets):
        names=['B0','B2_gat','B3_memory_gat','B4_memory_mlp','B3_prior_gat','B3_q_only_gat','B5_ECDF']
        ax.bar(np.arange(len(names)),[table.loc[d,m].auroc for m in names])
        ax.set(title=d,ylabel='AUROC',ylim=(0,1),xticks=np.arange(len(names)),
            xticklabels=['M1','GAT','Actual Mem','MLP','Prior Mem','q-GAT','Joint'])
        ax.tick_params(axis='x',rotation=35)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(figures/f'model_two_comparison.{ext}',dpi=170)
    plt.close(fig)
    atomic_json(root/'model_two_analysis_complete.json',dict(report=str(report),units=units,seeds=SEEDS))
    print('PRIORITY5 MODEL TWO ANALYSIS COMPLETE',flush=True)


def final_summary(root):
    one=read(root/'model_one_metrics.csv').query('q==.95')
    two=read(root/'model_two_seed_means.csv').query('q==.95')
    one['dataset']=np.where(one.unit.str.startswith('Brand3'),'Brand3',one.unit)
    first=one.groupby(['dataset','arm','score'])[['ap','auroc','f1']].mean()
    second=two.groupby(['dataset','method'])[['ap','auroc','f1']].mean()
    datasets=[d for d in ['LFP','NCM','Brand3'] if d in first.index.get_level_values(0)]
    rows=[]
    for d in datasets:
        for method in ['A4','A4_pure','A4_direct','A1','A6']:
            r=first.loc[d,method,'cal_abs'];rows.append(dict(dataset=d,method=method,**r.to_dict()))
        r=first.loc[d,'A4','raw'];rows.append(dict(dataset=d,method='A5',**r.to_dict()))
        for method in ['B0','B1_original','B2_gat','B3_memory_gat','B4_memory_mlp','B5_ECDF']:
            rows.append(dict(dataset=d,method=method,**second.loc[d,method].to_dict()))
    table=pd.DataFrame(rows).set_index(['dataset','method'])
    lines=['# 五项Memory核心机制验证结果','',
        '本轮保持原划分、原共同教师、Patch8和λ=.01，延长A4训练并比较纯Memory、直接融合、有界补偿；独立比较raw与重新拟合校准。模型二使用目标遮挡后的实际Memory检索条件，B3/B4比较3个固定随机种子。未开展Patch/λ搜索或编码器微调。','',
        '## 模型一','']
    write_table(lines,[(m,m+'＋校准') for m in ['A1','A4','A4_pure','A4_direct','A6']]+[('A5','A5：A4未校准')],table,datasets)
    lines+=['','A3按附件与A1合并。A0保留原RCR参考；A2仍为旧短预算结果，本轮五项优先验证未重跑，不能将其称为长预算对照。A4_final仅作诊断，未据测试指标选为关系父模型。','',
        '## 模型二','']
    write_table(lines,[(m,label) for m,label in [('B0','B0：固定A4'),('B1_original','B1：原关系头'),('B2_gat','B2：普通GAT'),
        ('B3_memory_gat','B3：实际Memory GAT，3 seeds'),('B4_memory_mlp','B4：同条件MLP，3 seeds'),('B5_ECDF','B5：ECDF等权融合，3 seeds')]],table,datasets)
    lines+=['','完整训练曲线、原型占用与稳定性、最佳/最终检查点、raw与校准对照、VIN成组区间、逐seed标准差与错漏检见：',
        '- [模型一长训练与融合](MSCM_PRIORITY5_MODEL_ONE_20261025.md)',
        '- [模型二实际Memory与三seed](MCR_PRIORITY5_MODEL_TWO_20261025.md)',
        '- [五项执行协议](../../system_design/MSCM_MCR_PRIORITY5_20261025.md)','',
        '模型一仍以原正常训练集拟合校准；没有宣称独立校准。Brand3沿用本地RCR来源，与历史Kaggle Scale检查点区分。三seed只衡量关系头随机性。触及100轮硬上限的情况在训练记录中明确列出，不据此声称已收敛。',
        '所有结果已完成样本映射、原始分数指标复算、模型冻结、原始目标及残差目标遮挡、关系归一化重放与源码签名审计。论文最终四张实验矩阵未改版或提前填值。','']
    report=Path('report/analysis/rcr/MSCM_MCR_PRIORITY5_RESULTS_20261025.md')
    report.write_text(bold_best_metrics('\n'.join(lines)))
    link='后续五项核心机制验证已完成，最新结果见[长训练、三种融合与实际Memory三seed报告](MSCM_MCR_PRIORITY5_RESULTS_20261025.md)。'
    for name in ['MSCM_MODEL_ONE_CORE_20261025.md','MCR_GAT_MODEL_TWO_20261025.md',
                 'MSCM_MCR_IMPLEMENTATION_RESULTS_20261025.md','MSCM_MCR_ABLATION_INDEX_20261025.md']:
        path=report.parent/name;text=path.read_text()
        if link not in text:
            heading,body=text.split('\n\n',1);path.write_text(heading+'\n\n'+link+'\n\n'+body)
    atomic_json(root/'analysis_complete.json',dict(report=str(report),all_requested_five_priorities=True))
    print('PRIORITY5 ALL ANALYSIS COMPLETE',report,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=DEFAULT_ROOT)
    parser.add_argument('--bootstrap',type=int,default=500)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    for stage,marker in [('one','model_one_complete.json'),('two','model_two_complete.json')]:
        done=root/f'model_{stage}_analysis_complete.json'
        if done.exists():continue
        while not (root/marker).exists():
            if not args.wait:raise RuntimeError(f'Finish {stage} training first')
            time.sleep(10)
        if stage=='one':analyze_one(root,args.bootstrap)
        else:analyze_two(root,args.bootstrap)
    final_summary(root)


if __name__=='__main__':main()
