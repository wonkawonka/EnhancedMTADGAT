"""Audit and report completed MSCM model-one cells without editing final tables."""
import argparse
import json
import os
from pathlib import Path
from src.runners.report_metric_format import bold_best_metrics

os.environ.setdefault('MPLCONFIGDIR','/tmp/mscm-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.runners.run_mscm_development import UNITS,VARIANTS,SCORES,summarize
from src.runners.probe_scale_relation_evidence import paired_bootstrap,read
from src.runners.run_rcr import atomic_json
from src.runners.run_rcr import state_hash
from src.runners.run_rcr_memory_probe import digest
from scripts.evaluate_rcr import metrics
from src.models.mscm_reconstruction import MSCMReconstructor
from src.runners.run_local_residual_models import source_for


PRIOR = Path('runs/internal/rcr_local_two_models_20261025_v1')
PAIRS = [('A0_reference','A0_cal_abs'),('A0_cal_abs','A1_cal_abs'),
    ('A0_reference','A1_raw'),('A0_raw','A1_raw'),
    ('A0_reference','A4_raw'),('A0_reference','A4_cal_abs'),('A0_reference','A4_rcr_legacy'),
    ('A0_cal_abs','A2_cal_abs'),('A1_cal_abs','A4_cal_abs'),
    ('A2_cal_abs','A4_cal_abs'),('A6_cal_abs','A4_cal_abs'),
    ('A1_raw','A4_raw'),('A2_raw','A4_raw'),('A6_raw','A4_raw'),
    ('A0_cal_abs','A4_cal_abs'),('A0_raw','A4_raw'),('A4_raw','A4_cal_abs')]


def combined(root,unit):
    result = {}
    thresholds = {}
    for partition in ['validation','test']:
        frames = {v:read(root/unit/v/f'{partition}_scores.csv') for v in VARIANTS}
        keys = ['vin','sample_label'] if unit.startswith('Brand3') else ['sample_id','vin','sample_label']
        frame = frames['A0'][keys].copy()
        for variant,values in frames.items():
            pd.testing.assert_frame_equal(frame[keys],values[keys])
            for score in SCORES:
                frame[variant+'_'+score] = values[score].to_numpy()
        old = read(PRIOR/unit/f'{partition}_scores.csv')
        pd.testing.assert_frame_equal(frame[keys],old[keys])
        frame['A0_reference'] = old.A.to_numpy()
        result[partition] = frame
    for name in [c for c in result['validation'] if c not in keys]:
        thresholds[name] = {str(q):float(result['validation'][name].quantile(q)) for q in [.95,.99]}
    return result,thresholds


def audit_unit(root,unit):
    rows,distribution,memory_rows = [],[],[]
    maximum = 0.
    for variant in VARIANTS:
        output = root/unit/variant
        scores = read(output/'test_scores.csv')
        summary = read(output/'metrics.csv')
        for row in summary.itertuples():
            recalculated = metrics(scores.sample_label,scores[row.score],row.threshold)
            for key,value in recalculated.items():
                maximum = max(maximum,abs(value-getattr(row,key)))
                assert abs(value-getattr(row,key)) < 1e-12
        for partition in ['train','validation','test']:
            matrix = np.load(output/f'{partition}_window_scores.npy',mmap_mode='r')
            assert np.isfinite(matrix).all()
        sample = read(output/'test_snippet_scores.csv')
        for label,subset in sample.groupby('sample_label'):
            for name in SCORES:
                distribution.append(dict(unit=unit,variant=variant,label=int(label),score=name,
                    snippets=len(subset),vins=subset.vin.nunique(),
                    **{f'p{q:02d}':float(np.quantile(subset[name],q/100)) for q in [5,25,50,75,95,99]}))
        if variant in ['A2','A4']:
            for partition in ['train','validation','test']:
                data = json.loads((output/f'{partition}_memory_diagnostics.json').read_text())
                for row in data:
                    memory_rows.append(dict(unit=unit,variant=variant,partition=partition,
                        scale=row['scale'],kernel=row['kernel'],effective_soft_prototypes=row['effective_soft_prototypes'],
                        hard_prototypes_above_one_percent=row['hard_prototypes_above_one_percent'],
                        pairwise_cosine_mean=row['pairwise_cosine_mean'],pairwise_distance_mean=row['pairwise_distance_mean'],
                        query_variance_mean=row['query_variance_mean'],retrieved_variance_mean=row['retrieved_variance_mean'],
                        distance_p50=row['nearest_distance_quantiles'][1],distance_p95=row['nearest_distance_quantiles'][2],
                        **{f'occupancy_{j}':row['hard_occupancy'][j] for j in range(8)}))
    a4 = json.loads((root/unit/'A4/training/initial.json').read_text())
    a6 = json.loads((root/unit/'A6/training/initial.json').read_text())
    assert a4['parameters'] == a6['parameters']
    assert a4['trainable_parameters'] == a6['trainable_parameters']
    return dict(unit=unit,max_metric_error=maximum,all_scores_finite=True,A4_A6_parameters=a4['parameters'],
        A4_A6_trainable=a4['trainable_parameters']),distribution,memory_rows


def local_analysis(root,unit):
    meta = read(PRIOR/unit/'test_snippet_scores.csv')
    metadata_columns = ['sample_id','operating_regime','fault_type']
    rows = []
    for variant in VARIANTS:
        frame = read(root/unit/variant/'test_snippet_scores.csv')
        frame = frame.drop(columns=[c for c in ['operating_regime','fault_type'] if c in frame]).merge(
            meta[metadata_columns],on='sample_id',validate='one_to_one')
        thresholds = json.loads((root/unit/variant/'thresholds.json').read_text())
        for grouping in ['fault_type','operating_regime']:
            for group,sub in frame.groupby(grouping):
                if grouping == 'fault_type':
                    if group == 'normal': continue
                    sub = pd.concat([frame[frame.sample_label==0],sub])
                if min((sub.sample_label==0).sum(),(sub.sample_label==1).sum())<10: continue
                for score in ['raw','cal_abs']:
                    # CH snippet ranking; Brand3 local cohorts are descriptive only.
                    from src.runners.probe_scale_relation_evidence import rank_frame
                    ranked = rank_frame(sub,[score],unit.startswith('Brand3'))
                    if ranked.sample_label.nunique()<2: continue
                    rows.append(dict(unit=unit,variant=variant,score=score,grouping=grouping,group=str(group),
                        normal_snippets=int((sub.sample_label==0).sum()),fault_snippets=int((sub.sample_label==1).sum()),
                        normal_vins=int(sub.loc[sub.sample_label==0,'vin'].nunique()),fault_vins=int(sub.loc[sub.sample_label==1,'vin'].nunique()),
                        **metrics(ranked.sample_label,ranked[score],thresholds[score]['0.95'])))
    pd.DataFrame(rows).to_csv(root/unit/'local_descriptive_metrics.csv',index=False)


def patch_distribution_probe(root,unit):
    """Exact patch distances on fixed normal/test window samples, not main scores."""
    records = []
    settings = json.loads((root/'protocol_manifest.json').read_text())
    source,_ = source_for(unit)
    mapping = json.loads((source/'mapping.json').read_text())
    metadata = pd.read_csv(source/'metadata.csv',dtype={'vin':str,'car':str}).set_index('sample_id')
    torch.set_num_threads(2)
    for variant in ['A2','A4']:
        p = root/unit/variant
        model = MSCMReconstructor(variant,patch=settings['patch'],gamma=settings['gamma'],fusion=settings['fusion']).eval()
        model.load_state_dict(torch.load(p/'training/best.pt',map_location='cpu',weights_only=True))
        for partition in ['train','test']:
            x = np.load(root/'inputs'/unit/f'{partition}_x.npy',mmap_mode='r')
            indices = np.sort(np.random.default_rng(3407).choice(len(x),min(4096,len(x)),replace=False))
            refs = np.load(source/'residuals'/partition/'references.npy',mmap_mode='r')
            snippet_labels = metadata.loc[mapping['sample_ids'][partition]].sample_label.to_numpy()
            labels = snippet_labels[refs[indices,0]]
            distances = [[] for _ in model.memories]
            routes=[[] for _ in model.memories]
            flips=[[] for _ in model.memories]
            margins=[[] for _ in model.memories]
            generators=[torch.Generator().manual_seed(3407+s) for s in range(len(model.memories))]
            with torch.inference_mode():
                for lo in range(0,len(indices),128):
                    batch = torch.from_numpy(np.array(x[indices[lo:lo+128]]))
                    for s,(query,memory) in enumerate(zip(model.patches(batch),model.memories)):
                        normalized = (query-memory.center)/memory.scale
                        nearest = (normalized[:,:,None]-memory.prototypes).square().mean(-1).sqrt().amin(-1)
                        distances[s].append(nearest.numpy())
                        weights=memory(query)[2]
                        route=weights.argmax(-1)
                        noise=torch.randn(normalized.shape,generator=generators[s])*.01
                        perturbed=(((normalized+noise)@memory.prototypes.T)/(memory.temperature*np.sqrt(normalized.shape[-1]))).argmax(-1)
                        routes[s].append(route.numpy().astype('int16'))
                        flips[s].append((route!=perturbed).numpy())
                        top=weights.topk(2,dim=-1).values
                        margins[s].append((top[...,0]-top[...,1]).numpy())
            arrays = {f'scale_{s}':np.concatenate(parts) for s,parts in enumerate(distances)}
            routing={f'scale_{s}':np.concatenate(parts) for s,parts in enumerate(routes)}
            flipping={f'scale_{s}':np.concatenate(parts) for s,parts in enumerate(flips)}
            confidence={f'scale_{s}':np.concatenate(parts) for s,parts in enumerate(margins)}
            np.savez(p/f'{partition}_patch_distance_probe.npz',indices=indices,labels=labels,**arrays)
            np.savez(p/f'{partition}_patch_routing_probe.npz',indices=indices,labels=labels,
                **{k+'_route':v for k,v in routing.items()},**{k+'_noise_flip':v for k,v in flipping.items()},
                **{k+'_margin':v for k,v in confidence.items()})
            intervention_indices = indices[np.linspace(0,len(indices)-1,min(128,len(indices)),dtype=int)]
            probe = torch.from_numpy(np.array(x[intervention_indices]))
            original_hash = state_hash(model)
            with torch.inference_mode():
                baseline = model(probe)
                prototypes = [m.prototypes.clone() for m in model.memories]
                for memory in model.memories:
                    memory.prototypes.copy_(memory.prototypes.mean(0,keepdim=True).expand_as(memory.prototypes))
                constant_memory = model(probe)
                for memory,value in zip(model.memories,prototypes): memory.prototypes.copy_(value)
                mode = model.fusion_mode; model.fusion_mode='pure'
                compensation_removed = model(probe);model.fusion_mode=mode
            assert state_hash(model)==original_hash
            atomic_json(p/f'{partition}_memory_intervention_probe.json',dict(
                sampled_windows=len(probe),source='temporary CPU model copy; trained files untouched',
                reconstruction_mse=float((baseline-probe).square().mean()),
                memory_replaced_with_dictionary_mean_output_mse=float((constant_memory-baseline).square().mean()),
                memory_replaced_with_dictionary_mean_reconstruction_mse=float((constant_memory-probe).square().mean()),
                compensation_removed_output_mse=float((compensation_removed-baseline).square().mean()),
                original_parameters_restored_exactly=True,
                intervention_not_retrained_fusion_ablation=True))
            for s,value in enumerate(arrays.values()):
                for label in np.unique(labels):
                    values = value[labels==label].ravel()
                    matched=routing[f'scale_{s}'][labels==label]
                    records.append(dict(unit=unit,variant=variant,partition=partition,scale=s,label=int(label),
                        sampled_windows=int((labels==label).sum()),patches=len(values),
                        adjacent_patch_switch_rate=float(np.mean(matched[:,1:]!=matched[:,:-1])),
                        standardized_query_noise=.01,
                        perturbation_argmax_flip_fraction=float(flipping[f'scale_{s}'][labels==label].mean()),
                        attention_margin_p50=float(np.median(confidence[f'scale_{s}'][labels==label])),
                        distance_p05=float(np.quantile(values,.05)),distance_p50=float(np.median(values)),
                        distance_p95=float(np.quantile(values,.95)),distance_p99=float(np.quantile(values,.99))))
    return records


def variable_and_state_diagnostics(root,unit):
    """Full-window variable errors and label-specific soft prototype occupancy."""
    source,_ = source_for(unit)
    mapping = json.loads((source/'mapping.json').read_text())
    metadata = pd.read_csv(source/'metadata.csv',dtype={'vin':str,'car':str}).set_index('sample_id')
    rows,occupancy = [],[]
    for partition in ['train','test']:
        refs = np.load(source/'residuals'/partition/'references.npy',mmap_mode='r')
        labels = metadata.loc[mapping['sample_ids'][partition]].sample_label.to_numpy()[refs[:,0]]
        for variant in VARIANTS:
            p=root/unit/variant
            moment=np.load(p/f'{partition}_moments.npy',mmap_mode='r')
            for label in np.unique(labels):
                mask=labels==label
                for j,feature in enumerate(mapping['feature_columns']):
                    raw_mse=moment[mask,1,j]+moment[mask,0,j]**2
                    absolute=moment[mask,2,j]
                    for name,value in [('window_RMS',np.sqrt(raw_mse)),('window_mean_absolute_error',absolute)]:
                        rows.append(dict(unit=unit,variant=variant,partition=partition,label=int(label),
                            feature=feature,statistic=name,windows=int(mask.sum()),
                            p05=float(np.quantile(value,.05)),p50=float(np.median(value)),
                            p95=float(np.quantile(value,.95)),p99=float(np.quantile(value,.99))))
            if variant in ['A2','A4']:
                states=np.load(p/f'{partition}_memory_state.npy',mmap_mode='r')
                banks=states.shape[1]//8
                for label in np.unique(labels):
                    mask=labels==label
                    for s in range(banks):
                        values=np.asarray(states[mask,s*8:(s+1)*8],dtype=np.float64)
                        soft=values.mean(0)
                        dominant=np.bincount(values.argmax(1),minlength=8)/len(values)
                        for prototype in range(8):
                            occupancy.append(dict(unit=unit,variant=variant,partition=partition,label=int(label),
                                scale=s,prototype=prototype,soft_patch_occupancy=float(soft[prototype]),
                                dominant_window_share=float(dominant[prototype]),windows=len(values)))
    return rows,occupancy


def dataset_intervals(root,dataset,group,repeats):
    signature=dict(pairs=PAIRS,units=group,repeats=repeats,seed=3407,
        score_sha256={u+'/'+v:digest(root/u/v/'test_scores.csv') for u in group for v in VARIANTS})
    # Normalize tuple/list serialization before comparing a cached protocol.
    signature=json.loads(json.dumps(signature))
    csv=root/f'dataset_paired_{dataset}.csv'
    manifest=root/f'dataset_paired_{dataset}_protocol.json'
    if csv.exists() and manifest.exists() and json.loads(manifest.read_text())==signature:
        return read(csv).to_dict('records')
    entries=[combined(root,u) for u in group]
    result=[]
    for q in [.95,.99]:result+=paired_bootstrap(entries,PAIRS,repeats,dataset,q)
    pd.DataFrame(result).to_csv(csv,index=False);atomic_json(manifest,signature)
    return result


def run(root,repeats,reuse_diagnostics=False):
    settings = json.loads((root/'protocol_manifest.json').read_text())
    units = settings['units']
    complete = [u for u in units if (root/u/'complete.json').exists()]
    summarize(root,units)
    if len(complete)!=len(units):
        print('MODEL ONE IN PROGRESS',len(complete),'/',len(units),'complete units',flush=True)
        return
    audits,distributions,memories,ci,patch_distances,variable_errors,label_occupancy = [],[],[],[],[],[],[]
    if reuse_diagnostics:
        audits=json.loads((root/'artifact_audit.json').read_text())['units']
        distributions=read(root/'normal_fault_score_quantiles.csv').to_dict('records')
        memories=read(root/'memory_diagnostics_summary.csv').to_dict('records')
        patch_distances=read(root/'normal_fault_patch_distance_quantiles.csv').to_dict('records')
        variable_errors=read(root/'normal_fault_variable_reconstruction.csv').to_dict('records')
        label_occupancy=read(root/'normal_fault_prototype_occupancy.csv').to_dict('records')
    else:
        for unit in complete:
            a,d,m = audit_unit(root,unit);audits.append(a);distributions+=d;memories+=m
            local_analysis(root,unit)
            patch_distances += patch_distribution_probe(root,unit)
            v,o=variable_and_state_diagnostics(root,unit);variable_errors+=v;label_occupancy+=o
    for dataset,group in [('Brand3',[u for u in units if u.startswith('Brand3')]),
                           ('NCM',[u for u in units if u=='NCM']),('LFP',[u for u in units if u=='LFP'])]:
        if not group: continue
        ci += dataset_intervals(root,dataset,group,repeats)
    pd.DataFrame(ci).to_csv(root/'paired_VIN_intervals.csv',index=False)
    pd.DataFrame(distributions).to_csv(root/'normal_fault_score_quantiles.csv',index=False)
    pd.DataFrame(memories).to_csv(root/'memory_diagnostics_summary.csv',index=False)
    pd.DataFrame(patch_distances).to_csv(root/'normal_fault_patch_distance_quantiles.csv',index=False)
    pd.DataFrame(variable_errors).to_csv(root/'normal_fault_variable_reconstruction.csv',index=False)
    pd.DataFrame(label_occupancy).to_csv(root/'normal_fault_prototype_occupancy.csv',index=False)
    code_status = {p:digest(p)==value for p,value in settings['implementations'].items()}
    assert all(code_status.values()), 'Training implementation changed after protocol was frozen'
    atomic_json(root/'artifact_audit.json',dict(units=audits,normal_only_selection=True,
        inference_frozen=True,bootstrap=repeats,diagnostics_reused=reuse_diagnostics,
        code_unchanged={p:digest(p)==value for p,value in settings['implementations'].items()}))
    all_metrics = read(root/'all_completed_metrics.csv')
    summary = all_metrics.query('q==.95').groupby(['dataset','variant','score'])[
        ['ap','auroc','f1','recall','fpr','recall_at_test_fpr05']].mean()
    datasets = [d for d in ['LFP','NCM','Brand3'] if d in all_metrics.dataset.unique()]
    references = pd.concat([read(PRIOR/u/'metrics.csv').query('method=="A" and q==.95').assign(
        unit=u,dataset='Brand3' if u.startswith('Brand3') else u) for u in units],ignore_index=True)
    references.to_csv(root/'original_RCR_reference_metrics.csv',index=False)
    reference_mean = references.groupby('dataset')[['ap','auroc','f1']].mean()
    lines = ['# MSCM-Net V1：模型一核心机制开发验证', '',
        '按附件编号查看完整的[A0–A5、B0–B5结果表](MSCM_MCR_ABLATION_INDEX_20261025.md)。', '',
        '这是新训练的多尺度Patch Memory重构实验。本报告仅记录模型一，条件GAT结果在模型二报告单独保存。', '',
        f'任务 `{settings["task"]}`，各重构组阶段A最多{settings["pretrain_epochs"]}个完整epoch、阶段B最多{settings["memory_epochs"]}个完整epoch。正常验证重构损失选检查点，FP32；所有训练/评估窗口均保留。结果属于已反复观察数据集上的首轮开发证据，不等于收敛后的正式比较。', '',
        'A0为原结构同预算继续训练；A1为多尺度；A2为单尺度Memory；A4为多尺度Memory；A6为同总参数量、同可训练参数量普通瓶颈。A3合并到A1；A5直接使用A4/raw。', '',
        '**基线来源：本表为RCR Scale校准幅值参考。CH沿用原检查点；Brand3使用RCR-MEM-BRAND3-20261025-v1本地重训检查点，未复用历史Kaggle权重。Brand3本地AP/AUROC/F1为0.8685/0.9156/0.6789；历史Scale为0.8728/0.9240/0.7236，属于不同训练结果。A0_reference及配对区间指本表来源，不能解释为与历史Kaggle权重的同检查点比较。历史报告指标顺序为AUROC/AP/F1/Recall/FPR，见[RCR主报告](RCR_REPORT_ZH.md)。**', '',
        '| 重构方法与评分 | '+' | '.join(d+' AP / AUROC / F1' for d in datasets)+' |',
        '| --- | '+' | '.join('---' for _ in datasets)+' |']
    lines.append('| RCR Scale参考（CH原检查点；Brand3本地重训） | '+' | '.join(
        f'{reference_mean.loc[d,"ap"]:.4f} / {reference_mean.loc[d,"auroc"]:.4f} / {reference_mean.loc[d,"f1"]:.4f}' for d in datasets)+' |')
    for variant in VARIANTS:
        for score,label in [('raw','未校准'),('cal_abs','绝对残差固定稳健尺度'),('rcr_legacy','原RCR校准规则重新拟合')]:
            cells = []
            for d in datasets:
                r = summary.loc[d,variant,score]
                cells.append(f'{r.ap:.4f} / {r.auroc:.4f} / {r.f1:.4f}')
            lines.append('| '+variant+' / '+label+' | '+' | '.join(cells)+' |')
    lines += ['', '主比较统一使用cal_abs后处理；raw分数用于确认重构路径的独立贡献。rcr_legacy将原RCR的条件均值/条件尺度规则重新拟合到每个新重构网络，而非复用旧残差参考。cal_signed另存给后续关系模型，不能与绝对残差参考混用。', '',
        '| 数据集 | 对照 → 方法 | AUROC差值 [95% VIN配对区间] |', '| --- | --- | --- |']
    cis = pd.DataFrame(ci)
    for r in cis.query('metric=="auroc" and q==.95').itertuples():
        lines.append(f'| {r.dataset} | {r.base} → {r.contender} | {r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] |')
    lines += ['', f'{repeats}次VIN分层成组bootstrap（seed3407）；Brand3跨折重复车辆共用权重。区间固定已训练网络和正常验证阈值，不包含重训练、阈值估计或多重检验不确定性。', '',
        '| 数据集/折 | A4尺度 | 正常训练有效软原型 | 硬占用≥1%的原型 | 检索/查询表示方差 |',
        '| --- | --- | --- | --- | --- |']
    for r in pd.DataFrame(memories).query('variant=="A4" and partition=="train"').itertuples():
        ratio = r.retrieved_variance_mean/max(r.query_variance_mean,1e-30)
        lines.append(f'| {r.unit} | {r.scale} (k={r.kernel}) | {r.effective_soft_prototypes:.2f} | {r.hard_prototypes_above_one_percent} / 8 | {ratio:.4f} |')
    convergence = []
    for unit in units:
        for variant in VARIANTS:
            result = json.loads((root/unit/variant/'training/complete.json').read_text())
            convergence.append(dict(unit=unit,variant=variant,best_epoch=result['best_epoch'],
                epochs_executed=result['epochs_executed'],normal_validation_huber=result['normal_validation_huber'],
                best_at_budget_boundary=result['best_epoch']==settings['memory_epochs']))
    pd.DataFrame(convergence).to_csv(root/'normal_validation_convergence.csv',index=False)
    boundary_count = sum(r['best_at_budget_boundary'] for r in convergence if r['variant']=='A4')
    lines += ['', f'A4在{boundary_count}/{len(units)}个单元的最优正常验证检查点位于阶段B预算上限。预算边界不等于已经收敛，尤其不能将尚在改善的正常拟合直接解释为结构的最终上限。逐单元实际epoch及验证误差见`normal_validation_convergence.csv`。']
    lines += ['', '| A4测试组件干预 | 字典替换为均值：输出变化MSE | 移除补偿：输出变化MSE |',
              '| --- | --- | --- |']
    intervention=[]
    for unit in units:
        result=json.loads((root/unit/'A4/test_memory_intervention_probe.json').read_text())
        intervention.append(dict(unit=unit,**result))
        lines.append(f'| {unit} | {result["memory_replaced_with_dictionary_mean_output_mse"]:.6g} | {result["compensation_removed_output_mse"]:.6g} |')
    pd.DataFrame(intervention).to_csv(root/'trained_memory_interventions.csv',index=False)
    lines += ['', '干预确认已训练Memory会影响重构，同时CH的补偿路径对输出很重要；不能仅凭γ=0.1就宣称Decoder主要依赖Memory。两种干预的扰动幅度与分布不同，不能用MSE比值计算贡献占比或直接断言Memory被绕过。纯Memory/直接融合/有界补偿的重新训练对照仍有必要，现有干预不替代它。']
    lines += ['', '软原型数较高不保证没有坍缩，需同时检查硬占用、原型距离、检索表示方差与raw检测指标。`observed_epoch_memory.json`是独立CPU观察器在固定512个正常训练窗口上的逐检查点记录，可能遗漏覆盖间隔内的epoch，不伪造缺失记录；最终全量正常训练占用保存在各单元。', '',
        '- 每折完整结果：`all_completed_metrics.csv`；固定测试ROC FPR召回与P95/P99阈值指标全部保存。固定测试ROC FPR仅为描述性指标，不是部署阈值。',
        '- 正常/故障分布：`normal_fault_score_quantiles.csv`；局部工况/故障类型：各数据集`local_descriptive_metrics.csv`。所有局部结果均为探索性。',
        '- 逐变量正常/故障重构幅值：`normal_fault_variable_reconstruction.csv`，为正常训练MinMax归一化后的单位；标签分组软原型占用：`normal_fault_prototype_occupancy.csv`。后者另列窗口主导原型占比，不能与逐Patch硬占用混淆。',
        '- 每尺度原型占用、距离、方差：`memory_diagnostics_summary.csv`和各单元`*_memory_diagnostics.json`；全量窗口状态和最近匹配距离另存NPY。',
        '- 正常/故障最近Patch距离：`normal_fault_patch_distance_quantiles.csv`。每分区固定抽4096个窗口的每个Patch精确距离，索引和标签均保存；这项抽样只用于表示诊断，主检测指标仍来自全量窗口。全量NPY距离为窗口内平均最近Patch距离。',
        '- 同表另列窗口内相邻Patch原型切换率、0.01训练标准化特征单位高斯扰动下的匹配翻转率，以及注意力前两项权重差；路由数组保存在`*_patch_routing_probe.npz`。这些是标签分组诊断，不是异常分数或故障发生时刻。Brand3仅比较真实窗口内Patch，不虚构跨片段时间顺序。',
        '- `*_memory_intervention_probe.json`在临时CPU模型副本上将字典替换为均值、移除有界补偿，检查已训练模型是否实际依赖Memory。参数随后精确恢复；这只是组件干预，不能替代重训的融合消融。',
        '- A4完整有符号残差、绝对/有符号独立校准器与检查点已保存，供随后冻结模型一训练模型二使用。',
        '- 重构与校准仍使用同一原正常训练集合，没有独立校准VIN或交叉拟合；原预训练也使用过这些正常训练数据。',
        '- 纯Memory/直接融合/有界补偿、λ=0/0.01/0.1、Patch=4/8/16及编码器微调均为后续单独消融，不能由本次核心组结果代替。',
        '- 未修改论文最终四张实验矩阵。', '']
    target = Path('report/analysis/rcr/MSCM_MODEL_ONE_CORE_20261025.md')
    target.write_text(bold_best_metrics('\n'.join(lines)))
    figures = Path('report/figures/mscm_model_one_20261025');figures.mkdir(parents=True,exist_ok=True)
    fig,axes = plt.subplots(1,len(datasets),figsize=(5*len(datasets),4),squeeze=False)
    for ax,d in zip(axes[0],datasets):
        x = np.arange(len(VARIANTS))
        ax.bar(x-.18,[summary.loc[d,v,'raw'].auroc for v in VARIANTS],.36,label='Raw residual')
        ax.bar(x+.18,[summary.loc[d,v,'cal_abs'].auroc for v in VARIANTS],.36,label='Calibrated absolute residual')
        ax.set(title=d,xticks=x,xticklabels=VARIANTS,ylabel='AUROC',ylim=(0,1));ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(figures/f'model_one_core.{suffix}',dpi=180)
    plt.close(fig)
    atomic_json(root/'model_one_analysis_complete.json',dict(report=str(target),complete_units=complete,
        scope='model_one_only',model_two_status_reported_separately=True,core_cells=len(units)*len(VARIANTS)))
    print('MSCM MODEL ONE ANALYSIS COMPLETE',target,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='runs/internal/mscm_mcr_development_20261025_v2')
    parser.add_argument('--bootstrap',type=int,default=500)
    parser.add_argument('--reuse-diagnostics',action='store_true')
    args = parser.parse_args();run(Path(args.root),args.bootstrap,args.reuse_diagnostics)
