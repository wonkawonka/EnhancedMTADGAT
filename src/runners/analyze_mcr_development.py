"""Audit completed neural MCR-GAT models, error overlap and paired VIN evidence."""
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

from src.runners.run_mcr_development import HEADS
from src.runners.probe_scale_relation_evidence import read,paired_bootstrap,rank_frame
from src.runners.run_rcr import atomic_json
from src.runners.run_rcr_memory_probe import digest
from scripts.evaluate_rcr import metrics
from src.models.mcr_gat import NormalContextMemoryPrior
from src.runners.run_local_residual_models import source_for
from src.runners.run_rcr_calibration_sensitivity import aggregate
from src.models.rcr import EmpiricalCDF
from sklearn.metrics import roc_curve


PAIRS=[('B2_gat','B3_memory_gat'),('B4_memory_mlp','B3_memory_gat'),
    ('B3_q_only_gat','B3_memory_gat'),('B1_original','B3_memory_gat'),
    ('B3_scaled_marginal','B3_memory_gat'),('B0_diagonal_ECDF_0p5','B5_ECDF_0p5'),
    ('B0_signed_L1','B3_memory_gat'),('B0_signed','B3_memory_gat'),('B0','B3_memory_gat'),
    ('B0','B5_ECDF_0p5'),('B0_ECDF','B5_ECDF_0p5'),
    ('B0','B5_MAD_0p5'),('B0_MAD','B5_MAD_0p5'),
    ('B0_raw','B5_ECDF_0p5'),('B0_raw','B5_MAD_0p5')]


def audit_unit(core,root,unit):
    output=root/unit;parent=core/unit/'A4'
    error=0.;audits=[]
    for partition in ['train','validation','test']:
        original=read(parent/f'{partition}_scores.csv')
        actual=read(output/f'{partition}_scores.csv')
        for new,old in [('B0','cal_abs'),('B0_raw','raw'),('B0_signed','cal_signed'),('B0_legacy','rcr_legacy')]:
            np.testing.assert_array_equal(actual[new].to_numpy(),original[old].to_numpy())
        for name in HEADS:
            e=np.load(output/name/f'{partition}_variable_mae.npy',mmap_mode='r')
            scale=np.load(output/name/'normal_relation_error_scale.npy')
            score=np.load(output/name/f'{partition}_window_scores.npy',mmap_mode='r')
            assert np.isfinite(e).all() and (e>=0).all() and (scale>=1e-5).all()
            np.testing.assert_array_equal((e/scale).mean(1),score)
    test=read(output/'test_scores.csv');saved=read(output/'metrics.csv')
    for row in saved.itertuples():
        recalculated=metrics(test.sample_label,test[row.method],row.threshold)
        for key,value in recalculated.items():
            error=max(error,abs(value-getattr(row,key)))
            assert abs(value-getattr(row,key))<1e-12
    for name in HEADS:
        a=json.loads((output/name/'mask_and_condition_audit.json').read_text())
        assert a['trained_target_perturbation_max_abs_error']<1e-6
        training=json.loads((output/name/'training_complete.json').read_text())
        initial=json.loads((output/name/'initial.json').read_text())
        audits.append(dict(unit=unit,head=name,parameters=initial['parameters'],
            best_epoch=training['best_epoch'],epochs_executed=training['epochs_executed'],
            normal_validation_loo_huber=training['normal_validation_loo_huber'],**a))
    assert next(a['parameters'] for a in audits if a['head']=='B3_memory_gat')==next(a['parameters'] for a in audits if a['head']=='B3_q_only_gat')
    return dict(unit=unit,original_model_one_scores_exact=True,max_metric_error=error),audits


def local_analysis(root,unit):
    if unit.startswith('Brand3'):return
    output=root/unit
    meta=read(Path('runs/internal/rcr_local_two_models_20261025_v1')/unit/'test_snippet_scores.csv')
    test=read(output/'test_snippet_scores.csv')
    test=test.drop(columns=[c for c in ['operating_regime','fault_type'] if c in test]).merge(
        meta[['sample_id','operating_regime','fault_type']],on='sample_id',validate='one_to_one')
    thresholds=json.loads((output/'thresholds.json').read_text())
    methods=['B0','B0_raw','B0_signed_L1']+list(HEADS)
    rows=[]
    for regime,group in test.groupby('operating_regime'):
        normal=group[group.sample_label==0]
        for fault,affected in group[group.sample_label==1].groupby('fault_type'):
            if min(len(normal),len(affected))<10:continue
            frame=pd.concat([normal,affected])
            for method in methods:
                rows.append(dict(unit=unit,regime=regime,fault_type=fault,method=method,
                    normal_snippets=len(normal),fault_snippets=len(affected),normal_vins=normal.vin.nunique(),
                    fault_vins=affected.vin.nunique(),**metrics(frame.sample_label,frame[method],thresholds[method]['0.95'])))
    pd.DataFrame(rows).to_csv(output/'local_fault_regime_metrics.csv',index=False)


def context_quality(core,root,unit):
    """Fixed normal-only diagnostic of the control-to-Memory state projection."""
    source,_=source_for(unit)
    prior=NormalContextMemoryPrior.load(root/unit/'normal_context_memory_prior.npz').eval()
    records=[];torch.set_num_threads(2)
    for partition in ['train','validation']:
        q=np.load(source/'residuals'/partition/'condition.npy',mmap_mode='r')
        state=np.load(core/unit/'A4'/f'{partition}_memory_state.npy',mmap_mode='r')
        indices=np.linspace(0,len(q)-1,min(16384,len(q)),dtype=int)
        square=np.zeros(3);kl=np.zeros(3);count=0
        with torch.inference_mode():
            for lo in range(0,len(indices),1024):
                ix=indices[lo:lo+1024]
                predicted=prior(torch.from_numpy(np.array(q[ix])))[:,[0,1,2],8:].reshape(len(ix),3,3,8)
                expected=torch.from_numpy(np.array(state[ix]))[:,None].reshape(len(ix),1,3,8)
                square+=((predicted-expected).square().mean((2,3)).sum(0)).numpy()
                divergence=(expected*(expected.clamp_min(1e-6).log()-predicted.clamp_min(1e-6).log())).sum(-1).mean(-1)
                kl+=divergence.sum(0).numpy();count+=len(ix)
        for j,label in enumerate(['response_full_q','current_without_current','SOC_without_SOC']):
            records.append(dict(unit=unit,partition=partition,context=label,sampled_normal_windows=count,
                memory_occupancy_mse=float(square[j]/count),memory_occupancy_KL=float(kl[j]/count),
                selection='fixed evenly spaced normal windows; diagnostic only'))
    return records


def marginal_controls(root,unit):
    """Hold B3's variable weights fixed while removing cross-variable prediction."""
    source,_=source_for(unit);output=root/unit
    mapping=json.loads((source/'mapping.json').read_text())
    scale=np.load(output/'B3_memory_gat/normal_relation_error_scale.npy')
    frames={}
    for partition in ['validation','test']:
        absolute=np.load(output/f'{partition}_signed_absolute_mean.npy',mmap_mode='r')
        window=(absolute/scale).mean(1)
        refs=np.load(source/'residuals'/partition/'references.npy',mmap_mode='r')
        snippets=aggregate(window[:,None],refs,mapping['sample_ids'][partition],['B3_scaled_marginal'])
        metadata=read(output/f'{partition}_snippet_scores.csv')[['sample_id','vin','sample_label']]
        snippets=metadata.merge(snippets,on='sample_id',validate='one_to_one')
        ranked=rank_frame(snippets,['B3_scaled_marginal'],unit.startswith('Brand3'))
        keys=['vin','sample_label'] if unit.startswith('Brand3') else ['sample_id','vin','sample_label']
        frames[partition]=read(output/f'{partition}_scores.csv').merge(ranked[keys+['B3_scaled_marginal']],on=keys,validate='one_to_one')
    cdf=EmpiricalCDF().fit(frames['validation'].B3_scaled_marginal)
    for partition,f in frames.items():
        f['B0_diagonal_ECDF_0p5']=.5*(f.B0_ECDF+cdf.transform(f.B3_scaled_marginal.to_numpy()))
        f.to_csv(output/f'{partition}_marginal_control_scores.csv',index=False)
    thresholds=json.loads((output/'thresholds.json').read_text());rows=[]
    for name in ['B3_scaled_marginal','B0_diagonal_ECDF_0p5']:
        thresholds[name]={}
        for q in [.95,.99]:
            value=float(frames['validation'][name].quantile(q));thresholds[name][str(q)]=value
            test=frames['test'];fpr,tpr,_=roc_curve(test.sample_label,test[name])
            rows.append(dict(unit=unit,method=name,q=q,threshold=value,
                validation_fpr=float(np.mean(frames['validation'][name]>=value)),
                recall_at_test_fpr05=float(tpr[fpr<=.05].max(initial=0.)),
                **metrics(test.sample_label,test[name],value)))
    pd.DataFrame(rows).to_csv(output/'marginal_control_metrics.csv',index=False)
    atomic_json(output/'marginal_control_protocol.json',dict(
        variable_scale='exact frozen B3 normal prediction-error scale',cross_variable_prediction=False,
        input='same signed calibrated residual',fusion='normal validation ECDF, fixed0.5',test_fitting=False))
    return frames,thresholds


def run(root,repeats):
    assert (root/'complete.json').exists(),'Finish all requested model-two units first'
    protocol=json.loads((root/'protocol_manifest.json').read_text());core=Path(protocol['core'])
    units=protocol['units'];audit,heads,ci,quality=[],[],[],[]
    entries_by_unit={}
    for unit in units:
        a,h=audit_unit(core,root,unit);audit.append(a);heads+=h
        local_analysis(root,unit)
        quality+=context_quality(core,root,unit)
        entries_by_unit[unit]=marginal_controls(root,unit)
    for dataset in ['Brand3','NCM','LFP']:
        group=[u for u in units if u.startswith('Brand3')] if dataset=='Brand3' else [u for u in units if u==dataset]
        if not group:continue
        entries=[entries_by_unit[u] for u in group]
        for q in [.95,.99]:ci+=paired_bootstrap(entries,PAIRS,repeats,dataset,q)
        print('MCR PAIRED BOOTSTRAP COMPLETE',dataset,flush=True)
    metrics_=pd.concat([read(root/u/filename) for u in units for filename in ['metrics.csv','marginal_control_metrics.csv']],ignore_index=True)
    metrics_['dataset']=np.where(metrics_.unit.str.startswith('Brand3'),'Brand3',metrics_.unit)
    metrics_.to_csv(root/'all_metrics.csv',index=False)
    summary=metrics_.groupby(['dataset','method','q'])[['ap','auroc','f1','recall','fpr','recall_at_test_fpr05']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(root/'summary_metrics.csv',index=False)
    cis=pd.DataFrame(ci);cis.to_csv(root/'paired_VIN_intervals.csv',index=False)
    pd.DataFrame(heads).to_csv(root/'trained_head_audits.csv',index=False)
    pd.DataFrame(quality).to_csv(root/'normal_context_memory_quality.csv',index=False)
    overlap=pd.concat([read(root/u/'error_overlap.csv') for u in units],ignore_index=True)
    overlap.to_csv(root/'error_overlap.csv',index=False)
    unchanged={p:digest(p)==value for p,value in protocol['implementations'].items()}
    assert all(unchanged.values())
    for unit in units:
        signature=protocol['parent_signatures'][unit]
        parent=core/unit/'A4'
        assert digest(parent/'training/best.pt')==signature['checkpoint']
        assert digest(parent/'calibration_signed.npz')==signature['calibration']
        assert digest(parent/'calibration_absolute.npz')==signature['absolute_calibration']
    atomic_json(root/'artifact_audit.json',dict(units=audit,model_one_unchanged=True,code_unchanged=unchanged,
        all_normal_relation_score_replays_exact=True,all_trained_LOO_masks_passed=True,requested_bootstrap=repeats))
    table=summary.reset_index().query('q==.95').set_index(['dataset','method'])
    datasets=[d for d in ['LFP','NCM','Brand3'] if d in metrics_.dataset.unique()]

    def cell(d,m):
        r=table.loc[d,m];return f'{r.ap_mean:.4f} / {r.auroc_mean:.4f} / {r.f1_mean:.4f}'

    def interval(d,a,b,metric='auroc',q=.95):
        r=cis[(cis.dataset==d)&(cis.base==a)&(cis.contender==b)&(cis.metric==metric)&(cis.q==q)].iloc[0]
        return f'{r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]'

    lines=['# MCR-GAT V1：冻结 MSCM 模型一后的关系检测开发验证','',
        '按附件编号查看完整的[A0–A5、B0–B5结果表](MSCM_MCR_ABLATION_INDEX_20261025.md)。','',
        '**本轮训练了新的神经关系网络；模型一的重构器、Memory、绝对/有符号校准器与边缘分数均保持冻结。**','',
        f'任务 `{protocol["task"]}`。固定继承各数据集A4，没有按测试AP选择父模型。各关系头最多{protocol["epochs"]}个完整正常训练epoch，patience={protocol["patience"]}，仅正常验证全LOO Huber选择检查点。','',
        '## 架构和条件的实际实现','',
        '- B1：原RCR变量图＋变量Transformer掩码头，在新A4有符号校准残差上重新训练。不能复用旧残差分布下的关系头成绩。',
        '- B2：每变量共享两层TCN(k3,hidden32)，4个时间分区池化为32维Token，普通2-head单层星形GAT预测目标完整残差序列。Query仅来自目标身份，Key/Value排除目标。',
        '- B3：同一TCN/GAT骨架，增加4个rank4低秩正常图模板，按正常上下文混合为边偏置。',
        '- B4：同TCN、同条件的无图MLP，目标Token在拼接前清零。',
        '- 附加工况条件GAT：与B3具有完全相同参数量，将Memory坐标固定为正常训练平均占用，只保留q变化，检查Memory先验是否超过普通工况条件。','',
        '按用户附件“首版优先控制上下文”的建议，B3条件使用当前电流/SOC统计，以及由正常训练Memory占用拟合的固定状态先验。采用normal-only ridge=0.001的log占用回归，逐尺度softmax；预测电流时移除电流统计，预测SOC时移除SOC统计，控制外推截断固定为训练P1/P99。首版没有直接使用测试多变量输入的Memory检索权重。完整多变量Memory条件仍属于后续独立对照。','',
        '正常Memory状态在这个版本中是q的确定函数，因此它提供的是表示/结构先验，不能声称带来了独立于q的新观测信息。B3与同参数量q-only GAT的差值才支持该先验的独立价值。','',
        '训练对每个正常窗口均匀抽一个完整变量作为目标；五种关系头使用相同窗口顺序和目标日程，这是完整LOO损失的无偏随机估计。验证和主评分均遍历全部变量、全部时刻、全部窗口。关系分数为有符号Z预测绝对误差，以正常训练窗口MAE中位数逐变量标准化，再时间/变量均值。','',
        '| 方法（关系单支优先比较） | '+' | '.join(d+' AP / AUROC / F1' for d in datasets)+' |',
        '| --- | '+' | '.join('---' for d in datasets)+' |']
    methods=[('B0','B0：模型一校准边缘评分'),('B0_raw','模型一未校准重构'),
        ('B0_signed_L1','同有符号Z的边缘L1对照'),('B1_original','B1：原RCR关系头'),
        ('B2_gat','B2：普通GAT'),('B3_memory_gat','B3：Memory先验条件GAT'),
        ('B4_memory_mlp','B4：同条件MLP'),('B3_q_only_gat','附加：同参数q-only条件GAT'),
        ('B3_scaled_marginal','附加：仅B3变量权重，无关系预测'),
        ('B5_ECDF_0p5','B5：正常验证ECDF等权融合'),('B5_MAD_0p5','连续median/MAD等权融合')]
    for method,label in methods:lines.append('| '+label+' | '+' | '.join(cell(d,method) for d in datasets)+' |')
    lines+=['','## 核心差值与不确定性','',
        '| 数据集 | 对照 → 方法 | AUROC差值 [95% VIN配对区间] |','| --- | --- | --- |']
    for d in datasets:
        for a,b in PAIRS:lines.append(f'| {d} | {a} → {b} | {interval(d,a,b)} |')
    lines+=['','只有B3同时超过B2、B4及q-only条件GAT，才支持这版Memory条件图结构的独立贡献；单个点估计提升或图偏置非零都不足以证明它提升了检测。与同有符号边缘L1对照比较，进一步排除仅更换残差方向/评分范数的收益。另将B3的正常误差尺度原样用于边缘L1，取消跨变量预测，用来排除仅变量重新加权的收益；融合也提供同样的ECDF等权对照。该对照不新增训练或测试选择。','',
        '## 实际检出与误报','',
        '| 数据集 | 方法 | Recall（正常验证P95） | 测试FPR |','| --- | --- | --- | --- |']
    for d in datasets:
        for m in ['B0','B3_memory_gat','B5_ECDF_0p5','B5_MAD_0p5']:
            r=table.loc[d,m];lines.append(f'| {d} | {m} | {r.recall_mean:.4f} | {r.fpr_mean:.4f} |')
    lines+=['','| 数据集/折 | 与模型一共同检出 | 仅模型一检出 | 仅B3补检 | 两者均漏检 | B3新增正常误报 |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in overlap.query('q==.95 and base=="B0"').itertuples():
        lines.append(f'| {r.unit} | {r.fault_both} | {r.fault_base_only} | {r.fault_relation_only} | {r.fault_neither} | {r.normal_relation_only} |')
    lines+=['','CH数量为片段，Brand3为车辆；Brand3同一九辆故障车跨折重复，不能直接把五折计数相加视为独立故障车。逐对象错检/漏检集合包含VIN和分数，保存在各单元`prediction_sets.csv`。','',
        '融合在原窗口/片段/车辆Top5聚合后、最终决策对象层进行，先按正常验证ECDF或连续median/MAD标准化，再固定α=0/0.5/1。预先指定ECDF等权融合为主，连续MAD为对照；不按测试结果切换策略。端点B0_ECDF/B3_ECDF及B0_MAD/B3_MAD全保存，用于检查尾部分位数离散化的影响。各方法重新使用正常验证P95/P99定阈值；固定测试FPR Recall仅是描述性ROC指标。','',
        '## 机制审计和边界','',
        '- 每个已训练头均在正常验证样本上实际扰动被预测残差坐标，目标预测差值小于1e-6；控制条件同时排除目标电流/SOC统计。',
        '- B3额外保存将Memory条件置为正常均值、将图偏置置零的组件干预；操作仅在临时推理状态中执行，参数精确恢复，不能替代重新训练的对照。',
        f'- {repeats}次标签分层VIN成组配对bootstrap，seed3407；跨折重复车辆共用权重。固定已训练模型与正常验证阈值，不包含重训练/阈值估计不确定性，未作多重比较校正。',
        '- 这是已多轮探索数据集上的首轮短预算开发证据；父模型是否收敛及其跨数据集退化仍须结合模型一报告评估。',
        '- `all_metrics.csv`、`summary_metrics.csv`、`paired_VIN_intervals.csv`保存全部指标及区间；`trained_head_audits.csv`保存训练epoch、参数量、目标排除和条件图干预。',
        '- `normal_context_memory_quality.csv`在固定正常训练/验证窗口上报告控制上下文预测Memory占用的MSE/KL，用于评估先验投影是否保留正常状态；不使用故障标签、不改变固定模型配置。',
        f'- `artifact_audit.json`核验模型一分数原样继承、关系误差全量归一化重放和{len(metrics_)}行主指标复算。CH各工况/故障类型的描述性指标和样本量另存各单元`local_fault_regime_metrics.csv`，不作为多重比较校正后的独立证据。',
        '- 本轮没有修改论文最终四张实验矩阵；没有将旧统计LOO结果当作新神经模型结果。','']
    target=Path('report/analysis/rcr/MCR_GAT_MODEL_TWO_20261025.md');target.write_text(bold_best_metrics('\n'.join(lines)))
    figures=Path('report/figures/mcr_gat_model_two_20261025');figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,len(datasets),figsize=(5*len(datasets),4),squeeze=False)
    chosen=['B0','B1_original','B2_gat','B3_memory_gat','B4_memory_mlp','B3_q_only_gat','B5_ECDF_0p5']
    for ax,d in zip(axes[0],datasets):
        ax.bar(np.arange(len(chosen)),[table.loc[d,m].auroc_mean for m in chosen])
        ax.set(title=d,xticks=np.arange(len(chosen)),xticklabels=['M1','RCR head','GAT','Mem-GAT','MLP','q-GAT','Joint'],ylabel='AUROC',ylim=(0,1))
        ax.tick_params(axis='x',rotation=35)
    fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(figures/f'model_two_comparison.{suffix}',dpi=180)
    plt.close(fig)
    atomic_json(root/'analysis_complete.json',dict(report=str(target),units=units,neural_heads=len(HEADS)*len(units),
        metric_rows=len(metrics_),model_one_preserved=True))
    print('MCR ANALYSIS COMPLETE',target,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='runs/internal/mcr_gat_development_20261025_v1')
    parser.add_argument('--bootstrap',type=int,default=500)
    args=parser.parse_args();run(Path(args.root),args.bootstrap)
