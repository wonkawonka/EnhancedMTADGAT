"""Small CPU-only progress record for the five-priority background pipeline."""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def last_row(path):
    if not path.exists():return None
    try:
        with path.open() as f:rows=list(csv.DictReader(f))
        return rows[-1] if rows else None
    except (OSError,csv.Error):return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='runs/internal/mscm_mcr_priority5_20261025_v1')
    args=parser.parse_args();root=Path(args.root)
    protocol=json.loads((root/'model_one_protocol.json').read_text())
    units=protocol['units'];arms=list(protocol['arms'])
    primary=['B1_original','B2_gat','B3_memory_gat','B4_memory_mlp','B3_prior_gat','B4_prior_mlp','B3_q_only_gat']
    active=[]
    while True:
        active=[];counts=[];relations=0
        for unit in units:
            done=[]
            for arm in arms:
                output=root/'model_one'/unit/arm
                if (output/'complete.json').exists():done.append(arm)
                elif (output/'training/initial.json').exists():
                    row=last_row(output/'training/history.csv')
                    active.append(dict(unit=unit,arm=arm,epoch=row.get('epoch') if row else None,
                        validation=row.get('normal_validation_huber') if row else None))
            final=(root/'model_one'/unit/'A4_final/complete.json').exists()
            heads=[]
            for seed in [3407,3408,3409]:
                seed_dir=root if seed==3407 else root/f'relation_seed{seed}'
                for head in primary if seed==3407 else ['B3_memory_gat','B4_memory_mlp']:
                    output=seed_dir/'model_two'/unit/head
                    if (output/'complete.json').exists():heads.append(f'{head}/{seed}');relations+=1
                    elif (output/'training/initial.json').exists():
                        row=last_row(output/'training/history.csv')
                        active.append(dict(unit=unit,arm=f'{head}/{seed}',epoch=row.get('epoch') if row else None,
                            validation=row.get('normal_validation_huber') if row else None))
            counts.append(dict(unit=unit,completed_reconstruction_arms=done,final_checkpoint_scored=final,
                completed_relation_heads=heads))
        finished=(root/'analysis_complete.json').exists()
        state=dict(updated_at=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(timespec='seconds'),
            scope='five core priorities; original cohort and teacher',
            reconstruction_completed=sum(len(r['completed_reconstruction_arms']) for r in counts),
            reconstruction_requested=len(arms)*len(units),relation_heads_completed=relations,
            relation_heads_requested=11*len(units),model_one_analysis_complete=(root/'model_one_analysis_complete.json').exists(),
            all_analysis_complete=finished,units=counts,active=active)
        execution=root/'model_two_execution.json'
        if execution.exists():
            state['model_two_execution']=json.loads(execution.read_text())
        external=root/'kaggle_pipeline_status.json'
        if external.exists():
            state['kaggle']=json.loads(external.read_text())
        temporary=root/'pipeline_status.json.tmp';temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2))
        temporary.replace(root/'pipeline_status.json')
        lines=['# 五项Memory验证：运行状态','',f'更新时间：{state["updated_at"]}。仅按实际完成标记计数。','',
            '本轮保留原数据划分、共同教师、Patch8和λ=.01；扩大版独立校准、参数搜索与微调队列已暂缓。',
            f'重构配置：{state["reconstruction_completed"]}/{state["reconstruction_requested"]}完成；关系头：{relations}/{state["relation_heads_requested"]}完成。',
            '模型一审计：'+('完成。' if state['model_one_analysis_complete'] else '等待全部重构与评分完成。'),
            '最终报告：'+('完成。' if finished else '待全部三seed关系验证和审计完成后更新。'),'',
            '| 单元 | 重构配置完成 | 最终A4已评分 | 关系头完成 |','| --- | --- | --- | --- |']
        for r in counts:
            lines.append(f'| {r["unit"]} | {len(r["completed_reconstruction_arms"])} / {len(arms)} | {r["final_checkpoint_scored"]} | {len(r["completed_relation_heads"])} / 11 |')
        if 'model_two_execution' in state:
            lines+=['','模型二已迁移到Kaggle四任务；本表关系头数量仅统计已回收并导入本地的完成标记，不能据此推断Kaggle覆盖或完成数。']
            for name,task in state.get('kaggle',{}).get('tasks',{}).items():
                lines.append(f'- Kaggle {name}：{task.get("audit_status",task.get("status"))}；{task.get("kernel", "等待父模型")}。')
        if active:
            lines+=['','当前训练或可恢复的任务：']
            for a in active:lines.append(f'- {a["unit"]}/{a["arm"]}：最近epoch {a["epoch"]}，正常验证Huber {a["validation"]}。')
        lines+=['','[五项执行协议](../../system_design/MSCM_MCR_PRIORITY5_20261025.md)。',
            '模型一日志：`/tmp/mscm_priority5_20261025_v1.log`；模型二日志：`/tmp/mcr_priority5_20261025_v1.log`；审计日志：`/tmp/analyze_priority5_20261025_v1.log`。',
            '本文件是运行记录，不替代或提前填写论文最终实验矩阵。','']
        report=Path('report/analysis/rcr/MSCM_MCR_PRIORITY5_STATUS_20261025.md')
        temporary=report.with_suffix('.md.tmp');temporary.write_text('\n'.join(lines));temporary.replace(report)
        if finished:break
        time.sleep(30)


if __name__=='__main__':main()
