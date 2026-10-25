"""Maintain an explicit local status document while the long CUDA queue runs."""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def last_epoch(path):
    if not path.exists():return None
    with path.open() as f:rows=list(csv.DictReader(f))
    return rows[-1] if rows else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',default='runs/internal/mscm_mcr_development_20261025_v2')
    parser.add_argument('--relations',default='runs/internal/mcr_gat_development_20261025_v1')
    args=parser.parse_args()
    core,relations=Path(args.core),Path(args.relations)
    config=json.loads((relations/'protocol_manifest.json').read_text())
    units,heads=config['units'],list(config['heads'])
    target=Path('report/analysis/rcr/MSCM_MCR_STATUS_20261025.md')
    while True:
        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
        completed=[];running=[]
        for unit in units:
            done=[]
            for name in heads:
                directory=relations/unit/name
                if (directory/'complete.json').exists():done.append(name)
                elif (directory/'initial.json').exists():
                    row=last_epoch(directory/'history.csv')
                    running.append(dict(unit=unit,head=name,last_finished_epoch=row))
            completed.append(dict(unit=unit,completed_heads=done,
                ranking_metrics_complete=(relations/unit/'complete.json').exists()))
        number=sum(len(r['completed_heads']) for r in completed)
        finished=(relations/'analysis_complete.json').exists()
        state=dict(updated_at=timestamp,model_one_analysis_complete=(core/'model_one_analysis_complete.json').exists(),
            model_one_core_units=35,model_two_heads_completed=number,model_two_heads_requested=len(units)*len(heads),
            model_two_analysis_complete=finished,units=completed,active_or_resumable_heads=running)
        temporary=relations/'pipeline_status.json.tmp'
        temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2));temporary.replace(relations/'pipeline_status.json')
        lines=['# MSCM-Net / MCR-GAT 本地运行状态','',f'更新时间：{timestamp}。本文件每30秒更新，完成计数仅来自各任务自己写出的完成标记。','',
            '模型一：35个核心重构单元及全量评估已完成。A3合并到A1，A5复用A4未校准评分。',
            f'模型二：{number}/{len(units)*len(heads)}个关系头完成训练、全量评分和掩码审计；整体统计报告'+('已完成。' if finished else '等待全部单元完成后自动生成。'),'',
            '已完成单元：']
        for row in completed:
            lines.append(f'- {row["unit"]}：{len(row["completed_heads"])}/{len(heads)}个头；整组指标'+('已完成。' if row['ranking_metrics_complete'] else '待本单元全部关系头完成。'))
        if running:
            lines+=['','当前训练或可断点恢复的头：']
            for row in running:
                epoch=row['last_finished_epoch']
                if epoch:
                    lines.append(f'- {row["unit"]}/{row["head"]}：最近完成epoch {epoch["epoch"]}，正常验证LOO Huber {float(epoch["normal_validation_all_loo_huber"]):.6g}。')
                else:lines.append(f'- {row["unit"]}/{row["head"]}：已初始化，首个epoch尚未完成。')
        lines+=['','固定配置：重构阶段5+5个epoch，关系头最多10个epoch，正常验证选模型，FP32；全量训练/评分，测试标签不参与选择。',
            '关系条件为正常训练Memory占用的控制上下文先验投影；完整多变量Memory条件、后续融合/λ/Patch/编码器微调消融尚未运行。',
            '模型一报告：`report/analysis/rcr/MSCM_MODEL_ONE_CORE_20261025.md`。',
            '模型二完整结果自动写入：`report/analysis/rcr/MCR_GAT_MODEL_TWO_20261025.md`。',
            'CUDA日志：`/tmp/mcr_gat_development_20261025_v1.log`；自动分析日志：`/tmp/mcr_gat_analysis_20261025_v1.log`。',
            '这是独立开发进度记录，不替代或改写论文最终四张实验矩阵。','']
        tmp=target.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines));tmp.replace(target)
        if finished:break
        time.sleep(30)


if __name__=='__main__':main()
