# 电池异常检测预实验：执行与验收状态

更新时间：2026-09-13。此文件记录 `configs/research/README.md` 的实际调度和证据状态。正式 CH 协议固定 `split_seed=3407`；`model_seed` 只控制模型随机性。

## 已完成并已验收

| 工作包 | 证据位置 | 完整性 |
|---|---|---:|
| P1 + P2 第一轮 | `runs/research/original_first_round/` | 32/32 |
| P3/4 + P5 第一轮 | `runs/research/original_first_round/` | 31/31 |
| P6 Temporal：MTAD-GAT | `runs/internal/research_06_ch_batterygen_temporal_scale/lfp/FORMAL_TEMPORAL_27_CELL_AUDIT.json` | 27/27 |
| P6 Temporal：DCdetector | `runs/internal/research_06_ch_batterygen_temporal_scale/lfp/FORMAL_TEMPORAL_DCDETECTOR_27_CELL_AUDIT.json` | 27/27 |
| P6 Temporal：PatchAD | `runs/internal/research_06_ch_batterygen_temporal_scale/lfp/FORMAL_TEMPORAL_PATCHAD_27_CELL_AUDIT.json` | 27/27 |
| P6 Temporal：TranAD A | `runs/internal/research_06_ch_batterygen_temporal_scale/lfp/FORMAL_TEMPORAL_TRANAD_A_15_CELL_AUDIT.json` | 15/15 |
| P8 外部确认 | Brand2/Brand3 五折报告 | 已验收 |
| P10 正常稳定性 | Isolation Forest、MTAD-GAT 结果 | 已完成 |

两份第一轮 Kaggle zip 已经解压并审计；每个单元都保留逐窗口 `scores.npz`、训练/验证正常分数、样本映射、配置和指标。总审计见 `runs/research/FIRST_ROUND_ORIGINAL_ZIP_AUDIT.json`。这些结果可作为 P7 的单 seed 问题筛选输入，不能和三 seed 确认阶段混合统计。

## 当前收尾

| 工作包 | Kaggle 账号与数据集 | Kaggle 状态 | 本地状态 | 下一动作 |
|---|---|---|---|---|
| P6 Temporal：TranAD B，4 设置 × 3 seeds | `chenmjdaisy/ch-batterygen-lfp-formal-20260912` | 用户确认 COMPLETE | 尚未取得本地 artifact | 下载紧凑 `artifacts/research06_ch_temporal_tranad_part_b/`，与 A 合并后运行 27-cell verifier |
| P7 raw-score / 阈值 / 形态分析 | 本地 CH V1.0 + 已验收 artifacts | COMPLETE | 153 个运行单元已完成分析 | `runs/research/P7_RAW_SCORE_ANALYSIS/` |
| P3 Condition confirmation：LFP | `chenmjdaisy/ch-batterygen-v107-raw` | RUNNING：`research11-condition-lfp-three-seed` | 待下载 | 36 个单元，固定 split、三 model seeds |
| P3 Condition confirmation：NCM | `chenmjdaisy/ch-batterygen-v107-raw` | RUNNING：`research11-condition-ncm-three-seed` | 待下载 | 36 个单元，固定 split、三 model seeds |

P7 会输出 `max` / `mean` / `top-5%` 片段聚合，以及训练正常、验证正常的 P95/P99 阈值敏感性；故障标签只在测试后的分层报告中使用。P6 的紧凑 exports 若未保留 `config.json`，仍以保存的样本映射和可复算 top-5% 分数建立序列映射；映射不能一一验证时分析会失败，不产生结论。

## 数据集归属

| 账号 | 仅使用其账号名下的数据集 |
|---|---|
| `daisychen2` | `daisychen2/ch-batterygen-v107-raw`、`daisychen2/ch-batterygen-lfp-processed` |
| `chenmjdaisy` | `chenmjdaisy/ch-batterygen-v107-raw`、`chenmjdaisy/ch-batterygen-lfp-formal-20260912`、`chenmjdaisy/tsinghua-ev` |

后续 P3 三 seed 确认、新模型和外部全面比较均须等待 P7 从第一轮中定义并冻结 1–2 个 failure mode 后再调度。
