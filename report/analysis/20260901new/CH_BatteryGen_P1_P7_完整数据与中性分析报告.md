# CH-BatteryGen P1–P7：完整数据登记与中性分析

生成日期：2026-09-19。此报告登记已经落盘并可追溯的结果，不预设后续模型方向，也不把筛选实验写成论文结论。

## 使用规则

- 训练只使用正常样本；故障标签只用于报告排序、召回与分层结果。
- 除另有说明，主视图为 `top5pct` 聚合、训练正常样本 P99 阈值。
- AUPRC/AUROC 衡量排序；FPR/Recall 依赖阈值。两类指标不可互相替代。
- P1 有三类产物：早期第一轮筛选、后续固定划分正式基线，以及后补的 DyAD-style CH adaptation。三者均保留，但**不合并平均，也不互相替代**。DyAD-style CH 是按 CH 七变量和 CURRENT/SOC 条件变量适配的 DynamicVAE，不是 NC 专用的原始 DyAD。
- 每个 P1–P5 阶段均保留第一轮与后续运行产物；后续运行的模型、种子和设置覆盖并不完全相同，不能直接与第一轮拼接平均。P6 包含三 seed 时间尺度实验；P7 是对已导入 raw score 的统一后处理，不是新的训练实验。

### 明细表排版规则

- 每个阶段先给“重点摘要”；其后的完整明细已拆为 **模型 → seed** 的小表。同一 seed 下不同数据集/设置同行，便于横向核对。
- 第一轮筛选表未在原始字段中显式列出 seed，因此仅按模型拆表；不在报告中臆补 seed。
- P7 是覆盖审计表，不含模型/seed 指标行，仍按实验阶段分组展示。

## 表格目录与覆盖范围

| table | rows | metric protocol | source |
| --- | --- | --- | --- |
| P1 formal fixed-split baseline | 26 | top5pct + train-normal P99 | runs/internal/research_02_ch_batterygen_indomain_models |
| P1 DyAD-style CH adaptation | 6 | top5pct + train-normal P99; fixed split | runs/internal/research_12_dyad_style_ch_p1 |
| P2 R11 Kaggle completed artifacts | 67 | top5pct + train-normal P99 | runs/kaggle_downloads/research11_condition_* |
| P1--P5 first-round screening | 63 | top5pct + train-normal P99 | runs/research/original_first_round (via P7) |
| P2--P5 later-run artifacts | 96 | top5pct + train-normal P99 | local/Kaggle later run families; coverage differs by phase |
| C3/C4 CH diagnostic rerun | 30 | top5pct + train-normal P99; seed 3407 | 3 local cells + runs/internal/run_ch_c3_c4_remaining.zip (27 Kaggle cells) |
| P6 artifact-complete temporal cells | 108 | artifact audit: AUROC/AUPRC/raw-score count | four FORMAL_TEMPORAL audits |
| P6 recovered primary-score rows | 105 | top5pct + train-normal P99 | internal and Kaggle artifact CSV files |
| P7 aggregation/threshold database | 1836 | 3 aggregations × 2 normal calibration sets × P95/P99 | runs/research/P7_RAW_SCORE_ANALYSIS |
| P7 fault-type database | 5508 | same P7 grid × fault type | runs/research/P7_RAW_SCORE_ANALYSIS |

完整表格均以 CSV 保存，便于筛选和复核：

- [P1 正式固定划分主指标](ch_batterygen_p1_p7_data/P1_formal_fixed_split_primary.csv)
- [P1 DyAD-style CH 主指标（6 个正式单元）](ch_batterygen_p1_p7_data/P1_dyad_style_ch_fixed_split_primary.csv)
- [P1 DyAD-style CH 全部聚合×阈值行](ch_batterygen_p1_p7_data/P1_dyad_style_ch_all_aggregation_threshold_rows.csv)
- [P1 早期第一轮域内筛选逐单元表](ch_batterygen_p1_p7_data/P1_indomain_screening_primary_rows.csv)
- [P1–P6、P7 已导入运行的主指标总表](ch_batterygen_p1_p7_data/P1_P6_P7_source_primary_rows.csv)
- [P2 第一轮工况筛选逐单元表](ch_batterygen_p1_p7_data/P2_condition_primary_rows.csv)
- [P2 后续本地重跑逐单元表](ch_batterygen_p1_p7_data/P2_later_local_primary_rows.csv)
- [P2 后续 Kaggle LFP 重跑表](ch_batterygen_p1_p7_data/P2_later_kaggle_lfp_primary_rows.csv)
- [P2 后续 Kaggle NCM 重跑表](ch_batterygen_p1_p7_data/P2_later_kaggle_ncm_primary_rows.csv)
- [P2 R11 Kaggle 已完成单元表](ch_batterygen_p1_p7_data/P2_research11_kaggle_completed_primary_rows.csv)
- [P3/P4 第一轮化学体系筛选逐单元表](ch_batterygen_p1_p7_data/P3_P4_chemistry_primary_rows.csv)
- [P3/P4 后续本地重跑逐单元表](ch_batterygen_p1_p7_data/P3_P4_later_local_primary_rows.csv)
- [P5 第一轮输入变量筛选逐单元表](ch_batterygen_p1_p7_data/P5_schema_primary_rows.csv)
- [P5 后续本地重跑逐单元表](ch_batterygen_p1_p7_data/P5_later_local_primary_rows.csv)
- [C3/C4 Kaggle 补跑归档（27 个单元）](../../../runs/internal/run_ch_c3_c4_remaining.zip)
- [P6 全部 108 个 artifact 审计单元](ch_batterygen_p1_p7_data/P6_all_108_artifact_audit_rows.csv)
- [P6 已恢复的主指标行](ch_batterygen_p1_p7_data/P6_recovered_primary_rows.csv)
- [P7 全部聚合×阈值表（1836 行）](ch_batterygen_p1_p7_data/P7_all_aggregation_threshold_rows.csv)
- [P7 全部故障类型分层表（5508 行）](ch_batterygen_p1_p7_data/P7_all_fault_type_rows.csv)
- [P2/P3-P4/P5 四种阈值控制完整表（220 行）](ch_batterygen_p1_p7_data/P2_P5_static_pot_primary_rows.csv)
- [IF 的 FP vs TN 状态归因表](ch_batterygen_if_fp_attribution_ncm_to_lfp/fp_vs_tn_attribution.csv)
- [四模型 FP/TN、FN/TP 归因明细](ch_batterygen_threshold_error_attribution/all_stratified_error_attribution.csv)
- [四模型归因汇总](ch_batterygen_threshold_error_attribution/model_phase_error_attribution_summary.csv)
- [后续反事实失效假设与验证计划](电池TSAD_反事实失效假设与验证计划.md)

## 一页总览：按研究问题看重点

| 研究问题 | 当前最清楚的发现 | 必须同时看的约束 |
| --- | --- | --- |
| P1 域内基线 | LFP：TranAD 平均 AUPRC **0.868**；NCM：MTAD-GAT **0.777** | Isolation Forest 的高 LFP 召回伴随 **5.9% FPR** |
| P2 工况变化 | `charge→discharge` 的源域 P99 FPR 可达 **0.91–1.00** | 重标定可压低 FPR，但召回同步明显下降 |
| P3/P4 体系迁移 | `mixed→LFP` 的 TranAD AUPRC **0.824**、FPR 0.015 | `NCM→LFP` 存在 FPR=**1.000** 的阈值失配 |
| P5 输入变量 | 变量重要性随模型而变，不存在单一“必需变量组” | 目前是单 seed、单体系筛选 |
| P6 时间尺度 | MTAD-GAT 已完成配置的 AUPRC 在 0.850–0.864 | TranAD 的 lookback 配置生效性仍待核验 |
| P7 后处理 | 覆盖 153 个运行、1,836 条阈值/聚合记录 | 不能把不同阶段、不同运行直接混合平均 |
| 阈值控制 | 源 P99、验证 P99、目标域正常 P99、POT 均已对 P2/P3-P4/P5 复算 | 先排除阈值混杂，再讨论模型表征问题 |
| 错误归因 | 四模型均已完成 FP/TN、FN/TP 分层；MTAD-GAT/PatchAD/TranAD 的 FP 多与高动态正常段、FN 多与弱动态故障有关 | 这是 score 与原始状态的统计关联，尚不是内部机制证据 |

**本表结论：**先按研究问题读，再进入对应的“摘要表 → 完整明细表”。**排序（AUPRC）与阈值风险（FPR/Recall）必须成对阅读。**

## C3/C4：CH 诊断补跑（seed 3407；不并入基线平均）

本节补入 **30 个** C3/C4 单元：P1 的 3 个本地已完成单元，加上 Kaggle 的 27 个剩余单元。Kaggle 压缩产物 `runs/internal/run_ch_c3_c4_remaining.zip` 中的 `task_status.json` 记录 **27/27 complete**；每项均核验到 `metrics.json`、训练/验证/测试分数与样本映射。C3 是 prototype-query 工况条件化 MTAD-GAT；C4 是物理图偏置加控制条件响应一致性分支，最终分数固定融合 5% 一致性残差。

表中 AP/AUROC 是阈值无关的排序指标；F1/FPR 是训练正常 P99 下的部署式读数。跨工况/跨体系的 FPR 很高时，不能把 F1 当作模型优劣结论；POT、目标域正常重标定仍应另行对照。

### 重点摘要：原始 MTAD-GAT vs C3/C4

| 条件 | 原始 AP | C3 AP | C4 AP | 最重要的证据 |
| --- | ---: | ---: | ---: | --- |
| P1 LFP 放电 | 0.8603 | 0.8566 | **0.8606** | 基本无改善 |
| P1 NCM 放电 | 0.7869 | 0.7866 | **0.7915** | C4 小幅改善；F1 0.3259、FPR 0.0150 |
| P2 LFP 充电域内 | 0.6412 | **0.6430** | 0.6222 | C4 排序退化 |
| P2 LFP 充→放 | **0.7566** | 0.7350 | 0.7526 | C4 优于 C3，但未超基线；FPR 仍 0.8925 |
| P2 LFP 放→充 | 0.6532 | **0.6784** | 0.6566 | C3 有条件性收益 |
| P2 NCM 充电域内 | 0.8692 | 0.8720 | **0.8775** | C4 稳定小幅改善 |
| P2 NCM 充→放 | 0.6159 | 0.6199 | **0.6252** | C4 小幅改善，但 FPR 0.9463 |
| P2 NCM 放→充 | 0.7311 | **0.6118** | **0.7639** | **C3 失效；C4 AP +0.0327、FPR 0.1761→0.0437** |
| P3/P4 LFP→NCM | 0.5650 | 0.5564 | 0.5641 | 排序近基线，但 FPR=1.000 |
| P3/P4 NCM→LFP | 0.7680 | 0.7845 | **0.8134** | C4 AP +0.0454，但 FPR=1.000 |
| P3/P4 混合→LFP | **0.8134** | 0.8079 | 0.8055 | 两者均略退化 |
| P3/P4 混合→NCM | 0.6521 | 0.6596 | **0.6649** | 排序略升，但 P99 Recall 仅 0.012 |

**本表结论：**C3 **不是稳定改进**：它在 LFP 放→充上提升 AP，但在 NCM 放→充将 AP 从 0.7311 降到 **0.6118**，且 FPR 从 0.1761 升到 **0.3761**，符合“把部分故障状态化解释”的风险。C4 的正向证据主要集中在 NCM，尤其 NCM 放→充；但 LFP、混合训练与跨化学体系上并不一致，不能写成普适优势。

### P5：C3 输入变量诊断（C4 仅完整七通道适用）

| 输入设置 | C3 AP | C3 F1 | C3 FPR | 解释 |
| --- | ---: | ---: | ---: | --- |
| full（7 通道） | 0.8581 | 0.3642 | 0.0075 | 参照 |
| controls-only（CURRENT、SOC） | 0.6446 | 0.0158 | 0.0038 | 排序信息不足 |
| responses-only | **0.8648** | **0.4000** | 0.0100 | 响应变量已承载主要故障证据 |
| 去温度极值 | 0.8468 | 0.3672 | 0.0063 | 仅小幅退化 |
| 去电压极值 | 0.7283 | 0.0894 | 0.0050 | **显著退化：电压极值是关键证据** |

完整七通道上，C4 的 AP=**0.8598**、F1=0.3666、FPR=0.0088，仅比 C3 小幅增加 0.0018 AP。C4 不在 `controls-only`、`responses-only`、去温度/去电压极值设置上硬跑：该物理图需要完整且有序的七通道语义，缺失通道的结果应标为**不适用**，而不是“无结果”。

**本表结论：**P5 支持一个更具体、但仍限于 MTAD-GAT/C3 和单 seed 的判断：故障排序主要由**响应变量，特别是电压极值**承载；CURRENT/SOC 更适合充当条件变量。它不推出“所有模型都应删去控制变量”。

## P1：域内基线

### 重点摘要：固定划分基线（按方法分组的三 seed 均值）

| chemistry | method | AUPRC | Recall | FPR | 重点 |
| --- | --- | ---: | ---: | ---: | --- |
| LFP | Isolation Forest | 0.834 | **0.305** | **0.059** | 召回最高，但误报偏高 |
| LFP | MTAD-GAT | 0.858 | 0.229 | 0.010 | 均衡 |
| LFP | TranAD | **0.868** | 0.238 | 0.010 | LFP 排序最佳 |
| NCM | MTAD-GAT | **0.777** | 0.145 | 0.010 | NCM 排序最佳 |
| NCM | TranAD | 0.761 | **0.150** | 0.008 | NCM 低 FPR 下召回最高 |

**本表结论：**可比固定划分中，**LFP 优先看 TranAD（AUPRC 0.868）**，**NCM 优先看 MTAD-GAT（0.777）**；不能只因 Isolation Forest 的 LFP 召回 0.305 就判为更好，因为其 FPR 达 **0.059**。

![P1 固定划分 AUPRC 折线图](figures/ch_batterygen_p1_fixed_split_auprc.svg)

图 1. 同一方法在 LFP/NCM 上的三 seed 平均 AUPRC。折线是化学体系对比，**不是时间趋势**。

### 第一轮（8 个单元）

<details>
<summary>展开逐模型 / seed 完整明细</summary>

#### 明细：dcdetector

| dataset | method | sample_auprc | recall | fpr | setting |
| --- | --- | --- | --- | --- | --- |
| CH_LFP_DISCHARGE | dcdetector | 0.7362 | 0.0590 | 0.0075 | p1_lfp_discharge |
| CH_NCM_DISCHARGE | dcdetector | 0.6648 | 0.0930 | 0.0100 | p1_ncm_discharge |

#### 明细：mtad_gat

| dataset | method | sample_auprc | recall | fpr | setting |
| --- | --- | --- | --- | --- | --- |
| CH_LFP_DISCHARGE | mtad_gat | 0.8603 | 0.2270 | 0.0088 | p1_lfp_discharge |
| CH_NCM_DISCHARGE | mtad_gat | 0.7869 | 0.1770 | 0.0150 | p1_ncm_discharge |

#### 明细：patchad

| dataset | method | sample_auprc | recall | fpr | setting |
| --- | --- | --- | --- | --- | --- |
| CH_LFP_DISCHARGE | patchad | 0.7446 | 0.0700 | 0.0138 | p1_lfp_discharge |
| CH_NCM_DISCHARGE | patchad | 0.6054 | 0.0280 | 0.0088 | p1_ncm_discharge |

#### 明细：tranad

| dataset | method | sample_auprc | recall | fpr | setting |
| --- | --- | --- | --- | --- | --- |
| CH_LFP_DISCHARGE | tranad | 0.8443 | 0.1780 | 0.0075 | p1_lfp_discharge |
| CH_NCM_DISCHARGE | tranad | 0.7646 | 0.1880 | 0.0088 | p1_ncm_discharge |

**本表结论：**第一轮中，LFP 的 MTAD-GAT AUPRC **0.8603**、NCM 的 MTAD-GAT **0.7869** 均较高；但它是早期筛选，**不与下方固定划分三 seed 表合并平均**。

</details>
### 后续固定划分重跑（26 个单元）

<details>
<summary>展开逐模型 / seed 完整明细</summary>

#### 明细：dcdetector（seed 3407.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | dcdetector | 3407.0000 | 0.6900 | 0.0000 | 0.0100 |
| NCM | dcdetector | 3407.0000 | 0.6304 | 0.0160 | 0.0088 |

#### 明细：dcdetector（seed 3408.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | dcdetector | 3408.0000 | 0.6088 | 0.0060 | 0.0063 |
| NCM | dcdetector | 3408.0000 | 0.6063 | 0.0090 | 0.0088 |

#### 明细：dcdetector（seed 3409.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | dcdetector | 3409.0000 | 0.7077 | 0.0320 | 0.0187 |
| NCM | dcdetector | 3409.0000 | 0.6128 | 0.0250 | 0.0063 |

#### 明细：isolation_forest（seed 3407.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | isolation_forest | 3407.0000 | 0.8344 | 0.3050 | 0.0587 |
| NCM | isolation_forest | 3407.0000 | 0.5545 | 0.0190 | 0.0312 |

#### 明细：mtad_gat（seed 3407.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | mtad_gat | 3407.0000 | 0.8591 | 0.2280 | 0.0075 |
| NCM | mtad_gat | 3407.0000 | 0.7871 | 0.1640 | 0.0125 |

#### 明细：mtad_gat（seed 3408.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | mtad_gat | 3408.0000 | 0.8665 | 0.2280 | 0.0100 |
| NCM | mtad_gat | 3408.0000 | 0.7837 | 0.1740 | 0.0088 |

#### 明细：mtad_gat（seed 3409.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | mtad_gat | 3409.0000 | 0.8482 | 0.2320 | 0.0138 |
| NCM | mtad_gat | 3409.0000 | 0.7609 | 0.0980 | 0.0088 |

#### 明细：patchad（seed 3407.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | patchad | 3407.0000 | 0.7181 | 0.0490 | 0.0125 |
| NCM | patchad | 3407.0000 | 0.5474 | 0.0110 | 0.0100 |

#### 明细：patchad（seed 3408.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | patchad | 3408.0000 | 0.6359 | 0.0120 | 0.0037 |
| NCM | patchad | 3408.0000 | 0.5359 | 0.0090 | 0.0100 |

#### 明细：patchad（seed 3409.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | patchad | 3409.0000 | 0.6768 | 0.0360 | 0.0050 |
| NCM | patchad | 3409.0000 | 0.5749 | 0.0110 | 0.0075 |

#### 明细：tranad（seed 3407.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | tranad | 3407.0000 | 0.8739 | 0.2160 | 0.0088 |
| NCM | tranad | 3407.0000 | 0.7732 | 0.1780 | 0.0075 |

#### 明细：tranad（seed 3408.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | tranad | 3408.0000 | 0.8838 | 0.2300 | 0.0050 |
| NCM | tranad | 3408.0000 | 0.7643 | 0.1300 | 0.0063 |

#### 明细：tranad（seed 3409.0000）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | tranad | 3409.0000 | 0.8469 | 0.2690 | 0.0163 |
| NCM | tranad | 3409.0000 | 0.7466 | 0.1430 | 0.0100 |

**本表结论：**三 seed 明细支持摘要表的排序：LFP 的最高单元为 TranAD **0.8838**，NCM 的最高单元为 MTAD-GAT **0.7871**。种子间仍有波动，因此正文以方法均值而非单个最高点作主判断。

</details>
### DyAD-style CH adaptation（6 个正式单元）

| chemistry | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| LFP | DyAD-style CH adaptation | 3407.0000 | 0.8463 | 0.2440 | 0.0050 |
| LFP | DyAD-style CH adaptation | 3408.0000 | 0.8452 | 0.2430 | 0.0037 |
| LFP | DyAD-style CH adaptation | 3409.0000 | 0.8266 | 0.2480 | 0.0075 |
| NCM | DyAD-style CH adaptation | 3407.0000 | 0.7667 | 0.0320 | 0.0088 |
| NCM | DyAD-style CH adaptation | 3408.0000 | 0.7598 | 0.0410 | 0.0088 |
| NCM | DyAD-style CH adaptation | 3409.0000 | 0.7889 | 0.0340 | 0.0088 |

三组结果独立展示，不作合并平均。DyAD-style CH 的训练使用正常 VIN、固定 `split_seed=3407`，三个 `model_seed` 仅改变训练随机性；每个单元保留 raw score 与聚合×阈值敏感性结果。

**本表结论：**DyAD-style CH 在 LFP 的 FPR 为 **0.0037–0.0075**、Recall 为 **0.243–0.248**；在 NCM 虽 AUPRC 可达 0.7889，但 Recall 仅 0.032–0.041。它应作为独立适配对照，不替换基线排名。

## P2：工况变化

### 重点摘要：同工况基线 → 跨工况迁移（第一轮、同一模型/seed）

| direction | chemistry | method | 同工况 AUPRC（基线） | 跨工况 AUPRC | ΔAUPRC | FPR：基线 → 迁移 | POT：FPR / R / F1 | 目标域 P99：FPR / R / F1 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| charge → discharge | LFP | MTAD-GAT | 0.641 | 0.757 | +0.116 | 0.015 → **0.909** | 0.895 / 0.978 / 0.837 | **0.022 / 0.103 / 0.185** |
| charge → discharge | LFP | TranAD | 0.714 | 0.685 | −0.029 | 0.006 → **0.993** | 0.990 / 1.000 / 0.835 | **0.020 / 0.052 / 0.098** |
| charge → discharge | NCM | MTAD-GAT | **0.869** | 0.616 | **−0.253** | 0.016 → **0.938** | 0.930 / 0.915 / 0.800 | **0.018 / 0.040 / 0.076** |
| charge → discharge | NCM | TranAD | 0.815 | 0.501 | **−0.314** | 0.009 → **0.998** | 1.000 / 0.997 / 0.832 | **0.002 / 0.016 / 0.031** |
| discharge → charge | LFP | MTAD-GAT | **0.860** | 0.653 | **−0.207** | 0.009 → 0.075 | 0.095 / 0.130 / 0.222 | **0.020 / 0.073 / 0.134** |
| discharge → charge | LFP | TranAD | 0.844 | 0.648 | **−0.196** | 0.008 → 0.135 | 0.135 / 0.211 / 0.334 | **0.018 / 0.065 / 0.122** |
| discharge → charge | NCM | MTAD-GAT | 0.787 | 0.731 | −0.056 | 0.015 → 0.176 | 0.141 / 0.373 / 0.522 | **0.014 / 0.224 / 0.365** |
| discharge → charge | NCM | TranAD | 0.765 | 0.784 | +0.019 | 0.009 → 0.234 | 0.212 / 0.466 / 0.600 | **0.014 / 0.283 / 0.439** |

新增两列中的 `FPR / R / F1` 均在**同一 VIN 隔离目标域评估集**上计算；因此应横向比较 POT 与目标域 P99，而不要把它与前列“全目标测试集”的原始 FPR 做逐位相减。

**本表结论：**这里的“同工况”就是迁移基线。最严重的是 NCM `charge→discharge`：MTAD-GAT/TranAD 的排序分别下降 **0.253/0.314**，且 FPR 接近 1。AUPRC 的下降不由阈值决定；但 FPR 是否为模型问题，必须以下面的统一阈值对照再判断。LFP `charge→discharge` 的 MTAD-GAT AUPRC 虽上升，但 FPR=**0.909**，同样不能视为迁移成功。

### 阈值控制对照（P2 全部 24 个已保存运行；不重训）

每个运行固定同一份 top-5% sample score 与 VIN 隔离的目标域测试集；AP/AUROC 因此在四种阈值间不变。目标域正常 P99 用 50% 目标域正常 VIN 校准，其余正常 VIN 与全部故障用于评估。表中是运行均值，**只用来排除阈值混杂，不能替代逐方向比较**。

| 阈值策略 | AUPRC（固定） | FPR | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 源训练正常 P99 | 0.763 | 0.329 | 0.687 | 0.380 | 0.384 |
| 源验证正常 P99 | 0.763 | 0.338 | 0.699 | 0.390 | 0.391 |
| 静态 POT（源训练正常尾部；目标 FPR=1%） | 0.763 | 0.327 | 0.695 | 0.379 | 0.384 |
| **目标域正常 P99（VIN 隔离）** | 0.763 | **0.017** | **0.784** | 0.072 | 0.125 |

**本表结论：**POT 在当前离线 sample-score 协议下与源 P99 几乎等价，未消除跨工况高 FPR；目标域正常校准把 FPR 拉回约 1–2%，却也把 Recall 从 0.380 降至 0.072。故 P2 剩余的研究问题不是“P99 取错了”，而是**目标域正常与故障分数有重叠：低误报与高召回不能同时保持**。完整 96 行见 [阈值控制 CSV](ch_batterygen_p1_p7_data/P2_P5_static_pot_primary_rows.csv)。

**读表澄清（避免把两个基线混为一个方法）：**AUPRC 只由同一批 score 的排序决定，换任何阈值都**不会变**。本报告的“静态 POT”是仅在**源训练正常分数**的 P98 以上拟合 GPD、目标 FPR=1% 的阈值；它没有看目标域正常数据，因此在这里得到的阈值恰好接近源 P99，仍会把大量目标域正常段判异常。F1 因此可能较高：阈值较低时 Recall 近 1，而故障样本比例高、Precision 尚可，二者的调和平均也会高；这不表示低误报部署更好。**目标域正常 P99 没有再做 POT**：它是另一条、使用 50% 无标签目标域正常 VIN 的校准基线；阈值升高后 FPR 降至约 1%，但原先被较低阈值“捞到”的大量故障也变为 FN，故 F1 下降。若要检验“目标域 POT”是否优于目标域 P99，必须另行定义可用的连续无标签目标流；当前离线、VIN 隔离协议不能把这一步伪装成在线 POT。

### 第一轮：charge→discharge 的配对展示（8 行）

<details>
<summary>展开逐模型 / seed 完整明细</summary>

#### 明细：dcdetector

| chemistry | method | same_fpr | cross_fpr | AUPRC_same_to_cross |
| --- | --- | --- | --- | --- |
| LFP | dcdetector | 0.0100 | 0.0000 | 0.474 → 0.475 |
| NCM | dcdetector | 0.0127 | 0.6525 | 0.586 → 0.538 |

#### 明细：mtad_gat

| chemistry | method | same_fpr | cross_fpr | AUPRC_same_to_cross |
| --- | --- | --- | --- | --- |
| LFP | mtad_gat | 0.0150 | 0.9087 | 0.641 → 0.757 |
| NCM | mtad_gat | 0.0155 | 0.9375 | 0.869 → 0.616 |

#### 明细：patchad

| chemistry | method | same_fpr | cross_fpr | AUPRC_same_to_cross |
| --- | --- | --- | --- | --- |
| LFP | patchad | 0.0075 | 0.8363 | 0.718 → 0.509 |
| NCM | patchad | 0.0197 | 0.9325 | 0.530 → 0.564 |

#### 明细：tranad

| chemistry | method | same_fpr | cross_fpr | AUPRC_same_to_cross |
| --- | --- | --- | --- | --- |
| LFP | tranad | 0.0063 | 0.9925 | 0.714 → 0.685 |
| NCM | tranad | 0.0085 | 0.9975 | 0.815 → 0.501 |

**本表结论：**该方向中，除 DCdetector 外，跨工况 FPR 均很高（**0.8363–0.9975**）；同时 AUPRC 有升有降，说明 FPR 灾难不能单用排序变化解释。

</details>
### 第一轮：discharge→charge 的配对展示（8 行）

<details>
<summary>展开逐模型 / seed 完整明细</summary>

#### 明细：dcdetector

| chemistry | method | same_fpr | cross_fpr | AUPRC_same_to_cross |
| --- | --- | --- | --- | --- |
| LFP | dcdetector | 0.0075 | 0.8087 | 0.736 → 0.716 |
| NCM | dcdetector | 0.0100 | 0.0028 | 0.665 → 0.681 |

#### 明细：mtad_gat

| chemistry | method | same_fpr | cross_fpr | AUPRC_same_to_cross |
| --- | --- | --- | --- | --- |
| LFP | mtad_gat | 0.0088 | 0.0750 | 0.860 → 0.653 |
| NCM | mtad_gat | 0.0150 | 0.1761 | 0.787 → 0.731 |

#### 明细：patchad

| chemistry | method | same_fpr | cross_fpr | AUPRC_same_to_cross |
| --- | --- | --- | --- | --- |
| LFP | patchad | 0.0138 | 0.1300 | 0.745 → 0.480 |
| NCM | patchad | 0.0088 | 0.0000 | 0.605 → 0.552 |

#### 明细：tranad

| chemistry | method | same_fpr | cross_fpr | AUPRC_same_to_cross |
| --- | --- | --- | --- | --- |
| LFP | tranad | 0.0075 | 0.1350 | 0.844 → 0.648 |
| NCM | tranad | 0.0088 | 0.2338 | 0.765 → 0.784 |

两个方向分别展示，不混成一个均值。

**本表结论：**`discharge→charge` 的 FPR 总体低于反向迁移，但 LFP DCdetector 仍为 **0.8087**，NCM TranAD 为 0.2338；方向不对称必须保留，不能两方向求平均。

</details>
### 后续重跑（30 个单元）

<details>
<summary>展开逐模型 / seed 完整明细</summary>

#### 明细：dcdetector（seed 3407.0000）

| later_run_family | dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2_later_local | CH_LFP | charge_to_charge | dcdetector | 3407.0000 | 0.4825 | 0.0000 | 0.0138 |
| P2_later_local | CH_LFP | charge_to_discharge | dcdetector | 3407.0000 | 0.4910 | 0.0130 | 0.0288 |
| P2_later_kaggle_lfp | CH_LFP | p2_lfp_charge_same | dcdetector | 3407.0000 | 0.4794 | 0.0010 | 0.0075 |
| P2_later_kaggle_ncm | CH_NCM | p2_ncm_charge_same | dcdetector | 3407.0000 | 0.5538 | 0.0045 | 0.0127 |
| P2_later_kaggle_ncm | CH_NCM | p2_ncm_charge_to_discharge | dcdetector | 3407.0000 | 0.6322 | 0.4090 | 0.3237 |
| P2_later_kaggle_ncm | CH_NCM | p2_ncm_discharge_to_charge | dcdetector | 3407.0000 | 0.6736 | 0.0000 | 0.0028 |

#### 明细：isolation_forest（seed 3407.0000）

| later_run_family | dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2_later_local | CH_LFP | charge_to_charge | isolation_forest | 3407.0000 | 0.6309 | 0.0410 | 0.0125 |
| P2_later_local | CH_LFP | charge_to_discharge | isolation_forest | 3407.0000 | 0.8244 | 0.0000 | 0.0000 |

#### 明细：mtad_gat（seed 3407.0000）

| later_run_family | dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2_later_local | CH_LFP | charge_to_charge | mtad_gat | 3407.0000 | 0.6387 | 0.0443 | 0.0163 |
| P2_later_local | CH_LFP | charge_to_discharge | mtad_gat | 3407.0000 | 0.7525 | 0.9740 | 0.8975 |
| P2_later_kaggle_lfp | CH_LFP | p2_lfp_charge_same | mtad_gat | 3407.0000 | 0.6412 | 0.0443 | 0.0150 |
| P2_later_kaggle_ncm | CH_NCM | p2_ncm_charge_same | mtad_gat | 3407.0000 | 0.8692 | 0.3363 | 0.0155 |
| P2_later_kaggle_ncm | CH_NCM | p2_ncm_charge_to_discharge | mtad_gat | 3407.0000 | 0.6159 | 0.9130 | 0.9375 |
| P2_later_kaggle_ncm | CH_NCM | p2_ncm_discharge_to_charge | mtad_gat | 3407.0000 | 0.7311 | 0.4002 | 0.1761 |

#### 明细：mtad_gat（seed 3408.0000）

| later_run_family | dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2_later_local | CH_LFP | charge_to_charge | mtad_gat | 3408.0000 | 0.6767 | 0.0342 | 0.0063 |
| P2_later_local | CH_LFP | charge_to_discharge | mtad_gat | 3408.0000 | 0.7082 | 0.9780 | 0.9325 |

#### 明细：mtad_gat（seed 3409.0000）

| later_run_family | dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2_later_local | CH_LFP | charge_to_charge | mtad_gat | 3409.0000 | 0.6575 | 0.0352 | 0.0100 |
| P2_later_local | CH_LFP | charge_to_discharge | mtad_gat | 3409.0000 | 0.7198 | 0.9830 | 0.9387 |

#### 明细：tranad（seed 3407.0000）

| later_run_family | dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2_later_local | CH_LFP | charge_to_charge | tranad | 3407.0000 | 0.6901 | 0.0594 | 0.0125 |
| P2_later_local | CH_LFP | charge_to_discharge | tranad | 3407.0000 | 0.6848 | 0.9980 | 0.9900 |
| P2_later_kaggle_ncm | CH_NCM | p2_ncm_charge_same | tranad | 3407.0000 | 0.8149 | 0.0785 | 0.0085 |
| P2_later_kaggle_ncm | CH_NCM | p2_ncm_charge_to_discharge | tranad | 3407.0000 | 0.5012 | 0.9970 | 0.9975 |
| P2_later_kaggle_ncm | CH_NCM | p2_ncm_discharge_to_charge | tranad | 3407.0000 | 0.7841 | 0.4787 | 0.2338 |

#### 明细：tranad（seed 3408.0000）

| later_run_family | dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2_later_local | CH_LFP | charge_to_charge | tranad | 3408.0000 | 0.7149 | 0.0724 | 0.0112 |
| P2_later_local | CH_LFP | charge_to_discharge | tranad | 3408.0000 | 0.7014 | 0.9980 | 0.9838 |

#### 明细：tranad（seed 3409.0000）

| later_run_family | dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2_later_local | CH_LFP | charge_to_charge | tranad | 3409.0000 | 0.6779 | 0.0634 | 0.0150 |
| P2_later_local | CH_LFP | charge_to_discharge | tranad | 3409.0000 | 0.5924 | 0.9980 | 0.9938 |

#### 明细：patchad（seed 3407.0000）

| later_run_family | dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2_later_kaggle_ncm | CH_NCM | p2_ncm_charge_same | patchad | 3407.0000 | 0.5297 | 0.0191 | 0.0197 |
| P2_later_kaggle_ncm | CH_NCM | p2_ncm_charge_to_discharge | patchad | 3407.0000 | 0.5641 | 0.9560 | 0.9325 |
| P2_later_kaggle_ncm | CH_NCM | p2_ncm_discharge_to_charge | patchad | 3407.0000 | 0.5520 | 0.0011 | 0.0000 |

**本表结论：**后续重跑再次出现 `charge_to_discharge` 的高 FPR：LFP 的 MTAD-GAT 为 **0.8975–0.9387**，TranAD 为 **0.9838–0.9938**；与第一轮方向性风险一致，但覆盖不齐，不能作为新均值证据。

</details>
### R11 Kaggle 工况确认（67 个已导入单元）

#### 重点摘要：目标域正常重标定（固定模型、无重训）

| setting | chemistry | method | AP（不变） | FPR：source → target | Recall：source → target |
| --- | --- | --- | ---: | --- | --- |
| charge → discharge | LFP | MTAD-GAT | **0.851** | **0.919 → 0.022** | 0.983 → 0.102 |
| charge → discharge | LFP | TranAD | 0.811 | **0.989 → 0.018** | 1.000 → 0.050 |
| charge → discharge | NCM | MTAD-GAT | 0.751 | **0.952 → 0.019** | 0.934 → 0.041 |
| charge → discharge | NCM | TranAD | 0.671 | **0.999 → 0.002** | 0.997 → 0.017 |
| discharge → charge | NCM | MTAD-GAT | **0.822** | 0.207 → 0.015 | 0.396 → **0.217** |

**本表结论：**重标定能把 `charge→discharge` 的 FPR 从 **0.919–0.999** 压至 **0.002–0.022**，但 Recall 降至 0.017–0.102。AP 不随阈值改变：这证明存在显著阈值偏移，**不证明仅调阈值即可恢复检测能力**。

其中 LFP 为 31 个、NCM 为 36 个。该批任务在取消前留下的已完成单元覆盖不平衡，因此逐单元展示，不把不同模型或不等数量 seed 合并成统一均值。

<details>
<summary>展开逐模型 / seed 完整明细</summary>

#### 明细：mtad_gat（seed 3407.0000）

| dataset | setting | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP_CHARGE | p3_lfp_charge_to_charge | mtad_gat | 3407.0000 | 0.6412 | 0.0443 | 0.0150 |
| CH_LFP_CHARGE | p3_lfp_charge_to_discharge | mtad_gat | 3407.0000 | 0.7566 | 0.9810 | 0.9087 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_charge | mtad_gat | 3407.0000 | 0.6532 | 0.1329 | 0.0750 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_discharge | mtad_gat | 3407.0000 | 0.8603 | 0.2270 | 0.0088 |
| CH_NCM_CHARGE | p3_ncm_charge_to_charge | mtad_gat | 3407.0000 | 0.8692 | 0.3363 | 0.0155 |
| CH_NCM_CHARGE | p3_ncm_charge_to_discharge | mtad_gat | 3407.0000 | 0.6159 | 0.9130 | 0.9375 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_charge | mtad_gat | 3407.0000 | 0.7311 | 0.4002 | 0.1761 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_discharge | mtad_gat | 3407.0000 | 0.7869 | 0.1770 | 0.0150 |

#### 明细：mtad_gat（seed 3408.0000）

| dataset | setting | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP_CHARGE | p3_lfp_charge_to_charge | mtad_gat | 3408.0000 | 0.6772 | 0.0363 | 0.0187 |
| CH_LFP_CHARGE | p3_lfp_charge_to_discharge | mtad_gat | 3408.0000 | 0.7399 | 0.9820 | 0.9287 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_charge | mtad_gat | 3408.0000 | 0.6663 | 0.1682 | 0.1037 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_discharge | mtad_gat | 3408.0000 | 0.8665 | 0.2250 | 0.0075 |
| CH_NCM_CHARGE | p3_ncm_charge_to_charge | mtad_gat | 3408.0000 | 0.8653 | 0.3330 | 0.0141 |
| CH_NCM_CHARGE | p3_ncm_charge_to_discharge | mtad_gat | 3408.0000 | 0.5915 | 0.9430 | 0.9675 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_charge | mtad_gat | 3408.0000 | 0.7109 | 0.4260 | 0.2775 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_discharge | mtad_gat | 3408.0000 | 0.7908 | 0.1690 | 0.0138 |

#### 明细：mtad_gat（seed 3409.0000）

| dataset | setting | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP_CHARGE | p3_lfp_charge_to_charge | mtad_gat | 3409.0000 | 0.6943 | 0.0413 | 0.0175 |
| CH_LFP_CHARGE | p3_lfp_charge_to_discharge | mtad_gat | 3409.0000 | 0.7533 | 0.9860 | 0.9325 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_discharge | mtad_gat | 3409.0000 | 0.8633 | 0.2280 | 0.0063 |
| CH_NCM_CHARGE | p3_ncm_charge_to_charge | mtad_gat | 3409.0000 | 0.8557 | 0.3173 | 0.0141 |
| CH_NCM_CHARGE | p3_ncm_charge_to_discharge | mtad_gat | 3409.0000 | 0.5824 | 0.9450 | 0.9712 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_charge | mtad_gat | 3409.0000 | 0.7202 | 0.3632 | 0.1986 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_discharge | mtad_gat | 3409.0000 | 0.7876 | 0.1650 | 0.0125 |

#### 明细：patchad（seed 3407.0000）

| dataset | setting | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP_CHARGE | p3_lfp_charge_to_charge | patchad | 3407.0000 | 0.7179 | 0.0302 | 0.0075 |
| CH_LFP_CHARGE | p3_lfp_charge_to_discharge | patchad | 3407.0000 | 0.5093 | 0.8970 | 0.8363 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_charge | patchad | 3407.0000 | 0.4800 | 0.0735 | 0.1300 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_discharge | patchad | 3407.0000 | 0.7446 | 0.0700 | 0.0138 |
| CH_NCM_CHARGE | p3_ncm_charge_to_charge | patchad | 3407.0000 | 0.5297 | 0.0191 | 0.0197 |
| CH_NCM_CHARGE | p3_ncm_charge_to_discharge | patchad | 3407.0000 | 0.5641 | 0.9560 | 0.9325 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_charge | patchad | 3407.0000 | 0.5520 | 0.0011 | 0.0000 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_discharge | patchad | 3407.0000 | 0.6054 | 0.0280 | 0.0088 |

#### 明细：patchad（seed 3408.0000）

| dataset | setting | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP_CHARGE | p3_lfp_charge_to_charge | patchad | 3408.0000 | 0.7333 | 0.0252 | 0.0138 |
| CH_LFP_CHARGE | p3_lfp_charge_to_discharge | patchad | 3408.0000 | 0.6859 | 1.0000 | 0.9775 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_charge | patchad | 3408.0000 | 0.4893 | 0.0554 | 0.0900 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_discharge | patchad | 3408.0000 | 0.7057 | 0.0330 | 0.0088 |
| CH_NCM_CHARGE | p3_ncm_charge_to_charge | patchad | 3408.0000 | 0.5230 | 0.0045 | 0.0141 |
| CH_NCM_CHARGE | p3_ncm_charge_to_discharge | patchad | 3408.0000 | 0.5553 | 0.9470 | 0.9825 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_charge | patchad | 3408.0000 | 0.4952 | 0.0045 | 0.0127 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_discharge | patchad | 3408.0000 | 0.5874 | 0.0160 | 0.0138 |

#### 明细：patchad（seed 3409.0000）

| dataset | setting | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP_CHARGE | p3_lfp_charge_to_charge | patchad | 3409.0000 | 0.7505 | 0.0242 | 0.0187 |
| CH_LFP_CHARGE | p3_lfp_charge_to_discharge | patchad | 3409.0000 | 0.7160 | 0.9570 | 0.6050 |
| CH_NCM_CHARGE | p3_ncm_charge_to_charge | patchad | 3409.0000 | 0.5179 | 0.0056 | 0.0183 |
| CH_NCM_CHARGE | p3_ncm_charge_to_discharge | patchad | 3409.0000 | 0.5866 | 1.0000 | 0.9975 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_charge | patchad | 3409.0000 | 0.6761 | 0.0045 | 0.0014 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_discharge | patchad | 3409.0000 | 0.6192 | 0.0420 | 0.0100 |

#### 明细：tranad（seed 3407.0000）

| dataset | setting | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP_CHARGE | p3_lfp_charge_to_charge | tranad | 3407.0000 | 0.7137 | 0.0795 | 0.0063 |
| CH_LFP_CHARGE | p3_lfp_charge_to_discharge | tranad | 3407.0000 | 0.6847 | 1.0000 | 0.9925 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_charge | tranad | 3407.0000 | 0.6477 | 0.2113 | 0.1350 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_discharge | tranad | 3407.0000 | 0.8443 | 0.1780 | 0.0075 |
| CH_NCM_CHARGE | p3_ncm_charge_to_charge | tranad | 3407.0000 | 0.8149 | 0.0785 | 0.0085 |
| CH_NCM_CHARGE | p3_ncm_charge_to_discharge | tranad | 3407.0000 | 0.5012 | 0.9970 | 0.9975 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_charge | tranad | 3407.0000 | 0.7841 | 0.4787 | 0.2338 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_discharge | tranad | 3407.0000 | 0.7646 | 0.1880 | 0.0088 |

#### 明细：tranad（seed 3408.0000）

| dataset | setting | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP_CHARGE | p3_lfp_charge_to_charge | tranad | 3408.0000 | 0.7519 | 0.1087 | 0.0037 |
| CH_LFP_CHARGE | p3_lfp_charge_to_discharge | tranad | 3408.0000 | 0.6948 | 1.0000 | 0.9938 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_charge | tranad | 3408.0000 | 0.6799 | 0.2294 | 0.1237 |
| CH_LFP_DISCHARGE | p3_lfp_discharge_to_discharge | tranad | 3408.0000 | 0.8784 | 0.2270 | 0.0150 |
| CH_NCM_CHARGE | p3_ncm_charge_to_charge | tranad | 3408.0000 | 0.8285 | 0.0751 | 0.0141 |
| CH_NCM_CHARGE | p3_ncm_charge_to_discharge | tranad | 3408.0000 | 0.5048 | 0.9960 | 0.9988 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_charge | tranad | 3408.0000 | 0.6218 | 0.1760 | 0.1155 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_discharge | tranad | 3408.0000 | 0.6811 | 0.0640 | 0.0063 |

#### 明细：tranad（seed 3409.0000）

| dataset | setting | method | model_seed | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP_CHARGE | p3_lfp_charge_to_charge | tranad | 3409.0000 | 0.7013 | 0.0644 | 0.0100 |
| CH_LFP_CHARGE | p3_lfp_charge_to_discharge | tranad | 3409.0000 | 0.6706 | 0.9990 | 0.9912 |
| CH_NCM_CHARGE | p3_ncm_charge_to_charge | tranad | 3409.0000 | 0.8276 | 0.1939 | 0.0056 |
| CH_NCM_CHARGE | p3_ncm_charge_to_discharge | tranad | 3409.0000 | 0.4983 | 0.9980 | 1.0000 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_charge | tranad | 3409.0000 | 0.6175 | 0.3173 | 0.3408 |
| CH_NCM_DISCHARGE | p3_ncm_discharge_to_discharge | tranad | 3409.0000 | 0.6853 | 0.0780 | 0.0063 |

该表仅登记已从 Kaggle 输出下载或从用户提供的 Kaggle output ZIP 中提取、并通过文件完整性检查的单元；未完成单元不补录为结果。

</details>
## P3/P4：化学体系迁移

### 重点摘要：目标域同域基线 → 化学体系迁移（第一轮、同一 seed）

下表基线为 P1 第一轮中**同一目标体系、同一方法、同一 seed**的域内 AUPRC；因此 ΔAUPRC 才可解释为迁移后相对同域的排序变化。

| transfer → target | method | 目标域同域基线 AUPRC | 迁移 AUPRC | ΔAUPRC | 迁移 FPR | POT：FPR / R / F1 | 目标域 P99：FPR / R / F1 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| LFP → NCM | DCdetector | 0.665 | 0.565 | −0.099 | 0.112 | 0.098 / 0.112 / 0.195 | **0.005 / 0.006 / 0.012** |
| LFP → NCM | MTAD-GAT | 0.787 | 0.565 | **−0.222** | **1.000** | 1.000 / 1.000 / 0.833 | **0.020 / 0.028 / 0.054** |
| LFP → NCM | PatchAD | 0.605 | 0.536 | −0.070 | **1.000** | 1.000 / 0.996 / 0.831 | **0.022 / 0.030 / 0.058** |
| LFP → NCM | TranAD | 0.765 | 0.560 | **−0.205** | **1.000** | 1.000 / 1.000 / 0.833 | **0.015 / 0.026 / 0.050** |
| NCM → LFP | DCdetector | 0.736 | 0.438 | **−0.299** | 0.338 | 0.282 / 0.101 / 0.166 | **0.000 / 0.000 / 0.000** |
| NCM → LFP | MTAD-GAT | 0.860 | 0.768 | −0.092 | **1.000** | 1.000 / 1.000 / 0.833 | **0.050 / 0.111 / 0.196** |
| NCM → LFP | PatchAD | 0.745 | 0.815 | +0.070 | 0.324 | 0.338 / 0.803 / 0.829 | **0.022 / 0.088 / 0.160** |
| NCM → LFP | TranAD | 0.844 | 0.815 | −0.030 | **1.000** | 1.000 / 1.000 / 0.833 | **0.030 / 0.124 / 0.218** |
| mixed → LFP | TranAD | 0.844 | 0.824 | −0.020 | 0.015 | 0.022 / 0.253 / 0.401 | 0.035 / 0.281 / 0.434 |
| mixed → NCM | TranAD | 0.765 | 0.688 | −0.077 | 0.009 | 0.015 / 0.073 / 0.135 | 0.035 / 0.121 / 0.213 |

新增两列同样在 VIN 隔离的目标域正常留出集上计算，指标顺序为 `FPR / Recall / F1`。

**本表结论：**体系迁移的主参照是目标域同域基线，而非不同体系之间的绝对 AUPRC。`LFP→NCM` 中 MTAD-GAT/TranAD 分别下降 **0.222/0.205**；`NCM→LFP` 虽有部分排序接近基线，但训练域阈值下 MTAD-GAT/TranAD 的 FPR=**1.000**。这些高 FPR 需以下述阈值控制结果与排序退化分开读。

### 阈值控制对照（P3/P4 全部 16 个已保存运行；不重训）

| 阈值策略 | AUPRC（固定） | FPR | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 源训练正常 P99 | 0.783 | 0.365 | 0.761 | 0.436 | 0.429 |
| 源验证正常 P99 | 0.783 | 0.383 | 0.760 | 0.453 | 0.451 |
| 静态 POT（源训练正常尾部；目标 FPR=1%） | 0.783 | 0.363 | 0.760 | 0.435 | 0.427 |
| **目标域正常 P99（VIN 隔离）** | 0.783 | **0.022** | **0.809** | 0.102 | 0.168 |

**本表结论：**源验证 P99 和静态 POT 均未纠正体系迁移的平均高 FPR；目标域正常 P99 可将平均 FPR 控制至约 2%，却牺牲大部分召回。因此 P3/P4 的核心应改为：**体系 × 工况造成的分数分布与类别排序同时变化**，不能仅写成“LFP/NCM 哪个更难”。完整 64 行仍见 [阈值控制 CSV](ch_batterygen_p1_p7_data/P2_P5_static_pot_primary_rows.csv)。

### IF false-positive attribution：NCM→LFP（固定训练 P99；不重训）

这是对已保存的 `NCM→LFP / Isolation Forest / seed 3407` 的逐正常样本审计：800 个目标域正常段中 **597 个 FP（FPR=0.746）**、203 个 TN。FP/TN 仅由已保存的源训练正常 P99 阈值划分；下表不使用故障标签调阈值。

| 目标域正常段属性 | FP 中位数 | TN 中位数 | FP / TN | 高动态区间的 P(·\|FP) vs P(·\|TN) |
| --- | ---: | ---: | ---: | --- |
| 平均 \|dI/dt\| | **2.609** | 1.032 | **2.53×** | **33.3% vs 0.5%**（≥目标域 P75） |
| P95 \|dI/dt\| | **11.185** | 5.995 | **1.87×** | **33.5% vs 0.0%**（≥目标域 P75） |
| 平均 \|dSOC/dt\| | **0.00303** | 0.00123 | **2.48×** | **33.5% vs 0.0%**（≥目标域 P75） |
| 电流绝对值中位数 | **4.7** | 3.4 | **1.38×** | **33.2% vs 1.0%**（≥目标域 P75） |
| SOC 最小值 | 79 | 87 | 低 8 个点 | 低 SOC 边界并非主要富集项 |

上述动态变量的 FP/TN 差异均有 Mann–Whitney 和 Fisher 精确检验 `p < 0.001`。**本表结论：**这里首次把“阈值不稳定”具体化为可检验假设：IF 在迁移后会系统性把**高电流变化、高 SOC 变化的合法正常放电段**推到源域分数尾部。此数据集的每个 CSV 已是单一充/放电段，`CHARGE_STATUS` 在段内没有转换，故不能诚实报告“距离工况转换多近”；`cycle_index` 也只是同一 VIN 内的循环序号，不是全局时间位置。完整样本特征与检验结果见 [IF FP attribution](ch_batterygen_if_fp_attribution_ncm_to_lfp/README.md) 和 [CSV](ch_batterygen_if_fp_attribution_ncm_to_lfp/fp_vs_tn_attribution.csv)。

### 第一轮（16 个单元）

<details>
<summary>展开逐模型 / seed 完整明细</summary>

#### 明细：dcdetector

| dataset | setting | method | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| CH_LFP_DISCHARGE | p34_lfp_to_ncm | dcdetector | 0.5654 | 0.1170 | 0.1125 |
| CH_MIXED_LFP_DISCHARGE | p34_mixed_to_lfp | dcdetector | 0.7230 | 0.3980 | 0.0225 |
| CH_MIXED_NCM_DISCHARGE | p34_mixed_to_ncm | dcdetector | 0.5872 | 0.0000 | 0.0013 |
| CH_NCM_DISCHARGE | p34_ncm_to_lfp | dcdetector | 0.4377 | 0.1160 | 0.3375 |

#### 明细：mtad_gat

| dataset | setting | method | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| CH_LFP_DISCHARGE | p34_lfp_to_ncm | mtad_gat | 0.5650 | 1.0000 | 1.0000 |
| CH_MIXED_LFP_DISCHARGE | p34_mixed_to_lfp | mtad_gat | 0.8134 | 0.1830 | 0.0112 |
| CH_MIXED_NCM_DISCHARGE | p34_mixed_to_ncm | mtad_gat | 0.6521 | 0.0100 | 0.0000 |
| CH_NCM_DISCHARGE | p34_ncm_to_lfp | mtad_gat | 0.7680 | 1.0000 | 1.0000 |

#### 明细：patchad

| dataset | setting | method | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| CH_LFP_DISCHARGE | p34_lfp_to_ncm | patchad | 0.5357 | 0.9960 | 1.0000 |
| CH_MIXED_LFP_DISCHARGE | p34_mixed_to_lfp | patchad | 0.7419 | 0.0080 | 0.0013 |
| CH_MIXED_NCM_DISCHARGE | p34_mixed_to_ncm | patchad | 0.5861 | 0.0190 | 0.0100 |
| CH_NCM_DISCHARGE | p34_ncm_to_lfp | patchad | 0.8147 | 0.8020 | 0.3237 |

#### 明细：tranad

| dataset | setting | method | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| CH_LFP_DISCHARGE | p34_lfp_to_ncm | tranad | 0.5597 | 1.0000 | 1.0000 |
| CH_MIXED_LFP_DISCHARGE | p34_mixed_to_lfp | tranad | 0.8240 | 0.2530 | 0.0150 |
| CH_MIXED_NCM_DISCHARGE | p34_mixed_to_ncm | tranad | 0.6876 | 0.0730 | 0.0088 |
| CH_NCM_DISCHARGE | p34_ncm_to_lfp | tranad | 0.8148 | 1.0000 | 1.0000 |

</details>
### 后续重跑（40 个单元）

<details>
<summary>展开逐模型 / seed 完整明细</summary>

#### 明细：dcdetector（seed 3407.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | lfp_to_ncm | dcdetector | 3407.0000 | 0.6115 | 0.0120 | 0.0000 |
| CH_LFP | mixed_to_lfp | dcdetector | 3407.0000 | 0.6927 | 0.4020 | 0.0250 |
| CH_NCM | mixed_to_ncm | dcdetector | 3407.0000 | 0.5836 | 0.0000 | 0.0013 |
| CH_NCM | ncm_to_lfp | dcdetector | 3407.0000 | 0.5191 | 0.2240 | 0.3900 |

#### 明细：dcdetector（seed 3408.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | lfp_to_ncm | dcdetector | 3408.0000 | 0.5747 | 0.0010 | 0.0000 |
| CH_LFP | mixed_to_lfp | dcdetector | 3408.0000 | 0.5139 | 0.0230 | 0.0200 |
| CH_NCM | mixed_to_ncm | dcdetector | 3408.0000 | 0.6660 | 0.0000 | 0.0000 |
| CH_NCM | ncm_to_lfp | dcdetector | 3408.0000 | 0.5386 | 0.6320 | 0.6375 |

#### 明细：dcdetector（seed 3409.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | lfp_to_ncm | dcdetector | 3409.0000 | 0.5779 | 0.0000 | 0.0000 |
| CH_LFP | mixed_to_lfp | dcdetector | 3409.0000 | 0.6856 | 0.3950 | 0.0288 |
| CH_NCM | mixed_to_ncm | dcdetector | 3409.0000 | 0.5387 | 0.0020 | 0.0037 |
| CH_NCM | ncm_to_lfp | dcdetector | 3409.0000 | 0.4495 | 0.0080 | 0.0262 |

#### 明细：isolation_forest（seed 3407.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | lfp_to_ncm | isolation_forest | 3407.0000 | 0.4831 | 0.1210 | 0.1975 |
| CH_NCM | ncm_to_lfp | isolation_forest | 3407.0000 | 0.7809 | 0.9870 | 0.7462 |

#### 明细：mtad_gat（seed 3407.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | lfp_to_ncm | mtad_gat | 3407.0000 | 0.5614 | 1.0000 | 1.0000 |
| CH_LFP | lfp_to_ncm | mtad_gat | 3407.0000 | 0.5627 | 1.0000 | 1.0000 |
| CH_LFP | mixed_to_lfp | mtad_gat | 3407.0000 | 0.8103 | 0.1830 | 0.0112 |
| CH_NCM | mixed_to_ncm | mtad_gat | 3407.0000 | 0.6339 | 0.0100 | 0.0000 |
| CH_NCM | ncm_to_lfp | mtad_gat | 3407.0000 | 0.8202 | 1.0000 | 1.0000 |
| CH_NCM | ncm_to_lfp | mtad_gat | 3407.0000 | 0.7996 | 1.0000 | 1.0000 |

#### 明细：mtad_gat（seed 3408.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | lfp_to_ncm | mtad_gat | 3408.0000 | 0.5543 | 1.0000 | 1.0000 |
| CH_LFP | mixed_to_lfp | mtad_gat | 3408.0000 | 0.8144 | 0.1950 | 0.0150 |
| CH_NCM | mixed_to_ncm | mtad_gat | 3408.0000 | 0.6107 | 0.0120 | 0.0000 |
| CH_NCM | ncm_to_lfp | mtad_gat | 3408.0000 | 0.7733 | 1.0000 | 1.0000 |

#### 明细：mtad_gat（seed 3409.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | lfp_to_ncm | mtad_gat | 3409.0000 | 0.5473 | 1.0000 | 1.0000 |
| CH_LFP | mixed_to_lfp | mtad_gat | 3409.0000 | 0.7957 | 0.1630 | 0.0075 |
| CH_NCM | mixed_to_ncm | mtad_gat | 3409.0000 | 0.6048 | 0.0200 | 0.0100 |
| CH_NCM | ncm_to_lfp | mtad_gat | 3409.0000 | 0.7191 | 1.0000 | 1.0000 |

#### 明细：tranad（seed 3407.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | lfp_to_ncm | tranad | 3407.0000 | 0.5570 | 1.0000 | 1.0000 |
| CH_LFP | mixed_to_lfp | tranad | 3407.0000 | 0.8301 | 0.2390 | 0.0100 |
| CH_NCM | mixed_to_ncm | tranad | 3407.0000 | 0.6369 | 0.0600 | 0.0088 |
| CH_NCM | ncm_to_lfp | tranad | 3407.0000 | 0.8245 | 1.0000 | 1.0000 |

#### 明细：tranad（seed 3408.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | lfp_to_ncm | tranad | 3408.0000 | 0.5470 | 1.0000 | 1.0000 |
| CH_LFP | mixed_to_lfp | tranad | 3408.0000 | 0.8296 | 0.2380 | 0.0013 |
| CH_NCM | mixed_to_ncm | tranad | 3408.0000 | 0.6669 | 0.0880 | 0.0138 |
| CH_NCM | ncm_to_lfp | tranad | 3408.0000 | 0.8503 | 1.0000 | 1.0000 |

#### 明细：tranad（seed 3409.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | lfp_to_ncm | tranad | 3409.0000 | 0.5462 | 1.0000 | 1.0000 |
| CH_LFP | mixed_to_lfp | tranad | 3409.0000 | 0.7915 | 0.2210 | 0.0112 |
| CH_NCM | mixed_to_ncm | tranad | 3409.0000 | 0.6388 | 0.0490 | 0.0138 |
| CH_NCM | ncm_to_lfp | tranad | 3409.0000 | 0.8124 | 1.0000 | 1.0000 |

**本表结论：**后续重跑仍显示 `mixed_to_lfp` 相对可解释（TranAD AUPRC 0.7915–0.8301、FPR ≤0.0112），而 `ncm_to_lfp` 的 MTAD-GAT/TranAD FPR 均为 **1.000**；这一对比强化了“先检查标定”的判断。

</details>
## P5：输入变量设置

### 重点摘要：变量依赖随模型变化

| method | 最佳设置 | 最佳 AUPRC | full AUPRC | 最明显退化设置 | AUPRC | 最佳设置 POT：FPR / R / F1 | 最佳设置目标域 P99：FPR / R / F1 |
| --- | --- | ---: | ---: | --- | ---: | --- | --- |
| DCdetector | controls only | **0.802** | 0.714 | drop temperature extrema | **0.447** | 0.008 / 0.405 / 0.575 | 0.005 / 0.402 / 0.573 |
| PatchAD | full | **0.745** | 0.745 | drop voltage extrema | 0.602 | 0.012 / 0.061 / 0.114 | 0.010 / 0.057 / 0.107 |
| TranAD | responses only | **0.863** | 0.844 | controls only | **0.641** | 0.018 / 0.186 / 0.312 | 0.032 / 0.245 / 0.390 |

新增两列均对应该行的“最佳设置”，指标顺序为 `FPR / Recall / F1`，使用相同的 VIN 隔离目标域评估协议。

**本表结论：**DCdetector 对温度极值最敏感，而 TranAD 在仅响应变量下排序最高、在仅控制变量下明显变差。因仅覆盖**单 seed、LFP**，这里只形成变量依赖假设，不能推广。

### 阈值控制对照（P5 全部 15 个已保存运行；不重训）

| 阈值策略 | AUPRC（固定） | FPR | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 源训练正常 P99 | 0.807 | 0.011 | 0.823 | 0.084 | 0.137 |
| 源验证正常 P99 | 0.807 | 0.018 | 0.754 | 0.099 | **0.159** |
| 静态 POT（源训练正常尾部；目标 FPR=1%） | 0.807 | **0.010** | 0.820 | 0.083 | 0.136 |
| 目标域正常 P99（VIN 隔离） | 0.807 | 0.015 | 0.812 | 0.096 | 0.154 |

**本表结论：**P5 的 FPR 原本已约 1%，四种阈值策略差异很小；变量设置导致的 AUPRC 差异因而**不是全局阈值失配造成的**。这使“响应变量/控制变量的作用依模型而变”的现象比 P2/P3-P4 更值得保留，但仍须多 seed 验证。完整 60 行见 [阈值控制 CSV](ch_batterygen_p1_p7_data/P2_P5_static_pot_primary_rows.csv)。

### 第一轮（15 个单元）

<details>
<summary>展开逐模型 / seed 完整明细</summary>

#### 明细：dcdetector

| dataset | setting | method | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| CH_LFP_DISCHARGE | p5_schema_controls_only | dcdetector | 0.8015 | 0.4070 | 0.0150 |
| CH_LFP_DISCHARGE | p5_schema_drop_temperature_extrema | dcdetector | 0.4470 | 0.0030 | 0.0063 |
| CH_LFP_DISCHARGE | p5_schema_drop_voltage_extrema | dcdetector | 0.5396 | 0.0040 | 0.0112 |
| CH_LFP_DISCHARGE | p5_schema_full | dcdetector | 0.7139 | 0.0290 | 0.0150 |
| CH_LFP_DISCHARGE | p5_schema_responses_only | dcdetector | 0.4922 | 0.0030 | 0.0138 |

#### 明细：patchad

| dataset | setting | method | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| CH_LFP_DISCHARGE | p5_schema_controls_only | patchad | 0.6556 | 0.0090 | 0.0050 |
| CH_LFP_DISCHARGE | p5_schema_drop_temperature_extrema | patchad | 0.7318 | 0.0410 | 0.0063 |
| CH_LFP_DISCHARGE | p5_schema_drop_voltage_extrema | patchad | 0.6018 | 0.0190 | 0.0075 |
| CH_LFP_DISCHARGE | p5_schema_full | patchad | 0.7446 | 0.0700 | 0.0138 |
| CH_LFP_DISCHARGE | p5_schema_responses_only | patchad | 0.6633 | 0.0220 | 0.0100 |

#### 明细：tranad

| dataset | setting | method | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| CH_LFP_DISCHARGE | p5_schema_controls_only | tranad | 0.6409 | 0.0080 | 0.0063 |
| CH_LFP_DISCHARGE | p5_schema_drop_temperature_extrema | tranad | 0.8532 | 0.2160 | 0.0100 |
| CH_LFP_DISCHARGE | p5_schema_drop_voltage_extrema | tranad | 0.8226 | 0.0610 | 0.0150 |
| CH_LFP_DISCHARGE | p5_schema_full | tranad | 0.8443 | 0.1780 | 0.0075 |
| CH_LFP_DISCHARGE | p5_schema_responses_only | tranad | 0.8629 | 0.1890 | 0.0112 |

</details>
### 后续重跑（26 个单元）

<details>
<summary>展开逐模型 / seed 完整明细</summary>

#### 明细：isolation_forest（seed 3407.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | controls_only | isolation_forest | 3407.0000 | 0.6895 | 0.0870 | 0.0213 |
| CH_LFP | drop_temperature_extrema | isolation_forest | 3407.0000 | 0.8249 | 0.4510 | 0.1000 |
| CH_LFP | drop_voltage_extrema | isolation_forest | 3407.0000 | 0.8225 | 0.1100 | 0.0213 |
| CH_LFP | random_mask_10 | isolation_forest | 3407.0000 | 0.8389 | 0.7540 | 0.2238 |
| CH_LFP | random_mask_25 | isolation_forest | 3407.0000 | 0.7617 | 0.3880 | 0.1500 |
| CH_LFP | random_mask_50 | isolation_forest | 3407.0000 | 0.6504 | 0.5690 | 0.4950 |
| CH_LFP | responses_only | isolation_forest | 3407.0000 | 0.8324 | 0.2570 | 0.0488 |

#### 明细：mtad_gat（seed 3407.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | full | mtad_gat | 3407.0000 | 0.8808 | 0.2460 | 0.0075 |
| CH_LFP | random_mask_10 | mtad_gat | 3407.0000 | 0.4999 | 0.0490 | 0.0063 |
| CH_LFP | random_mask_25 | mtad_gat | 3407.0000 | 0.4670 | 0.0340 | 0.0063 |
| CH_LFP | random_mask_50 | mtad_gat | 3407.0000 | 0.4808 | 0.0440 | 0.0100 |

#### 明细：dcdetector（seed 3407.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | controls_only | dcdetector | 3407.0000 | 0.8216 | 0.4010 | 0.0100 |
| CH_LFP | drop_temperature_extrema | dcdetector | 3407.0000 | 0.4500 | 0.0020 | 0.0050 |
| CH_LFP | drop_voltage_extrema | dcdetector | 3407.0000 | 0.6100 | 0.0080 | 0.0088 |
| CH_LFP | full | dcdetector | 3407.0000 | 0.5331 | 0.0030 | 0.0088 |
| CH_LFP | responses_only | dcdetector | 3407.0000 | 0.4157 | 0.0010 | 0.0138 |

#### 明细：patchad（seed 3407.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | controls_only | patchad | 3407.0000 | 0.6257 | 0.0080 | 0.0063 |
| CH_LFP | drop_temperature_extrema | patchad | 3407.0000 | 0.8147 | 0.1060 | 0.0088 |
| CH_LFP | drop_voltage_extrema | patchad | 3407.0000 | 0.7748 | 0.0380 | 0.0200 |
| CH_LFP | full | patchad | 3407.0000 | 0.7082 | 0.0400 | 0.0075 |
| CH_LFP | responses_only | patchad | 3407.0000 | 0.6473 | 0.0310 | 0.0088 |

#### 明细：tranad（seed 3407.0000）

| dataset | setting | method | model_seed_from_path | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| CH_LFP | controls_only | tranad | 3407.0000 | 0.6578 | 0.0090 | 0.0075 |
| CH_LFP | drop_temperature_extrema | tranad | 3407.0000 | 0.8489 | 0.2110 | 0.0088 |
| CH_LFP | drop_voltage_extrema | tranad | 3407.0000 | 0.8430 | 0.0490 | 0.0163 |
| CH_LFP | full | tranad | 3407.0000 | 0.8779 | 0.1910 | 0.0088 |
| CH_LFP | responses_only | tranad | 3407.0000 | 0.8499 | 0.1820 | 0.0125 |

**本表结论：**后续表补充了随机掩码与 Isolation Forest，但仍不是完整的模型 × 设置矩阵；例如 MTAD-GAT 在 `random_mask_10/25/50` 的 AUPRC 为 **0.4999/0.4670/0.4808**，提示缺失变量会显著影响该模型，但不外推为通用规律。

</details>
## P6：时间尺度

### 先说结论：P6 暂时没有可作为论文主张的“最佳时间尺度”

P6 问的是：改变窗口长度（lookback）、滑动步长（stride）或下采样（resample）后，模型是否稳定、是否有统一的最优设置。**目前答案是否定的：只能得到局部、带条件的观察，不能提炼为“某个时间尺度普遍最好”。**

| 已经能读出的现象 | 证据 | 应如何表述 |
| --- | --- | --- |
| MTAD-GAT 在已完成的局部配置附近差异小 | 单个 seed 下 AUPRC 为 0.850–0.864；`lookback=64,stride=4` 为 0.864，`resample=2` 为 0.864 | **仅说明此局部邻域没有明显优劣**；不是跨 seed 鲁棒性结论 |
| 部分模型确实对时间尺度敏感 | PatchAD、DCdetector 在不同窗口/步长下 AUPRC 有明显波动，且 PatchAD 在不同 seed 的结果也不稳定 | **时间尺度可能影响性能，但方向不一致、受模型和随机种子影响** |
| TranAD 的窗口结果不能解释 | `lookback=8/16/32/64` 在每个 seed 内完全相同 | **先核验 lookback 是否传入 runner；在此之前不能称“窗口鲁棒”** |
| TranAD 的下采样有可见差异 | `resample=2` 平均 AUPRC 约 0.867，`resample=4` 约 0.840 | 这是一个候选现象，仍需固定其他条件、多 seed 复现 |

**本表结论：P6 不是“没有任何信息”，但它的有效结论是负向的：当前配置矩阵不足以找出统一最优时间尺度；其中 TranAD 的窗口实验还存在配置生效性疑点。** 因而 P6 不应作为主研究问题，也不应用来宣称时间尺度鲁棒。

P6 有 108 个通过 artifact 审计的单元；其中 105 个可从当前目录恢复出完整的 P99 主指标行。缺失的三行不被填补或推断。下表是全部已恢复主指标行；108 单元的完整性审计表见前述 CSV 链接。

### 已恢复主指标（105 行）

<details>
<summary>展开逐模型 / seed 完整明细</summary>

#### 明细：dcdetector（seed 3407.0000）

| method | case | model_seed | sample_auroc | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| dcdetector | lookback_16 | 3407.0000 | 0.5825 | 0.7387 | 0.4000 | 0.0063 |
| dcdetector | lookback_32 | 3407.0000 | 0.6204 | 0.6268 | 0.0150 | 0.0125 |
| dcdetector | lookback_64 | 3407.0000 | 0.3757 | 0.4643 | 0.0000 | 0.0125 |
| dcdetector | lookback_8 | 3407.0000 | 0.5704 | 0.5627 | 0.0050 | 0.0125 |
| dcdetector | resample_2 | 3407.0000 | 0.6643 | 0.7250 | 0.0020 | 0.0075 |
| dcdetector | resample_4 | 3407.0000 | 0.6131 | 0.7191 | 0.0910 | 0.0075 |
| dcdetector | stride_1 | 3407.0000 | 0.6256 | 0.6738 | 0.0000 | 0.0013 |
| dcdetector | stride_4_resample_1 | 3407.0000 | 0.6666 | 0.6922 | 0.0180 | 0.0075 |
| dcdetector | stride_8 | 3407.0000 | 0.6326 | 0.7622 | 0.3980 | 0.0163 |

#### 明细：dcdetector（seed 3408.0000）

| method | case | model_seed | sample_auroc | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| dcdetector | lookback_16 | 3408.0000 | 0.6114 | 0.7484 | 0.4000 | 0.0112 |
| dcdetector | lookback_32 | 3408.0000 | 0.5455 | 0.5424 | 0.0010 | 0.0100 |
| dcdetector | lookback_64 | 3408.0000 | 0.5356 | 0.5681 | 0.0190 | 0.0125 |
| dcdetector | lookback_8 | 3408.0000 | 0.4490 | 0.4790 | 0.0020 | 0.0100 |
| dcdetector | resample_2 | 3408.0000 | 0.6773 | 0.7660 | 0.3420 | 0.0125 |
| dcdetector | resample_4 | 3408.0000 | 0.6092 | 0.6965 | 0.0770 | 0.0088 |
| dcdetector | stride_1 | 3408.0000 | 0.3824 | 0.4869 | 0.0060 | 0.0075 |
| dcdetector | stride_4_resample_1 | 3408.0000 | 0.5147 | 0.5239 | 0.0020 | 0.0112 |
| dcdetector | stride_8 | 3408.0000 | 0.4435 | 0.5110 | 0.0060 | 0.0100 |

#### 明细：dcdetector（seed 3409.0000）

| method | case | model_seed | sample_auroc | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| dcdetector | lookback_16 | 3409.0000 | 0.5964 | 0.7429 | 0.4000 | 0.0063 |
| dcdetector | lookback_32 | 3409.0000 | 0.6356 | 0.7526 | 0.4000 | 0.0175 |
| dcdetector | lookback_64 | 3409.0000 | 0.4385 | 0.4984 | 0.0050 | 0.0163 |
| dcdetector | lookback_8 | 3409.0000 | 0.6052 | 0.6024 | 0.0000 | 0.0088 |
| dcdetector | resample_2 | 3409.0000 | 0.6752 | 0.6696 | 0.0020 | 0.0112 |
| dcdetector | resample_4 | 3409.0000 | 0.6410 | 0.6624 | 0.0010 | 0.0150 |
| dcdetector | stride_1 | 3409.0000 | 0.6194 | 0.7559 | 0.4000 | 0.0150 |
| dcdetector | stride_4_resample_1 | 3409.0000 | 0.6365 | 0.7527 | 0.4000 | 0.0138 |
| dcdetector | stride_8 | 3409.0000 | 0.5632 | 0.7276 | 0.3960 | 0.0112 |

#### 明细：isolation_forest（seed 3407.0000）

| method | case | model_seed | sample_auroc | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| isolation_forest | lookback_16_stride_4 | 3407.0000 | 0.8421 | 0.8322 | 0.2740 | 0.0537 |
| isolation_forest | lookback_64_stride_4 | 3407.0000 | 0.8421 | 0.8322 | 0.3470 | 0.0675 |
| isolation_forest | lookback_8_stride_4 | 3407.0000 | 0.8421 | 0.8322 | 0.2480 | 0.0462 |

#### 明细：mtad_gat（seed 3407.0000）

| method | case | model_seed | sample_auroc | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| mtad_gat | lookback_32_stride_1 | 3407.0000 | 0.8300 | 0.8605 | 0.2240 | 0.0063 |
| mtad_gat | lookback_32_stride_8 | 3407.0000 | 0.8293 | 0.8585 | 0.2250 | 0.0075 |
| mtad_gat | lookback_64_stride_4 | 3407.0000 | 0.8328 | 0.8640 | 0.2300 | 0.0050 |
| mtad_gat | lookback_8_stride_4 | 3407.0000 | 0.7943 | 0.8498 | 0.2680 | 0.0063 |
| mtad_gat | resample_2 | 3407.0000 | 0.8337 | 0.8644 | 0.2310 | 0.0100 |
| mtad_gat | resample_4 | 3407.0000 | 0.8221 | 0.8588 | 0.2220 | 0.0112 |

#### 明细：patchad（seed 3407.0000）

| method | case | model_seed | sample_auroc | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| patchad | lookback_16 | 3407.0000 | 0.6387 | 0.6637 | 0.0260 | 0.0112 |
| patchad | lookback_16 | 3407.0000 | 0.6387 | 0.6637 | 0.0260 | 0.0112 |
| patchad | lookback_32 | 3407.0000 | 0.7098 | 0.7446 | 0.0700 | 0.0138 |
| patchad | lookback_32 | 3407.0000 | 0.7098 | 0.7446 | 0.0700 | 0.0138 |
| patchad | lookback_64 | 3407.0000 | 0.7748 | 0.8039 | 0.1170 | 0.0088 |
| patchad | lookback_64 | 3407.0000 | 0.7748 | 0.8039 | 0.1170 | 0.0088 |
| patchad | lookback_8 | 3407.0000 | 0.7406 | 0.7596 | 0.0490 | 0.0050 |
| patchad | lookback_8 | 3407.0000 | 0.7406 | 0.7596 | 0.0490 | 0.0050 |
| patchad | resample_2 | 3407.0000 | 0.7188 | 0.7500 | 0.0500 | 0.0037 |
| patchad | resample_4 | 3407.0000 | 0.6221 | 0.6297 | 0.0120 | 0.0138 |
| patchad | stride_1 | 3407.0000 | 0.7184 | 0.7393 | 0.0440 | 0.0088 |
| patchad | stride_1 | 3407.0000 | 0.7184 | 0.7393 | 0.0440 | 0.0088 |
| patchad | stride_4_resample_1 | 3407.0000 | 0.7098 | 0.7446 | 0.0700 | 0.0138 |
| patchad | stride_8 | 3407.0000 | 0.6892 | 0.7164 | 0.0620 | 0.0112 |

#### 明细：patchad（seed 3408.0000）

| method | case | model_seed | sample_auroc | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| patchad | lookback_16 | 3408.0000 | 0.6835 | 0.7088 | 0.0470 | 0.0125 |
| patchad | lookback_16 | 3408.0000 | 0.6835 | 0.7088 | 0.0470 | 0.0125 |
| patchad | lookback_32 | 3408.0000 | 0.6806 | 0.7057 | 0.0330 | 0.0088 |
| patchad | lookback_32 | 3408.0000 | 0.6806 | 0.7057 | 0.0330 | 0.0088 |
| patchad | lookback_64 | 3408.0000 | 0.5327 | 0.6038 | 0.0350 | 0.0112 |
| patchad | lookback_64 | 3408.0000 | 0.5327 | 0.6038 | 0.0350 | 0.0112 |
| patchad | lookback_8 | 3408.0000 | 0.6654 | 0.6910 | 0.0430 | 0.0112 |
| patchad | lookback_8 | 3408.0000 | 0.6654 | 0.6910 | 0.0430 | 0.0112 |
| patchad | resample_2 | 3408.0000 | 0.6811 | 0.7114 | 0.0310 | 0.0088 |
| patchad | resample_4 | 3408.0000 | 0.6620 | 0.6778 | 0.0200 | 0.0075 |
| patchad | stride_1 | 3408.0000 | 0.7474 | 0.7643 | 0.0780 | 0.0163 |
| patchad | stride_1 | 3408.0000 | 0.7474 | 0.7643 | 0.0780 | 0.0163 |
| patchad | stride_4_resample_1 | 3408.0000 | 0.6806 | 0.7057 | 0.0330 | 0.0088 |
| patchad | stride_8 | 3408.0000 | 0.7257 | 0.7410 | 0.0310 | 0.0050 |

#### 明细：patchad（seed 3409.0000）

| method | case | model_seed | sample_auroc | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| patchad | lookback_16 | 3409.0000 | 0.6612 | 0.6746 | 0.0360 | 0.0125 |
| patchad | lookback_16 | 3409.0000 | 0.6612 | 0.6746 | 0.0360 | 0.0125 |
| patchad | lookback_32 | 3409.0000 | 0.7347 | 0.7637 | 0.0730 | 0.0112 |
| patchad | lookback_32 | 3409.0000 | 0.7347 | 0.7637 | 0.0730 | 0.0112 |
| patchad | lookback_64 | 3409.0000 | 0.4749 | 0.5396 | 0.0150 | 0.0112 |
| patchad | lookback_64 | 3409.0000 | 0.4749 | 0.5396 | 0.0150 | 0.0112 |
| patchad | lookback_8 | 3409.0000 | 0.6632 | 0.6935 | 0.0460 | 0.0163 |
| patchad | lookback_8 | 3409.0000 | 0.6632 | 0.6935 | 0.0460 | 0.0163 |
| patchad | resample_2 | 3409.0000 | 0.6925 | 0.7048 | 0.0420 | 0.0187 |
| patchad | resample_4 | 3409.0000 | 0.6758 | 0.7055 | 0.0250 | 0.0037 |
| patchad | stride_1 | 3409.0000 | 0.7713 | 0.7890 | 0.0930 | 0.0075 |
| patchad | stride_1 | 3409.0000 | 0.7713 | 0.7890 | 0.0930 | 0.0075 |
| patchad | stride_4_resample_1 | 3409.0000 | 0.7347 | 0.7637 | 0.0730 | 0.0112 |
| patchad | stride_8 | 3409.0000 | 0.7033 | 0.7257 | 0.0430 | 0.0100 |

#### 明细：tranad（seed 3407.0000）

| method | case | model_seed | sample_auroc | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| tranad | lookback_16 | 3407.0000 | 0.8170 | 0.8443 | 0.1780 | 0.0075 |
| tranad | lookback_32 | 3407.0000 | 0.8170 | 0.8443 | 0.1780 | 0.0075 |
| tranad | lookback_64 | 3407.0000 | 0.8170 | 0.8443 | 0.1780 | 0.0075 |
| tranad | lookback_8 | 3407.0000 | 0.8170 | 0.8443 | 0.1780 | 0.0075 |
| tranad | resample_2 | 3407.0000 | 0.8723 | 0.8809 | 0.1660 | 0.0100 |
| tranad | resample_4 | 3407.0000 | 0.8088 | 0.8291 | 0.1450 | 0.0125 |
| tranad | stride_1 | 3407.0000 | 0.8582 | 0.8734 | 0.2060 | 0.0100 |
| tranad | stride_4_resample_1 | 3407.0000 | 0.8170 | 0.8443 | 0.1780 | 0.0075 |
| tranad | stride_8 | 3407.0000 | 0.8751 | 0.8719 | 0.1770 | 0.0100 |

#### 明细：tranad（seed 3408.0000）

| method | case | model_seed | sample_auroc | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| tranad | lookback_16 | 3408.0000 | 0.8767 | 0.8784 | 0.2270 | 0.0150 |
| tranad | lookback_32 | 3408.0000 | 0.8767 | 0.8784 | 0.2270 | 0.0150 |
| tranad | lookback_64 | 3408.0000 | 0.8767 | 0.8784 | 0.2270 | 0.0150 |
| tranad | lookback_8 | 3408.0000 | 0.8767 | 0.8784 | 0.2270 | 0.0150 |
| tranad | resample_2 | 3408.0000 | 0.8422 | 0.8613 | 0.1960 | 0.0063 |
| tranad | resample_4 | 3408.0000 | 0.8295 | 0.8546 | 0.1850 | 0.0088 |
| tranad | stride_1 | 3408.0000 | 0.8532 | 0.8645 | 0.2230 | 0.0088 |
| tranad | stride_4_resample_1 | 3408.0000 | 0.8767 | 0.8784 | 0.2270 | 0.0150 |
| tranad | stride_8 | 3408.0000 | 0.8581 | 0.8691 | 0.2110 | 0.0075 |

#### 明细：tranad（seed 3409.0000）

| method | case | model_seed | sample_auroc | sample_auprc | recall | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| tranad | lookback_16 | 3409.0000 | 0.8439 | 0.8588 | 0.2090 | 0.0125 |
| tranad | lookback_32 | 3409.0000 | 0.8439 | 0.8588 | 0.2090 | 0.0125 |
| tranad | lookback_64 | 3409.0000 | 0.8439 | 0.8588 | 0.2090 | 0.0125 |
| tranad | lookback_8 | 3409.0000 | 0.8439 | 0.8588 | 0.2090 | 0.0125 |
| tranad | resample_2 | 3409.0000 | 0.8331 | 0.8592 | 0.1970 | 0.0088 |
| tranad | resample_4 | 3409.0000 | 0.8120 | 0.8375 | 0.1810 | 0.0112 |
| tranad | stride_1 | 3409.0000 | 0.8197 | 0.8464 | 0.2190 | 0.0112 |
| tranad | stride_4_resample_1 | 3409.0000 | 0.8439 | 0.8588 | 0.2090 | 0.0125 |
| tranad | stride_8 | 3409.0000 | 0.8405 | 0.8561 | 0.2200 | 0.0138 |

artifact 完整不等于每一个配置因素确实对模型生效。例如 TranAD 的若干 lookback 设置出现相同分数和相同 raw-score 数量，需在下一轮前检查配置是否真实传入 runner，才能把相同结果解释为时间尺度鲁棒性。

</details>
## P7：raw-score 敏感性

### 先说清楚：P7 目前还没有“实验结论”

P7 **没有训练新模型，也不是 P1–P6 的第七个性能对比实验**。它只是把每次已完成运行的同一份 raw score，用 12 种后处理方式重新切阈值：

`3 种样本分数聚合（max / mean / top5%） × 2 个正常样本校准集（训练 / 验证） × 2 个阈值分位数（P95 / P99）`。

它唯一要回答的问题是：**P1–P6 中看到的现象，是否只在某一套“分数聚合 + 阈值”规则下才出现？** 例如，跨工况的高误报是否在换成 `mean`、P95 或验证正常集校准后仍存在。若 12 种规则下方向一致，原结论才更可信；若方向反转，说明原结论依赖后处理，不能作为模型本体缺陷。

**当前 P7 只完成了“153 个运行均有 12 种后处理记录”的覆盖审计，尚未针对 P2/P3/P5/P6 的关键现象逐项统计“12 种规则中有几种方向一致”。所以它当前不能产生任何性能优劣结论。** 之前那张 60 行表只是证明数据齐全，不是结果表；已移到折叠明细。

### 覆盖审计摘要

| phase | imported runs | 聚合 × 校准集 × 阈值组合 | primary rows | fault-type rows |
| --- | ---: | ---: | ---: | ---: |
| P1 域内筛选 | 8 | 12 | 96 | 288 |
| P2 工况 | 24 | 12 | 288 | 864 |
| P3/P4 体系迁移 | 16 | 12 | 192 | 576 |
| P5 变量设置 | 15 | 12 | 180 | 540 |
| P6 时间尺度 | 90 | 12 | **1,080** | **3,240** |
| 合计 | **153** | — | **1,836** | **5,508** |

**本表结论：**P7 是同一批已导入运行的后处理覆盖审计，不产生新的训练证据。不同阶段的运行数不同，**不能将 1,836 行直接混合平均后比较阶段优劣**。

P7 覆盖 153 个已导入运行，生成 1836 条聚合/阈值记录和 5508 条故障类型分层记录。它支持检查结论是否仅由某种聚合或某个阈值产生；它不增加新的模型训练证据。1836 条逐行指标和 5508 条故障类型逐行指标已分别保存在前述 CSV；下表完整列出每个阶段和后处理条件的覆盖行数，不对不同运行或不同设置求平均。

<details>
<summary>展开 60 行「聚合 × 正常校准集 × 阈值」覆盖明细（仅核对数据完整性，不是性能结论）</summary>

### 聚合 × 正常校准集 × 阈值覆盖（60 行）

| phase | aggregation | calibration_set | quantile | runs | rows |
| --- | --- | --- | --- | --- | --- |
| P1_indomain_screening | max | training_normal | 0.9500 | 8.0000 | 8.0000 |
| P1_indomain_screening | max | training_normal | 0.9900 | 8.0000 | 8.0000 |
| P1_indomain_screening | max | validation_normal | 0.9500 | 8.0000 | 8.0000 |
| P1_indomain_screening | max | validation_normal | 0.9900 | 8.0000 | 8.0000 |
| P1_indomain_screening | mean | training_normal | 0.9500 | 8.0000 | 8.0000 |
| P1_indomain_screening | mean | training_normal | 0.9900 | 8.0000 | 8.0000 |
| P1_indomain_screening | mean | validation_normal | 0.9500 | 8.0000 | 8.0000 |
| P1_indomain_screening | mean | validation_normal | 0.9900 | 8.0000 | 8.0000 |
| P1_indomain_screening | top5pct | training_normal | 0.9500 | 8.0000 | 8.0000 |
| P1_indomain_screening | top5pct | training_normal | 0.9900 | 8.0000 | 8.0000 |
| P1_indomain_screening | top5pct | validation_normal | 0.9500 | 8.0000 | 8.0000 |
| P1_indomain_screening | top5pct | validation_normal | 0.9900 | 8.0000 | 8.0000 |
| P2_condition | max | training_normal | 0.9500 | 24.0000 | 24.0000 |
| P2_condition | max | training_normal | 0.9900 | 24.0000 | 24.0000 |
| P2_condition | max | validation_normal | 0.9500 | 24.0000 | 24.0000 |
| P2_condition | max | validation_normal | 0.9900 | 24.0000 | 24.0000 |
| P2_condition | mean | training_normal | 0.9500 | 24.0000 | 24.0000 |
| P2_condition | mean | training_normal | 0.9900 | 24.0000 | 24.0000 |
| P2_condition | mean | validation_normal | 0.9500 | 24.0000 | 24.0000 |
| P2_condition | mean | validation_normal | 0.9900 | 24.0000 | 24.0000 |
| P2_condition | top5pct | training_normal | 0.9500 | 24.0000 | 24.0000 |
| P2_condition | top5pct | training_normal | 0.9900 | 24.0000 | 24.0000 |
| P2_condition | top5pct | validation_normal | 0.9500 | 24.0000 | 24.0000 |
| P2_condition | top5pct | validation_normal | 0.9900 | 24.0000 | 24.0000 |
| P3_P4_chemistry | max | training_normal | 0.9500 | 16.0000 | 16.0000 |
| P3_P4_chemistry | max | training_normal | 0.9900 | 16.0000 | 16.0000 |
| P3_P4_chemistry | max | validation_normal | 0.9500 | 16.0000 | 16.0000 |
| P3_P4_chemistry | max | validation_normal | 0.9900 | 16.0000 | 16.0000 |
| P3_P4_chemistry | mean | training_normal | 0.9500 | 16.0000 | 16.0000 |
| P3_P4_chemistry | mean | training_normal | 0.9900 | 16.0000 | 16.0000 |
| P3_P4_chemistry | mean | validation_normal | 0.9500 | 16.0000 | 16.0000 |
| P3_P4_chemistry | mean | validation_normal | 0.9900 | 16.0000 | 16.0000 |
| P3_P4_chemistry | top5pct | training_normal | 0.9500 | 16.0000 | 16.0000 |
| P3_P4_chemistry | top5pct | training_normal | 0.9900 | 16.0000 | 16.0000 |
| P3_P4_chemistry | top5pct | validation_normal | 0.9500 | 16.0000 | 16.0000 |
| P3_P4_chemistry | top5pct | validation_normal | 0.9900 | 16.0000 | 16.0000 |
| P5_schema | max | training_normal | 0.9500 | 15.0000 | 15.0000 |
| P5_schema | max | training_normal | 0.9900 | 15.0000 | 15.0000 |
| P5_schema | max | validation_normal | 0.9500 | 15.0000 | 15.0000 |
| P5_schema | max | validation_normal | 0.9900 | 15.0000 | 15.0000 |
| P5_schema | mean | training_normal | 0.9500 | 15.0000 | 15.0000 |
| P5_schema | mean | training_normal | 0.9900 | 15.0000 | 15.0000 |
| P5_schema | mean | validation_normal | 0.9500 | 15.0000 | 15.0000 |
| P5_schema | mean | validation_normal | 0.9900 | 15.0000 | 15.0000 |
| P5_schema | top5pct | training_normal | 0.9500 | 15.0000 | 15.0000 |
| P5_schema | top5pct | training_normal | 0.9900 | 15.0000 | 15.0000 |
| P5_schema | top5pct | validation_normal | 0.9500 | 15.0000 | 15.0000 |
| P5_schema | top5pct | validation_normal | 0.9900 | 15.0000 | 15.0000 |
| P6_temporal | max | training_normal | 0.9500 | 90.0000 | 90.0000 |
| P6_temporal | max | training_normal | 0.9900 | 90.0000 | 90.0000 |
| P6_temporal | max | validation_normal | 0.9500 | 90.0000 | 90.0000 |
| P6_temporal | max | validation_normal | 0.9900 | 90.0000 | 90.0000 |
| P6_temporal | mean | training_normal | 0.9500 | 90.0000 | 90.0000 |
| P6_temporal | mean | training_normal | 0.9900 | 90.0000 | 90.0000 |
| P6_temporal | mean | validation_normal | 0.9500 | 90.0000 | 90.0000 |
| P6_temporal | mean | validation_normal | 0.9900 | 90.0000 | 90.0000 |
| P6_temporal | top5pct | training_normal | 0.9500 | 90.0000 | 90.0000 |
| P6_temporal | top5pct | training_normal | 0.9900 | 90.0000 | 90.0000 |
| P6_temporal | top5pct | validation_normal | 0.9500 | 90.0000 | 90.0000 |
| P6_temporal | top5pct | validation_normal | 0.9900 | 90.0000 | 90.0000 |

</details>

现阶段可检验、但尚未确认的候选现象包括：跨工况 FPR 变化、跨体系 FPR 变化、输入变量设置差异、时间尺度敏感性和故障类型差异。它们应并列保留，通过固定划分的多 seed 实验、目标域正常重校准和外部数据验证比较，而不能在当前阶段选定任何一个作为最终问题。

## R11：目标域正常重标定（固定模型、无重训）

协议：每个跨工况单元固定已保存 raw score 与模型；目标域正常 VIN 按固定 `calibration_vin_seed=3407` 分为 50% 阈值校准、50% 留出正常测试，全部故障保留在测试集。source-training P99 与 target-normal P99 在**同一留出测试集**上比较。AP/AUROC 只由分数排序决定，改阈值后保持不变；FPR/Recall 随阈值改变。

**解释边界：**目标域重标定（包括用 POT 等方法替换固定 P99）是必要的部署基线与诊断对照，**不是本报告单独的研究贡献**。它只能修正分数尺度/阈值错配，不能提升 AP；值得研究的是在无标签、分布持续变化且目标域正常数据有限时，如何可靠地完成该校准，以及校准后排序退化是否仍存在。

完整逐单元结果：[CSV](ch_batterygen_p1_p7_data/P2_R11_target_normal_recalibration_paired_rows.csv)；33 张 source/target 正常分数分布图保存在 `runs/research/target_normal_recalibration_r11/normal_score_distributions/`。

<details>
<summary>展开逐模型 / seed 完整明细</summary>

#### 明细：mtad_gat（seed 3407.0000）

| chemistry | setting | method | model_seed | threshold source→target | AP (unchanged) | FPR source→target | Recall source→target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LFP | p3_lfp_charge_to_discharge | mtad_gat | 3407.0000 | 0.00676602 → 0.0626761 | 0.8541 | 0.9025 → 0.0225 | 0.9810 → 0.1030 |
| LFP | p3_lfp_discharge_to_charge | mtad_gat | 3407.0000 | 0.00512145 → 0.0303165 | 0.7780 | 0.0950 → 0.0200 | 0.1329 → 0.0725 |
| NCM | p3_ncm_charge_to_discharge | mtad_gat | 3407.0000 | 0.00189697 → 0.0150063 | 0.7668 | 0.9275 → 0.0175 | 0.9130 → 0.0400 |
| NCM | p3_ncm_discharge_to_charge | mtad_gat | 3407.0000 | 0.00232306 → 0.00294123 | 0.8296 | 0.1576 → 0.0136 | 0.4002 → 0.2242 |

#### 明细：mtad_gat（seed 3408.0000）

| chemistry | setting | method | model_seed | threshold source→target | AP (unchanged) | FPR source→target | Recall source→target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LFP | p3_lfp_charge_to_discharge | mtad_gat | 3408.0000 | 0.00586786 → 0.0606919 | 0.8450 | 0.9275 → 0.0200 | 0.9820 → 0.0910 |
| LFP | p3_lfp_discharge_to_charge | mtad_gat | 3408.0000 | 0.0052868 → 0.042559 | 0.7899 | 0.1175 → 0.0175 | 0.1682 → 0.0594 |
| NCM | p3_ncm_charge_to_discharge | mtad_gat | 3408.0000 | 0.00182283 → 0.0178261 | 0.7470 | 0.9600 → 0.0175 | 0.9430 → 0.0430 |
| NCM | p3_ncm_discharge_to_charge | mtad_gat | 3408.0000 | 0.00224076 → 0.00314046 | 0.8141 | 0.2772 → 0.0245 | 0.4260 → 0.2220 |

#### 明细：mtad_gat（seed 3409.0000）

| chemistry | setting | method | model_seed | threshold source→target | AP (unchanged) | FPR source→target | Recall source→target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LFP | p3_lfp_charge_to_discharge | mtad_gat | 3409.0000 | 0.00595034 → 0.063856 | 0.8529 | 0.9275 → 0.0225 | 0.9860 → 0.1110 |
| NCM | p3_ncm_charge_to_discharge | mtad_gat | 3409.0000 | 0.00189382 → 0.0187128 | 0.7378 | 0.9675 → 0.0225 | 0.9450 → 0.0400 |
| NCM | p3_ncm_discharge_to_charge | mtad_gat | 3409.0000 | 0.00233675 → 0.00310248 | 0.8229 | 0.1848 → 0.0082 | 0.3632 → 0.2063 |

#### 明细：patchad（seed 3407.0000）

| chemistry | setting | method | model_seed | threshold source→target | AP (unchanged) | FPR source→target | Recall source→target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LFP | p3_lfp_charge_to_discharge | patchad | 3407.0000 | 3.22668 → 3.7504 | 0.6641 | 0.8325 → 0.0150 | 0.8970 → 0.0010 |
| LFP | p3_lfp_discharge_to_charge | patchad | 3407.0000 | 3.52331 → 3.63714 | 0.6398 | 0.1400 → 0.0175 | 0.0735 → 0.0282 |
| NCM | p3_ncm_charge_to_discharge | patchad | 3407.0000 | 3.69759 → 3.91573 | 0.7170 | 0.9350 → 0.0125 | 0.9560 → 0.0110 |
| NCM | p3_ncm_discharge_to_charge | patchad | 3407.0000 | 5.50976 → 5.36056 | 0.6958 | 0.0000 → 0.0408 | 0.0011 → 0.0269 |

#### 明细：patchad（seed 3408.0000）

| chemistry | setting | method | model_seed | threshold source→target | AP (unchanged) | FPR source→target | Recall source→target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LFP | p3_lfp_charge_to_discharge | patchad | 3408.0000 | 2.65293 → 3.25458 | 0.8057 | 0.9700 → 0.0025 | 1.0000 → 0.0160 |
| LFP | p3_lfp_discharge_to_charge | patchad | 3408.0000 | 4.85516 → 4.97189 | 0.6515 | 0.1050 → 0.0275 | 0.0554 → 0.0191 |
| NCM | p3_ncm_charge_to_discharge | patchad | 3408.0000 | 5.42323 → 5.93883 | 0.7132 | 0.9750 → 0.0200 | 0.9470 → 0.0220 |
| NCM | p3_ncm_discharge_to_charge | patchad | 3408.0000 | 5.75146 → 5.7617 | 0.6526 | 0.0109 → 0.0082 | 0.0045 → 0.0034 |

#### 明细：patchad（seed 3409.0000）

| chemistry | setting | method | model_seed | threshold source→target | AP (unchanged) | FPR source→target | Recall source→target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LFP | p3_lfp_charge_to_discharge | patchad | 3409.0000 | 1.86549 → 2.13038 | 0.8327 | 0.5850 → 0.0100 | 0.9570 → 0.0110 |
| NCM | p3_ncm_charge_to_discharge | patchad | 3409.0000 | 5.74829 → 6.26601 | 0.7389 | 0.9950 → 0.0000 | 1.0000 → 0.0150 |
| NCM | p3_ncm_discharge_to_charge | patchad | 3409.0000 | 3.01601 → 2.9941 | 0.7890 | 0.0027 → 0.0136 | 0.0045 → 0.0179 |

#### 明细：tranad（seed 3407.0000）

| chemistry | setting | method | model_seed | threshold source→target | AP (unchanged) | FPR source→target | Recall source→target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LFP | p3_lfp_charge_to_discharge | tranad | 3407.0000 | 0.00117437 → 0.113656 | 0.8122 | 0.9900 → 0.0200 | 1.0000 → 0.0520 |
| LFP | p3_lfp_discharge_to_charge | tranad | 3407.0000 | 0.00067928 → 0.0204912 | 0.7779 | 0.1350 → 0.0175 | 0.2113 → 0.0654 |
| NCM | p3_ncm_charge_to_discharge | tranad | 3407.0000 | 0.00104208 → 0.0925667 | 0.6703 | 0.9975 → 0.0025 | 0.9970 → 0.0160 |
| NCM | p3_ncm_discharge_to_charge | tranad | 3407.0000 | 0.000753801 → 0.00113504 | 0.8656 | 0.2337 → 0.0136 | 0.4787 → 0.2825 |

#### 明细：tranad（seed 3408.0000）

| chemistry | setting | method | model_seed | threshold source→target | AP (unchanged) | FPR source→target | Recall source→target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LFP | p3_lfp_charge_to_discharge | tranad | 3408.0000 | 0.00133187 → 0.108576 | 0.8187 | 0.9900 → 0.0200 | 1.0000 → 0.0610 |
| LFP | p3_lfp_discharge_to_charge | tranad | 3408.0000 | 0.000745106 → 0.0207547 | 0.8011 | 0.1250 → 0.0200 | 0.2294 → 0.0674 |
| NCM | p3_ncm_charge_to_discharge | tranad | 3408.0000 | 0.00105273 → 0.093162 | 0.6737 | 1.0000 → 0.0025 | 0.9960 → 0.0170 |
| NCM | p3_ncm_discharge_to_charge | tranad | 3408.0000 | 0.000810995 → 0.0010875 | 0.7528 | 0.1196 → 0.0245 | 0.1760 → 0.0561 |

#### 明细：tranad（seed 3409.0000）

| chemistry | setting | method | model_seed | threshold source→target | AP (unchanged) | FPR source→target | Recall source→target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LFP | p3_lfp_charge_to_discharge | tranad | 3409.0000 | 0.00142557 → 0.120642 | 0.8035 | 0.9875 → 0.0150 | 0.9990 → 0.0380 |
| NCM | p3_ncm_charge_to_discharge | tranad | 3409.0000 | 0.00103576 → 0.0968819 | 0.6679 | 1.0000 → 0.0025 | 0.9980 → 0.0170 |
| NCM | p3_ncm_discharge_to_charge | tranad | 3409.0000 | 0.000923689 → 0.00174741 | 0.7470 | 0.3587 → 0.0272 | 0.3173 → 0.0852 |

`charge→discharge` 的 source P99 FPR 在多数单元接近 1，而 target-normal P99 后接近设计的 1% 水平，支持存在显著的阈值/分数标定偏移。该现象不证明表示一定充分：AP 较低的单元在阈值重标定后仍是较弱的排序器，需与表征迁移问题分开讨论。

</details>

## 阈值控制后的错误归因：FP vs TN、FN vs TP（不重训）

已对 P2、P3/P4、P5 的 **55 个已保存运行**做完该归因，复用了 10,596 个原始 CSV 段及保存的 sample score。每个运行仍采用 VIN 隔离的目标域评估集；分别在静态 POT 与目标域正常 P99 下比较：正常段的 `FP vs TN`，以及**同一 fault type 内**的 `FN vs TP`。后者避免把“某种故障本来更难”误写成状态变量效应。比较变量为 CURRENT/SOC、水温/电压极值差、`|dI/dt|`、`|dSOC/dt|`、序列长度；Mann–Whitney 与高状态区间 Fisher 检验均 `p<.05` 才计为一个重复的显著分层。

下面只放最能重复的 P2、**目标域正常 P99** 结果；`比值`为“错误组中位数 / 正确组中位数”，`重复`是显著的 fault/迁移分层数占可比较分层数。完整逐运行、逐 fault-type、逐阈值表见 [错误归因明细](ch_batterygen_threshold_error_attribution/all_stratified_error_attribution.csv) 与 [模型汇总](ch_batterygen_threshold_error_attribution/model_phase_error_attribution_summary.csv)。

| 模型 | FP vs TN：目标域正常仍会在哪些合法状态误报 | FN vs TP：校准后仍漏掉什么故障段 | 可支持的解释 |
| --- | --- | --- | --- |
| DCdetector | FP 很少，未出现跨方向稳定的动态富集（最多 `SOC_min`，重复 1/5） | `SOC_min` 较低，**4/8**；其余状态不稳定 | 目前更像某些 SOC 区域的可分性不足，不能声称统一的高动态误报机制 |
| MTAD-GAT | **CURRENT 中位绝对值 3.85×（5/5）**、`|dSOC/dt|` **2.45×（5/5）**、`|dI/dt|` 2.53×（3/5） | cell-voltage spread 较小，**15/18**；电流较小，14/18；`|dSOC/dt|` 较小，14/18 | 图耦合/预测残差把高动态正常工况当作异常；同时对低动态、低电压分离度故障缺乏证据 |
| PatchAD | `|dI/dt|` **2.61×（4/6）**，P95 `|dI/dt|` 1.70×（4/6） | `|dSOC/dt|`、电流均明显较小（各 **9/13**），`|dI/dt|` 较小（8/13） | patch 对高动态正常片段敏感；较平稳/弱幅度故障更易落入正常 patch 分数区 |
| TranAD | `|dSOC/dt|` **1.88×（3/4）**、`|dI/dt|` **2.41×（3/4）** | SUM_VOLTAGE range 较小（**12/15**）、`|dSOC/dt|` 较小（12/15）、cell-voltage spread 较小（12/15） | 历史窗口重构在动态上下文下有正常尾部；对幅度较弱、变化较慢的故障排序不足 |

**P3/P4 与 P5 的交叉核验：**体系迁移中最一致的是 TranAD 的 FN 位于较低温度跨度（9/11）、较低 P95 `|dI/dt|`（8/11）段；P5 中 PatchAD 的 FN 电流较小（11/13）、TranAD 的 FN `|dI/dt|`/P95 `|dI/dt|` 都较小（各 14/14）。这支持“弱动态/弱幅度故障重叠”是可继续研究的候选，而不是简单的 LFP 或 NCM 谁更难。相反，DCdetector 的很多分层样本不足或方向不稳定，现阶段应保留为**未定论**。

**边界：**目标域 P99 把 FPR 压到约 1%，所以一些单元只剩 0–2 个 FP，不能以“不显著”反证不存在状态依赖；静态 POT 下的 FP 模式也已保存在明细中，但它仍混有源—目标分数尺度错配。这里报告的是**阈值控制后仍可见的错误条件**，不是模型内部因果机制；下一步若要从假设走向机制，需要导出逐变量残差、注意力或 patch 位置，而不是再堆总指标。

## 当前可支持的模型缺陷假设（不是最终论文结论）

下表把“实验现象”与“模型原理所导出的解释假设”严格分开。前者来自当前已保存 score；后者必须再用多 seed、按工况分层的错误归因验证，不能把结构名称当成因果证明。

| 模型 | 原理上最可能的脆弱点 | 当前证据 | 下一步应验证什么 |
| --- | --- | --- | --- |
| Isolation Forest | 用静态特征空间中的随机切分估计低密度，不建模时间顺序或跨变量动态关系；工况改变时，合法高动态状态可落入源域稀疏叶节点 | `NCM→LFP` 的 597 个 FP 集中在高 `|dI/dt|`、高 `|dSOC/dt|` 正常段 | 在每个目标工况内比较 FP/TN 的状态变量与 score 条件分布；检验状态分箱后 FPR 是否仍偏高 |
| MTAD-GAT | 图注意力与预测/重构残差依赖训练域的变量耦合；若 CURRENT、SOC 与响应变量的正常关系随工况改变，残差尺度和排序都可能失效 | NCM `charge→discharge` 的 AUPRC 下降 0.253；目标域 P99 后 FP 在高电流/高动态正常段重复出现，FN 多为低 cell-voltage spread 故障 | 导出边注意力或逐变量预测残差，检验高动态正常段的哪条边/哪一变量主导 score |
| TranAD | Transformer 重构/自条件分数依赖历史窗口的上下文分布；对工况整体平移可能保留部分排序，却使分数绝对尺度错配 | 跨工况常见极高源 P99 FPR；归因中 FP 偏高动态、FN 偏低 voltage range/低动态 | 分解每变量、每时间位置的重构误差；比较 responses-only / controls-only 下正常尾部与故障分数重叠 |
| PatchAD | 将序列切为局部 patch 后建模，patch 尺度与目标段长度、瞬态位置不匹配时，局部异常分数易失真；窗口/尺度选择也会改变有效上下文 | P6 中不同窗口的表现和 seed 波动较大；FP 偏高 `|dI/dt|`，FN 偏低电流/低动态 | 将 FP/FN 映射到 patch 内相对位置与序列长度，检验短段、边界 patch、高导数 patch 是否富集 |
| DCdetector | 双视图差异检测倚赖不同表示之间可对齐且互补；若关键变量移除或工况让两个视图同时改变，差异信号会变弱或不稳定 | P5 移除温度极值后 AUPRC 从约 0.714 降至 0.447；部分工况方向 Recall 很低 | 分别记录两视图分数与差异分数，做温度极值和工况的条件消融，而非只看总 AUPRC |

**本表结论：**当前最具体、最可验证的共性问题不是“模型都迁移差”，而是**正常数据的分数分布具有工况/动态状态依赖性；全局源域阈值会误报高动态正常状态，而目标域重标定后仍暴露正常—故障重叠。** 这比“换数据集性能下降”更接近可研究的问题。

## 数据与协议缺口

| 项目 | 当前状态 | 对解释的影响 |
|---|---|---|
| P2/P3/P4/P5 多 seed | 未完成 | 单 seed 不能估计随机训练方差。 |
| 目标域正常重校准 | P2/P3-P4/P5 的固定 score 已完成 VIN 隔离阈值对照 | 仍须按每个具体工况/模型报告，不能只看阶段均值。 |
| P5 的模型、体系覆盖 | 不完整 | 不能推广变量依赖结论。 |
| P6 配置生效性 | TranAD lookback 有待核对 | 不应直接声称窗口长度鲁棒。 |
| CH 标签粒度 | 片段级 | 不可报告真实故障发生后的点级检测延迟。 |
| 外部验证 | Brand/CALCE/ESS 未纳入本报告主结论 | 不能断言 CH 上的现象可泛化到真实车辆或实验室退化。 |

## 可追溯源

- `runs/research/FIRST_ROUND_ORIGINAL_ZIP_AUDIT.json`：早期第一轮 63/63 单元完整性审计。
- `runs/internal/research_06_ch_batterygen_temporal_scale/lfp/FORMAL_TEMPORAL*_AUDIT.json`：P6 四个模型的时间尺度 artifact 审计。
- `runs/research/P7_RAW_SCORE_ANALYSIS/P7_INPUT_AUDIT.json`：P7 行数与输入运行审计。
