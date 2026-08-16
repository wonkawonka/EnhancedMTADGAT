# C3 受约束状态 FiLM 五数据集统一验证报告

日期：2026-08-16  
数据集：MSL、SMAP、清华 EV Brand3 fold1、SWaT、WADI  
实验性质：seed 3407、10 epochs、架构筛选；不是论文正式多 seed / 多折最终结果。

## 1. 统一结论

本轮五数据集结果用于界定冻结 C3 的适用边界。最终工程路线已冻结为
`RestrictedStateEncoder + fusion FiLM + 状态语义辅助损失`；这里的“冻结”指结构和
复现实验口径不再漂移，并不等于声称它在每个数据集都优于 baseline。

1. `film_true` 在 MSL、SMAP、清华 Brand3 fold1 的 AP 和 AUROC 均超过各自 baseline。
2. 清华上的状态语义证据最明显：`film_true` 相对 baseline 的 AP/AUROC 分别提高 `+0.204406/+0.142857`，也明显超过 shuffled 对照。
3. MSL、SMAP 的增益方向一致但幅度较小，真实状态相对 shuffled 的优势也较小，只能视为弱支持。
4. SWaT baseline 本身具有可用排序能力（AP `0.714338`、AUROC `0.823407`），但三个 FiLM 臂均未同时超过 baseline；真实状态不优于 shuffled。
5. WADI 四臂均接近随机排序。其核心问题是正常训练/验证工况与测试工况严重漂移，而不是训练未完成或数据文件损坏。
6. 因此冻结 C3 的论文表述必须限定为“状态语义明确时有明显收益、跨工业数据外推不稳定”，不能声称五数据集普遍提升。

## 2. 统一协议与口径

五个数据集采用相同的四臂定义：

| 实验臂 | 状态 FiLM | 状态辅助约束 | 状态配对 |
|---|---:|---:|---|
| baseline | 否 | 否 | — |
| film_no_aux | 是 | 否 | 真实 |
| film_true | 是 | 是 | 真实 |
| film_shuffled | 是 | 是 | 批内错配 |

共同设置：

- 骨干为 Conv1D + Feature-GAT + Temporal-GAT + GRU，不使用 Transformer。
- 状态表示为 8-D，只在三路融合表示后通过 FiLM 注入。
- seed `3407`，训练 `10 epochs`。
- MSL、SMAP、SWaT、WADI 使用原始逐点 AP/AUROC。
- 清华使用固定 Top-5% 聚合后的车辆级 AP/AUROC。
- AP/AUROC 可用于判断各数据集内部的模型排序，但清华车辆级指标不能和其余四个逐点指标直接比较绝对大小。

状态输入语义并不完全相同：

- 清华只使用电流、SOC 等明确控制量的受约束统计描述。
- MSL/SMAP 使用目标通道以外匿名上下文通道的置换不变池化描述。
- SWaT/WADI 当前没有明确的控制量映射，状态编码器使用全部输入通道；这会把受攻击的响应传感器也当作“工况”，是本轮工业数据结果的重要限制。

## 3. 五数据集统一主表

括号为相对同一数据集 baseline 的绝对增量。

| 数据集 | 实验臂 | AP | ΔAP | AUROC | ΔAUROC |
|---|---|---:|---:|---:|---:|
| MSL | baseline | 0.239876 | — | 0.684256 | — |
| MSL | film_no_aux | 0.231627 | -0.008249 | 0.667205 | -0.017050 |
| MSL | film_shuffled | 0.243120 | +0.003244 | 0.692399 | +0.008144 |
| MSL | **film_true** | **0.246150** | **+0.006274** | **0.705825** | **+0.021570** |
| SMAP | baseline | 0.141868 | — | 0.545149 | — |
| SMAP | film_no_aux | 0.141742 | -0.000126 | 0.549199 | +0.004049 |
| SMAP | film_shuffled | 0.149276 | +0.007408 | 0.595070 | +0.049921 |
| SMAP | **film_true** | **0.151236** | **+0.009368** | **0.604501** | **+0.059351** |
| 清华 Brand3 fold1 | baseline | 0.462771 | — | 0.634921 | — |
| 清华 Brand3 fold1 | film_no_aux | 0.586590 | +0.123820 | 0.714286 | +0.079365 |
| 清华 Brand3 fold1 | film_shuffled | 0.487812 | +0.025041 | 0.666667 | +0.031746 |
| 清华 Brand3 fold1 | **film_true** | **0.667177** | **+0.204406** | **0.777778** | **+0.142857** |
| SWaT | **baseline** | 0.714338 | — | **0.823407** | — |
| SWaT | film_no_aux | 0.706576 | -0.007762 | 0.818051 | -0.005356 |
| SWaT | film_shuffled | **0.716019** | +0.001682 | 0.822022 | -0.001385 |
| SWaT | film_true | 0.710065 | -0.004273 | 0.819026 | -0.004381 |
| WADI | **baseline** | 0.065525 | — | **0.505425** | — |
| WADI | film_no_aux | 0.047011 | -0.018514 | 0.445498 | -0.059926 |
| WADI | film_shuffled | **0.080005** | +0.014480 | 0.491908 | -0.013517 |
| WADI | film_true | 0.071349 | +0.005824 | 0.485986 | -0.019439 |

粗略胜负统计：

- `film_true` 同时提高 AP 和 AUROC：MSL、SMAP、清华，共 `3/5` 个数据集。
- `film_true` 同时低于 baseline：SWaT；WADI 虽 AP 略升，但 AUROC 下降且整体接近随机。
- `film_true` 同时优于 shuffled：MSL、SMAP、清华，共 `3/5`；SWaT/WADI 均不成立。
- 最强且最清晰的证据来自清华；最明确的失败来自 WADI。

## 4. 低误报工作点补充

SWaT/WADI 使用保存的原始全局异常分数，在测试 ROC 曲线上计算低误报召回。该指标用于描述排序能力，不是独立正常验证集校准阈值的部署结果。

| 数据集 | 实验臂 | Recall@FPR≤1% | Recall@FPR≤0.5% | Recall@FPR≤0.1% |
|---|---|---:|---:|---:|
| SWaT | baseline | 0.611192 | 0.599122 | 0.596745 |
| SWaT | film_no_aux | 0.599122 | 0.596745 | 0.595465 |
| SWaT | film_shuffled | 0.602780 | 0.598939 | 0.595465 |
| SWaT | film_true | **0.612838** | **0.607718** | **0.596379** |
| WADI | baseline | 0.010030 | 0.000000 | 0.000000 |
| WADI | film_no_aux | 0.000000 | 0.000000 | 0.000000 |
| WADI | film_shuffled | 0.054162 | **0.024072** | 0.000000 |
| WADI | film_true | **0.055165** | 0.007021 | 0.000000 |

SWaT 的低误报召回约为 `0.60`，说明 baseline 和 FiLM 都有实用检测信号；FiLM 的改善只出现在部分低误报工作点，没有转化为整体 AP/AUROC 增益。WADI 在 FPR≤0.1% 时四臂召回均为零。

## 5. 分数据集判断

### 5.1 MSL 与 SMAP：弱正向证据

- `film_true` 在两个数据集上都提高 AP/AUROC。
- 相对 `film_no_aux`，辅助状态约束在 MSL 提高 AP/AUROC `+0.014523/+0.038620`，在 SMAP 提高 `+0.009494/+0.055302`。
- `film_true` 相对 shuffled 仅小幅领先：MSL 为 `+0.003030/+0.013426`，SMAP 为 `+0.001960/+0.009431`。
- 方向一致，但单 seed 和小幅差距不足以证明模型稳定依赖正确的状态—窗口配对。

### 5.2 清华：最强状态语义证据

- `film_true` 相对 baseline 的 AP/AUROC 提高 `+0.204406/+0.142857`。
- 相对 shuffled 提高 `+0.179365/+0.111111`。
- 相对 no-aux 提高 `+0.080587/+0.063492`。
- 明确的电流/SOC 控制语义与车辆级故障差异使状态 FiLM 更容易形成有效条件化。
- 该表仅为 Brand3 fold1 快速验证；后续五折结果显示 FiLM 平均并不稳定，因此不能把 fold1 增益当作最终结论。

### 5.3 SWaT：baseline 可用，FiLM 无稳定收益

- baseline AP/AUROC 为 `0.714338/0.823407`，明显高于随机基线；异常点占比约 `0.1216`。
- 正常点平均异常分数为 `0.2648`，异常点为 `0.7819`，排序方向正确。
- shuffled 获得最高 AP，但 AUROC 仍略低于 baseline；`film_true` 的 AP/AUROC 也均低于 baseline。
- 真实状态没有超过 shuffled，说明当前“全部通道作为状态输入”的工业数据配置没有提供可信状态语义。

### 5.4 WADI：分布漂移主导，当前结果不可用于模型主张

- baseline AP/AUROC 仅为 `0.065525/0.505425`；测试异常占比约 `0.0577`，AP 接近随机水平。
- WADI baseline 验证总损失为 `0.11856`，测试总损失达到 `41.02634`，约为验证损失的 `346` 倍。
- 测试正常点平均异常分数为 `3.4388`，真实异常点反而只有 `2.5091`，大量正常工况漂移压过了攻击信号。
- processed 数据没有 NaN、全 NaN 列或常数训练列，因此不是下载或序列化损坏。
- WADI 失败首先应归因于正常工况分布漂移、逐传感器校准和状态通道语义缺失；继续增加 epoch 不能直接解决问题。

## 6. 与已归档关系分支的边界

本报告主表只比较同一 restricted-state FiLM 四臂，不把不同架构混入胜负统计。

既有快速实验还表明：

- 旧 relation-only / 二维联合高斯 NLL 在 MSL、SMAP 明显降低 AP/AUROC，并在清华抹掉 FiLM 的主要收益；该版本已被否定。
- 后续 Dual-relation 五折属于另一条历史结构。其清华指标曾高于 baseline，但该结构不属于用户最终确认的 C3 三模块，现已归档，不能替换冻结路线。
- 因此本报告保留该结果只为完整披露模型选择历史；当前复现、补折和论文结构图均以 restricted-state FiLM 为准。

## 7. 最终冻结决策

1. C3 结构冻结为 restricted-state encoder、fusion FiLM、状态语义辅助损失；不再切换到 Dual-relation 或其他历史候选。
2. 四臂负对照固定为 baseline、film_no_aux、film_true、film_shuffled；评价时必须完整报告失败数据集，不能只汇报正向结果。
3. MSL/SMAP 记录为单 seed 弱正向证据；清华 fold1 是强正向但仍需结合多折方差解释。
4. SWaT baseline 有效而 C3 无稳定整体增益；WADI 受分布漂移主导。这两项是冻结路线的适用边界，不再通过更换模块规避。
5. 旧 relation/joint-NLL、Dual-relation 和其他候选只保留在历史归档中，不进入最终结构或活动实验计划。

## 8. 结果来源

- MSL/SMAP/清华同路线报告：`report/analysis/c3_restricted_state_three_dataset_validation_20260816.md`
- 后续 C3 清理汇总：`report/analysis/86_c3_results_clean_summary.md`
- SWaT/WADI 指标：`runs/kaggle_downloads/swat_wadi_v8/evaluation/swat_wadi_metrics.csv`
- SWaT/WADI 完整 JSON：`runs/kaggle_downloads/swat_wadi_v8/evaluation/swat_wadi_metrics.json`
- SWaT/WADI 单模型输出：`runs/kaggle_downloads/swat_wadi_v8/evaluation/runs/manual/`
- 最终架构说明：`report/final_c3_c4_design.md`
