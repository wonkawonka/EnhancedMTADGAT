# 电池异常检测问题发现预实验

本目录独立于 C3/C4 正式实验矩阵，用于在电池数据上比较不同范式异常检测模型，定位可复现的失效模式，再决定模型设计。CH-BatteryGen 用于受控问题发现，真实车辆数据用于外部确认，ESS 仅用于无标签正常运行稳定性压力测试。

所有结果在完成协议核验、三随机种子复现及外部电池数据验证前，不得写入论文最终实验矩阵。

配置入口为 [00_program_manifest.json](00_program_manifest.json)。`01` 是已运行的 LFP MTAD-GAT 基线；`02`–`07` 是 CH-BatteryGen 统一协议计划；`08`–`10` 是外部确认计划。

## 数据集角色

| 数据集 | 数据与标签粒度 | 作用 | 覆盖维度 |
|---|---|---|---|
| CH-BatteryGen LFP | 片段级正常/三类故障；LFP 有故障细节与严重度信息 | 主受控数据集 | Baseline、Condition、Schema、Temporal scale、Fault morphology、Severity、Calibration |
| CH-BatteryGen NCM | 片段级正常/三类故障 | 化学体系迁移测试 | Baseline、Condition、Chemistry/domain、Schema、Temporal scale、Fault morphology、Calibration |
| NC Battery / DyAD | 真实车辆充电片段、车辆级故障标签 | 外部真实车辆确认 | Baseline、故障形态外部确认、Calibration |
| CALCE 异常退化数据 | 取决于最终可获得的标签粒度 | 实验室退化外部验证 | Temporal scale、微弱/渐变异常验证 |
| 自有 ESS/BMS | 当前无故障标签，且只有调频工况 | 真实无标签稳定性压力测试 | Calibration、正常报警稳定性；不做 Recall/F1 |

CH-BatteryGen 当前数据已就绪：

| 子集 | VIN | 放电片段 | 正常 | 高内阻 | 低容量 | 自放电 |
|---|---:|---:|---:|---:|---:|---:|
| LFP | 500 | 5,000 | 4,000 | 300 | 300 | 400 |
| NCM | 500 | 5,000 | 4,000 | 300 | 300 | 400 |

LFP 与 NCM 都有 charge/discharge 文件。`operation` 不能仅凭文件名假定存在；先审计 `CHARGE_STATUS` 与片段定义。若无法形成独立、可靠的 operation 样本组，Condition 实验只做 `charge ↔ discharge`。

## 模型矩阵

| 代号 | 模型 | 范式 | 在本研究中的作用 | 当前状态 |
|---|---|---|---|---|
| M1 | MTAD-GAT | 图注意力、预测与重构 | 检查变量依赖与固定图结构敏感性 | 已接入；LFP 首个基线已完成 |
| M2 | TranAD | Transformer 重构 | 标准深度重构路线 | 已接入 CH 样本级适配器 |
| M3 | DCdetector | 双注意力、对比自监督 | 检验对比学习鲁棒性 | 已接入 CH 协议 |
| M4 | PatchAD | 多尺度 patch、MLP-Mixer、对比学习 | 检验时间尺度与局部 patch 建模 | 已接入；正式三种子基线待完成 |
| S1 | Robust PCA 或 Isolation Forest | 非深度基线 | 排除复杂网络并非必要的情况 | 已接入 CH 适配器 |
| B1 | DyAD | 电池条件—响应建模 | 电池领域专用外部强基线 | 仅在 NC Battery/DyAD 协议中比较 |

规则：

- M1–M4 使用相同 VIN 划分、输入变量、缩放器和样本级聚合规则。
- 所有主模型仅使用正常样本训练；故障类别仅用于测试和错误分析。
- AUROC、AUPRC 是主排序指标。
- F1 只在训练正常集确定的阈值下报告。
- 测试标签求得的最佳 F1 必须标为 `oracle diagnostic`，不得作为主结果。

## 统一数据协议

- 输入变量：`SUM_VOLTAGE`、`SUM_CURRENT`、`SOC`、`MAX_CELL_VOLT`、`MIN_CELL_VOLT`、`MAX_TEMP`、`MIN_TEMP`。
- 正式统一适配器使用 VIN 隔离的 70%/10%/20% 正常 VIN 训练、验证、测试划分；故障 VIN 仅进入测试。`01` 是历史 80%/20% LFP pilot。
- 缩放器仅拟合训练正常数据。
- 每个模型都保存逐窗口 raw score；第一轮同时比较片段内 `max`、`mean`、`top 5%` 窗口分数聚合。主结果暂使用 top 5%，并以敏感性表检查结论是否依赖该选择。
- 阈值敏感性同时报告训练正常与验证正常样本校准的 P95/P99；主阈值为训练正常样本分数 P99。
- 正式验证集只含正常 VIN，不能定义 validation-optimal threshold。若另用标签选择最优阈值，只能标为 oracle diagnostic，不得参与主结果。
- 数据划分与训练正常缩放器固定使用 `split_seed=3407`；筛选阶段使用 `model_seed=3407`，确认阶段使用 `model_seed=3407/3408/3409`。模型 seed 不得改变 VIN 划分。此前将同一 seed 同时用于两者的运行仅作为联合扰动筛选诊断。
- 每个模型保存逐样本分数、VIN、化学体系、工况、故障类别、严重度、片段长度和窗口尺度。

## 按观察维度执行的实验计划

| 维度 | 数据与控制变量 | 训练/测试设置 | 主指标 | 失效解释 | 对应配置 |
|---|---|---|---|---|---|
| Baseline | LFP、NCM，VIN 隔离 | LFP→LFP；NCM→NCM；LFP+NCM→各自测试 | AUROC、AUPRC、P99-F1、P99-FPR | 基本检测能力与混合正常分布能力 | `01`、`02` |
| Condition | charge/discharge；operation 仅在标签可确认时加入 | charge→charge、charge→discharge、discharge→discharge、discharge→charge | ΔFPR、ΔAUPRC、分数分布距离 | context-dependent normality | `03` |
| Chemistry/domain | LFP、NCM | LFP→LFP、LFP→NCM、NCM→NCM、NCM→LFP | ΔAUPRC、ΔFPR、最差域指标 | domain shift | `04` |
| Schema | 7 个包级变量 | Full、删变量、随机掩码 10/25/50%、仅控制变量、仅响应变量 | robustness ratio | variable heterogeneity | `05` |
| Temporal scale | 窗口、步长、重采样率、观测片段长度 | lookback=8/16/32/64；stride=1/4/8；原采样/下采样 | AUPRC、按片段长度分组 Recall | scale mismatch | `06` |
| Fault morphology | 高内阻、低容量、自放电 | 不重训，按测试分数分组 | per-type Recall、AUPRC、分数分布 | 特定异常结构难检测 | `07` |
| Severity | LFP 官方故障严重度字段 | 不重训，按可用等级分组 | severity Recall、平均异常分数 | subtle anomaly failure | `07` |
| Calibration | 化学体系、工况 | 固定源域阈值；少量目标域正常样本重新校准；oracle 仅诊断 | threshold shift、FPR、Recall、阈值稳定性 | 异常分数不可迁移 | `07` |

\[
\Delta FPR = FPR_{\text{cross}} - FPR_{\text{same}}
\]

\[
R_{\text{schema}} = \frac{\mathrm{AUPRC}_{\text{masked}}}{\mathrm{AUPRC}_{\text{full}}}
\]

\[
\Delta \mathrm{AUPRC} = \mathrm{AUPRC}_{\text{in-domain}} - \mathrm{AUPRC}_{\text{cross-domain}}
\]

## 具体判据

### Baseline

先完成四个主模型的 LFP/NCM 域内基线，并在确认阶段各跑三组随机种子；S1 在两个体系各运行一次。只有模型能在域内合理工作，后续迁移失败才可解释。

### Condition

重点不是单一 F1，而是比较训练工况正常、未见工况正常、同工况故障三组分数。若未见工况正常样本分数接近故障样本，且多个模型均出现显著正的 ΔFPR，才形成“工况相关正常性”的候选问题。

### Chemistry/domain

固定相同 7 个变量、窗口长度和阈值协议。若跨体系时 AUROC 仍高但固定阈值 FPR 大幅增加，主要是校准迁移问题；若 AUROC/AUPRC 同时下降，才更可能是表征迁移失败。

### Schema

| 设置 | 变量处理 | 要回答的问题 |
|---|---|---|
| Full | 全部 7 个变量 | 完整信息基线 |
| Controls-only | 仅电流、SOC | 条件变量自身是否足够 |
| Responses-only | 去掉电流、SOC | 响应变量是否可独立检测 |
| Drop-voltage-extrema | 去掉最大/最小单体电压 | 局部电芯异常证据是否关键 |
| Drop-temperature-extrema | 去掉最大/最小温度 | 热异常信息是否关键 |
| Random-mask | 随机掩码 10/25/50% | 输入缺失鲁棒性 |

第一阶段不直接把 28-cell 与 124-cell 的全部单体电压作为跨包输入；先使用固定维度统计量，避免维度不兼容被误判为模型缺陷。

### Temporal scale

CH-BatteryGen 是片段级故障标签，因此可比较窗口尺度、重采样和片段观测长度，但不能称为真实故障持续时间或报告故障发生后的检测延迟。提前检测能力只在具备时间边界标签的 CALCE 或受控注入实验中验证。

### Fault morphology 与 Severity

故障类别不作为分类任务，而是异常形态探针。若多个模型都对低容量或自放电持续低召回，进一步检查偏差幅度、变化速度、受影响变量数、工况重叠，以及 top-k 聚合是否稀释异常证据。

Severity 仅在官方元数据覆盖充分的 LFP 子集上报告；先审计每个等级的样本数，等级定义不清或样本严重不平衡时不强行作 L1/L2/L3 结论。

### Calibration

| 设置 | 可用信息 | 用途 |
|---|---|---|
| Source-fixed | 源域训练正常样本 | 最接近零样本部署 |
| Target-normal recalibration | 独立目标域正常 VIN | 判断是否仅是阈值问题 |
| Oracle diagnostic | 测试标签 | 理论上限诊断，不计入主结果 |

若重新校准后表现基本恢复，应优先研究阈值与校准；若 AUPRC 仍低，则问题在异常评分或表征。

## 分阶段运行顺序

| 阶段 | 运行内容 | 目标 |
|---|---|---|
| P0 | LFP/NCM 数据审计、VIN 划分、charge/discharge 可用性审计 | 固定协议 |
| P1 | M1–M4 + S1 的 LFP/NCM 域内基线 | 确认模型基本能力 |
| P2 | 四个主模型的 Condition、Chemistry、Schema、Temporal 单 seed 筛选 | 找候选失效模式 |
| P3 | 最强的 1–2 个候选失效模式使用三 seed 复现 | 排除偶然结果 |
| P4 | Fault morphology、Severity、Calibration 分析 | 解释失败原因 |
| P5 | NC Battery/DyAD 外部确认，并加入 DyAD | 验证不是 CH 特例 |
| P6 | ESS/BMS 正常运行稳定性测试 | 验证真实无标签运行场景 |

P2 只用于筛选，不进入最终论文矩阵。P3 与 P5 完成后，才决定模型具体解决哪个缺陷。

## 当前状态

- P0：LFP/NCM 数据已就绪；LFP 有 10,000 个原始 CSV，NCM 已从官方数据包提取。
- M1：历史 LFP 80/20 pilot 已完成；现已接入正式统一 runner，尚待正式 70/10/20 运行。
- P1：历史 MTAD-GAT、TranAD、DCdetector 的三 seed 运行同时改变了模型 seed 与 VIN 划分，只能用作联合扰动诊断；固定 VIN 划分的正式确认待补。PatchAD 已接入。
- 当前项：固定 `split_seed=3407` 的 LFP 正式预处理包已生成并上传 Kaggle；计划 05/06 将以 `model_seed=3407/3408/3409` 重跑。所有统一运行保存逐窗口 raw score、三种片段聚合和正常样本 P95/P99 阈值敏感性结果。
