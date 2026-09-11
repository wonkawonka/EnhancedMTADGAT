下面按你这张表把“数据集—模型—实验矩阵”定成完整预实验计划。核心是：**CH-BatteryGen 做受控问题发现，真实车辆数据做外部确认，ESS 只做无标签稳定性压力测试。**

## 1. 数据集角色

| 数据集 | 数据与标签粒度 | 作用 | 覆盖维度 |
|---|---|---|---|
| CH-BatteryGen LFP | 片段级正常/三类故障；LFP 有故障细节与严重度信息 | 主受控数据集 | Baseline、Condition、Schema、Temporal scale、Fault morphology、Severity、Calibration |
| CH-BatteryGen NCM | 片段级正常/三类故障 | 化学体系迁移测试 | Baseline、Condition、Chemistry/domain、Schema、Temporal scale、Fault morphology、Calibration |
| NC Battery / DyAD 数据 | 真实车辆充电片段、车辆级故障标签 | 外部真实车辆确认 | Baseline、Fault morphology 的外部确认、Calibration |
| CALCE 异常退化数据 | 取决于最终可获得的标签粒度 | 实验室退化外部验证 | Temporal scale、微弱/渐变异常验证 |
| 自有 ESS/BMS | 当前无故障标签，且只有调频工况 | 真实无标签稳定性压力测试 | Calibration、正常报警稳定性；不做 Recall/F1 |

CH-BatteryGen 当前数据已就绪：

| 子集 | VIN | 放电片段 | 正常 | 高内阻 | 低容量 | 自放电 |
|---|---:|---:|---:|---:|---:|---:|
| LFP | 500 | 5,000 | 4,000 | 300 | 300 | 400 |
| NCM | 500 | 5,000 | 4,000 | 300 | 300 | 400 |

LFP 与 NCM 都有 charge/discharge 文件。`operation` 不能直接按文件名假定存在，需先审计 `CHARGE_STATUS` 和片段定义；若无法形成独立、可靠的 operation 样本组，Condition 实验只做 `charge ↔ discharge`，不虚构第三类工况。

## 2. 模型矩阵

第一批主比较保留四个不同范式模型，避免只比较相似 Transformer。

| 代号 | 模型 | 范式 | 在本研究中的作用 | 当前状态 |
|---|---|---|---|---|
| M1 | MTAD-GAT | 图注意力、预测与重构 | 变量依赖、固定图结构敏感性 | 已接入，LFP 首个基线已完成 |
| M2 | TranAD | Transformer 重构 | 标准深度重构路线 | 需接入 CH 样本级适配器 |
| M3 | DCdetector | 双注意力、对比自监督 | 检验对比学习的鲁棒性 | 代码已在仓库，需接入 CH 协议 |
| M4 | PatchAD | 多尺度 patch、MLP-Mixer、对比学习 | 检验时间尺度与局部 patch 建模 | 需接入 CH 样本级适配器 |
| S1 | Robust PCA 或 Isolation Forest | 非深度基线 | 排除“复杂网络不必要”的可能 | 后续接入 |
| B1 | DyAD | 电池条件—响应建模 | 电池领域专用外部强基线 | 仅在 NC Battery/DyAD 协议中比较 |

规则：

- M1–M4 使用完全相同的 VIN 划分、输入变量、缩放器和样本级聚合规则。
- 所有主模型训练仅使用正常样本。
- 故障类别仅用于测试与错误分析。
- AUROC、AUPRC 是主排序指标。
- F1 只在训练正常集确定的阈值下报告。
- 使用测试标签求得的最佳 F1 只能标记为 `oracle diagnostic`，不得作为主结果。

## 3. 统一数据协议

所有 CH-BatteryGen 主实验使用：

- 输入变量：`SUM_VOLTAGE`、`SUM_CURRENT`、`SOC`、`MAX_CELL_VOLT`、`MIN_CELL_VOLT`、`MAX_TEMP`、`MIN_TEMP`。
- 划分：按 VIN 隔离；正常 VIN 按 80/10/10 或 80/20 划分训练、校准、测试；故障 VIN 只进入测试。
- 缩放：仅拟合训练正常数据。
- 样本分数：一个片段内 top 5% 窗口异常分数的平均值。
- 主阈值：训练正常样本分数的 P99。
- 重复：筛选阶段使用 seed=3407；确认阶段使用 3407、3408、3409 三个种子。
- 输出：每个模型必须保存逐样本分数、VIN、化学体系、工况、故障类别、严重度、片段长度和窗口尺度。

## 4. 按观察维度执行的实验计划

| 维度 | 数据与控制变量 | 训练/测试设置 | 主指标 | 失效解释 |
|---|---|---|---|---|
| Baseline | LFP、NCM，VIN 隔离 | LFP→LFP；NCM→NCM；LFP+NCM→各自测试 | AUROC、AUPRC、P99-F1、P99-FPR | 基本检测能力与混合正常分布能力 |
| Condition | charge/discharge；operation 仅在标签可确认时加入 | charge→charge、charge→discharge、discharge→discharge、discharge→charge | \(\Delta FPR\)、\(\Delta AUPRC\)、分数分布距离 | context-dependent normality |
| Chemistry/domain | LFP、NCM | LFP→LFP、LFP→NCM、NCM→NCM、NCM→LFP | \(\Delta AUPRC\)、\(\Delta FPR\)、最差域指标 | domain shift |
| Schema | 7 个包级变量；后续可扩展单体电压特征 | Full、删 1 个变量、随机掩码 10/25/50%、仅控制变量、仅响应变量 | robustness ratio | variable heterogeneity |
| Temporal scale | 窗口、步长、重采样率、观测片段长度 | lookback=8/16/32/64；stride=1/4/8；原采样/下采样 | AUPRC、按片段长度分组 Recall | scale mismatch |
| Fault morphology | 高内阻、低容量、自放电 | 不重训，使用相同测试分数分组 | per-type Recall、AUPRC、分数分布 | 特定异常结构难检测 |
| Severity | LFP 官方故障严重度字段 | 不重训，按可用等级分组 | severity Recall、平均异常分数 | subtle anomaly failure |
| Calibration | 化学体系、工况分别校准 | 固定源域阈值；少量目标域正常样本重新校准；oracle 仅诊断 | threshold shift、FPR、Recall、阈值稳定性 | 异常分数不可迁移 |

其中：

\[
\Delta FPR = FPR_{\text{cross}} - FPR_{\text{same}}
\]

\[
R_{\text{schema}} =
\frac{\mathrm{AUPRC}_{\text{masked}}}
{\mathrm{AUPRC}_{\text{full}}}
\]

\[
\Delta \mathrm{AUPRC} =
\mathrm{AUPRC}_{\text{in-domain}}
-
\mathrm{AUPRC}_{\text{cross-domain}}
\]

## 5. 每个维度的具体判据

### Baseline

先完成：

- 4 个主模型 × LFP 域内 × 3 seeds；
- 4 个主模型 × NCM 域内 × 3 seeds；
- S1 简单基线在 LFP、NCM 各运行一次。

只有模型能在域内合理工作，后面的迁移失败才可解释。

### Condition

重点不只是 F1，而是三类分数是否重叠：

```text
训练工况正常样本
未见工况正常样本
同工况故障样本
```

若未见工况正常样本的分数接近故障样本，且多个模型均出现显著 \(\Delta FPR\)，才可形成“工况相关正常性”问题。

### Chemistry/domain

化学体系实验必须固定：

- 同一组 7 个包级变量；
- 相同窗口长度；
- 同一阈值确定协议；
- 不混合 LFP 与 NCM 的测试标签进行阈值选择。

若跨体系时 AUROC 仍高、但固定阈值下 FPR 大幅增加，问题主要是**校准迁移**；若 AUROC、AUPRC 同时下降，才更可能是**表征迁移失败**。

### Schema

不能只做随机删变量，还要保留变量语义。

| 设置 | 保留变量 | 要回答的问题 |
|---|---|---|
| Full | 全部 7 个变量 | 完整信息基线 |
| Controls-only | 电流、SOC | 条件变量自身是否足够 |
| Responses-only | 电压、温度、极值变量 | 响应变量是否可独立检测 |
| Drop-voltage-extrema | 移除最大/最小单体电压 | 局部电芯异常证据是否关键 |
| Drop-temperature-extrema | 移除最大/最小温度 | 热异常信息是否关键 |
| Random-mask | 随机掩码 10/25/50% | 输入缺失鲁棒性 |

对于 28-cell 与 124-cell 的差异，第一阶段不直接输入全部单体电压做跨包迁移；先用极值、极差、标准差等固定维度统计量，避免“维度不兼容”被误判为模型缺陷。

### Temporal scale

CH-BatteryGen 当前是片段级故障标签，不能把片段中的每个时间点称为真实异常点。因此：

- 可以比较不同窗口尺度；
- 可以按片段实际长度分组；
- 可以讨论“观测时间尺度不匹配”；
- 不能直接报告“故障发生后检测延迟”。

真正的异常持续时间与提前检测能力，应放到具备时间边界标签的 CALCE 或后续受控注入实验中验证。

### Fault morphology 与 Severity

故障类别不作为分类任务，而是异常形态探针。

例如，若多个模型都呈现：

| 故障 | 高内阻 | 低容量 | 自放电 |
|---|---:|---:|---:|
| P99 阈值 Recall | 高 | 很低 | 很低 |

则需要进一步检查低容量和自放电是否具有：

- 偏差幅度小；
- 持续变化慢；
- 只影响少量变量；
- 与正常工况强重叠；
- 被 top-k 样本聚合或全局重构误差稀释。

Severity 只在官方元数据覆盖充分的 LFP 故障子集上报告。先审计每个等级的样本数；若等级定义或样本量不平衡，不强行写 L1/L2/L3 对比结论。

### Calibration

每个模型、每个域至少做三组：

| 设置 | 可用信息 | 用途 |
|---|---|---|
| Source-fixed | 仅源域训练正常样本 | 最接近零样本部署 |
| Target-normal recalibration | 少量独立目标域正常样本 | 判断是否只是阈值问题 |
| Oracle diagnostic | 测试标签 | 仅估计理论上限，不计入主结果 |

若 target-normal recalibration 后性能基本恢复，应优先研究阈值与校准；若重新校准后 AUPRC 仍低，说明问题在异常评分或表征。

## 6. 分阶段运行顺序

| 阶段 | 运行内容 | 目标 |
|---|---|---|
| P0 | LFP/NCM 数据审计、VIN 划分、charge/discharge 可用性审计 | 固定协议 |
| P1 | M1–M4 + S1 的 LFP/NCM 域内基线 | 确认模型基本能力 |
| P2 | 4 个主模型在 CH 上做 Condition、Chemistry、Schema、Temporal screening，单 seed | 快速找候选失效模式 |
| P3 | 对最强的 1–2 个候选失效模式，所有 M1–M4 跑 3 seeds | 排除偶然结果 |
| P4 | Fault morphology、Severity、Calibration 分析 | 解释失败原因 |
| P5 | 在 NC Battery/DyAD 上做外部确认；加入 DyAD | 验证不是 CH 特例 |
| P6 | ESS/BMS 正常运行稳定性测试 | 验证真实无标签运行场景 |

P2 是筛选阶段，不把结果写入最终论文矩阵。P3、P5 完成后，才决定你的模型具体解决哪个缺陷。

目前已经完成的是 P0 的 LFP/NCM 数据就绪，以及 M1 的 LFP 单 seed 基线。下一项应是：**为 TranAD、DCdetector、PatchAD 建立统一的 CH-BatteryGen 样本级适配器，然后完成 P1 的四模型 LFP/NCM 域内基线。**