# 简化版 C3 原型查询式工况调制网络：最新方法与实验协议

日期：2026-08-20  
适用数据集：MSL、SMAP、清华 EV、BMS、SWAT、WADI  
文档性质：小论文方法与实验章节底稿；未完成的数值统一标记为“待填”

> 本文只讨论当前冻结的简化 C3：全通道统计 Token、原型 Query 交叉注意力、原型输出
> 拼接投影和融合层 FiLM。最新实现已删除 Top-2、全局匹配、归一化软路由、连续残差分解、
> 复杂状态拼接和原型防坍缩损失。旧复杂版实验不能直接作为当前版本结果；文中的误报改善
> 属于待检验研究假设，不预写成实验结论。

## 1 研究动机与总体架构

原始 MTAD-GAT 通过卷积、特征图注意力、时间注意力和 GRU 建模多变量时间序列，并利用
预测残差和重构残差形成异常分数。它能够学习变量关系和时间依赖，但缺少显式窗口工况
条件：所有正常窗口主要由同一套参数解释。航天器、电动汽车和储能系统的正常分布却会随
工作阶段、负载、SOC 和温度变化。若模型不能区分正常工况变化，工况切换产生的分布偏移
就可能被误判为异常。

给定窗口

$$
X=[x_{t,c}]\in\mathbb{R}^{T\times C},
$$

C3 增加一条轻量状态旁路：

$$
X\rightarrow H_{\mathrm{stat}}
\rightarrow z_{\mathrm{regime}}
\rightarrow H_{\mathrm{mod}}.
$$

其中，$H_{\mathrm{stat}}$ 是全通道统计 Token，$z_{\mathrm{regime}}\in\mathbb{R}^{8}$
是窗口级连续工况，$H_{\mathrm{mod}}$ 是经过工况条件化的主干融合表示。状态旁路不直接
输出异常分数，最终评分仍由原预测和重构残差产生。

三个核心改进为：全通道统计 Token 化、原型 Query 交叉注意力与输出拼接、融合层 FiLM
状态调制。统计描述重建是唯一保留的 C3 辅助训练约束，不是推理期附加分支。

## 2 改进一：全通道统计 Token 化

### 2.1 通道统计描述

工况分支不再复制主干完成逐时刻编码，而是针对每个通道提取四个窗口统计量：

$$
\mu_c=\frac{1}{T}\sum_{t=1}^{T}x_{t,c},
$$

$$
\sigma_c=\sqrt{\frac{1}{T}\sum_{t=1}^{T}(x_{t,c}-\mu_c)^2+\varepsilon},
$$

$$
\delta_c=x_{T,c}-x_{1,c},
$$

$$
v_c=\frac{1}{T-1}\sum_{t=2}^{T}|x_{t,c}-x_{t-1,c}|.
$$

通道描述为

$$
d_c=[\mu_c,\sigma_c,\delta_c,v_c]\in\mathbb{R}^{4}.
$$

四项分别描述水平、波动、总体变化方向和局部变化活跃度。均值和标准差不足以区分稳定
静置与频繁调节；首尾变化量保留方向，平均绝对一阶差分避免正负变化抵消，因此四项组合
更适合描述充电、放电、静置、负载变化和高频调节等窗口状态。

### 2.2 通道 Token

共享映射把四维描述投影到 $d=32$ 维：

$$
\widetilde h_c=
\operatorname{LN}\left(\operatorname{GELU}(W_d d_c+b_d)\right).
$$

为保留通道身份，加入可学习通道嵌入：

$$
h_c=\widetilde h_c+e_c.
$$

全部通道组成

$$
H_{\mathrm{stat}}=[h_1,\ldots,h_C]^{\mathsf T}
\in\mathbb{R}^{C\times d}.
$$

通道嵌入类似位置编码，使后续 Query 能够学习“哪些变量构成当前工况”。当前提出的
prototype-query C3 对所有数据集统一使用完整输入：MSL 55 个 Token、SMAP 25 个、清华
EV 7 个、BMS 35 个；SWAT/WADI 则使用预处理 schema 的全部通道。restricted C3 作为历史
对照，在 MSL/SMAP 上仍使用排除目标通道后的池化上下文。该模块把工况旁路复杂度从依赖
全部 $T\times C$ 采样点，压缩为主要依赖通道数 $C$，避免与主干时间编码重复。

C3 输入包含电压、温度等响应量，因此它是通用数据驱动状态编码器，不是严格外生控制
编码器。异常响应可能影响状态表示，这一信息边界必须在论文中说明，并用状态打乱负对照
检验正确工况配对是否确有作用。

## 3 改进二：原型 Query 交叉注意力与输出拼接

### 3.1 可学习工况 Query

模型设置 $K$ 个可学习工况 Query：

$$
P=[p_1,\ldots,p_K]^{\mathsf T}\in\mathbb{R}^{K\times d},
$$

默认 $K=6,d=32$。Query 不预设充电、放电或高温等人工标签，而是在预测、重构和辅助
目标的共同梯度下学习。正交初始化只负责提供分离的优化起点，不作为独立创新或防坍缩
充分条件。

### 3.2 多头交叉注意力

原型作为 Query，通道 Token 作为 Key 和 Value：

$$
Q=PW_Q,\qquad K_H=H_{\mathrm{stat}}W_K,\qquad
V_H=H_{\mathrm{stat}}W_V.
$$

第 $m$ 个注意力头为

$$
A^{(m)}=\operatorname{Softmax}
\left(\frac{Q^{(m)}{K_H^{(m)}}^{\mathsf T}}{\sqrt{d_h}}\right),
$$

$$
O^{(m)}=A^{(m)}V_H^{(m)}.
$$

四个头拼接后执行残差连接和归一化：

$$
O=\operatorname{LN}
\left(P+\operatorname{Concat}(O^{(1)},\ldots,O^{(M)})W_O\right).
$$

得到 $O=[o_1,\ldots,o_K]^{\mathsf T}$。每个 $o_k$ 都是 Query 根据当前窗口 Token
动态查询得到的条件原型输出，并非固定字典向量。注意力矩阵保留 Query—通道关系，可在
训练后分析各原型主要关注的变量。

### 3.3 原型输出拼接与连续状态投影

当前实现不再计算全局 Token 摘要、余弦路由概率或温度软聚合。交叉注意力得到的全部
原型输出保留其 Query 槽位身份，直接展平并投影为窗口级状态：

$$
o_{\mathrm{flat}}=\operatorname{Flatten}(O)
\in\mathbb{R}^{Kd},
$$

$$
z_{\mathrm{regime}}=W_z o_{\mathrm{flat}}+b_z
\in\mathbb{R}^{8}.
$$

默认 $K=6,d=32$，因此投影输入为 $192$ 维。输出拼接保留每个 Query 的动态响应，避免
对原型输出做未经验证的全局匹配或单一加权平均；连续性来自可学习投影和 FiLM 调制，
而不是显式的路由温度。

### 3.4 当前实现边界

原型参数采用正交初始化，但当前正式路线没有熵、使用率或 winner-rate 防坍缩损失。
`regime_prototype_lambda` 仅保留为兼容参数，训练日志中的 `prototype_loss` 固定为零；
因此论文不能声称已经验证了原型均衡或防坍缩机制。可解释性分析应记录每个 Query 对各
通道 Token 的交叉注意力，而不是报告不存在的路由概率、路由熵或原型使用率。

## 4 改进三：融合层 FiLM 状态调制

主干三路表示拼接为

$$
H=[H_{\mathrm{conv}};H_{\mathrm{feat}};H_{\mathrm{temp}}]
\in\mathbb{R}^{T\times3C}.
$$

C3 只在三路融合之后、共享 GRU 之前调制。由状态生成缩放和平移量：

$$
[\gamma,\beta]=W_fz_{\mathrm{regime}}+b_f,
\qquad \gamma,\beta\in\mathbb{R}^{3C}.
$$

沿时间维广播后

$$
H_{\mathrm{mod}}=
H\odot(1+\alpha\gamma)+\alpha\beta,
\qquad \alpha=0.1.
$$

$\gamma$ 调整不同融合特征在当前工况下的重要性，$\beta$ 修正工况相关特征基线。FiLM
生成层使用零初始化，训练开始时 $\gamma=\beta=0$，因此 $H_{\mathrm{mod}}=H$，模型从
原 baseline 恒等路径开始学习，避免随机调制破坏主干。

状态向量还需重建原始统计描述：

$$
\widehat d=W_{\mathrm{rec}}z_{\mathrm{regime}}+b_{\mathrm{rec}},
$$

$$
L_{\mathrm{state}}=\operatorname{SmoothL1}(\widehat d,d).
$$

总目标为

$$
L=L_{\mathrm{forecast}}+L_{\mathrm{reconstruction}}
+\lambda_{\mathrm{state}}L_{\mathrm{state}},
$$

其中当前正式实验使用 $\lambda_{\mathrm{state}}=0.05$；baseline 将其置零。当前实现不
计算 $L_{\mathrm{proto}}$，因此不应在方法或结果中填写原型防坍缩损失。状态辅助损失只
参与训练，推理仍使用预测和重构残差。

## 5 参数量、误报机制与投稿水平

| 数据集 | MTAD-GAT 参数量 | 当前 C3 参数量 | 增加参数 | 增幅 |
|---|---:|---:|---:|---:|
| MSL | 450,085 | 480,875 | 30,790 | 6.84% |
| SMAP | 371,575 | 391,505 | 19,930 | 5.36% |
| 清华 EV | 365,590 | 379,004 | 13,414 | 3.67% |
| BMS | 395,345 | 418,895 | 23,550 | 5.96% |
| SWAT | 438,369 | 467,711 | 29,342 | 6.69% |
| WADI | 580,539 | 625,085 | 44,546 | 7.67% |

3.67%–7.67% 是当前实现的实际增量；新增比例随通道数增加而上升。交叉注意力处理 $C$
个统计 Token 而不是 $T\times C$ 个原始点，新增计算量主要来自 $O(KCd)$ 和 $O(Kd^2)$。
正式论文仍需报告训练时间、推理时延、显存和参数量，验证性能收益是否值得开销。

设正常残差为 $R$、工况为 $Z$：

$$
\operatorname{Var}(R)=
\mathbb{E}_Z[\operatorname{Var}(R\mid Z)]
+\operatorname{Var}_Z(\mathbb{E}[R\mid Z]).
$$

C3 若能适配不同正常工况，就可能降低第二项，使正常分数右尾收缩并减少固定阈值误报。
但响应量进入状态编码器也可能使异常被解释成工况，或者同时压低正常和故障分数。因此，
当前只能称其具备降低工况型误报的机制，不能在正式实验前宣称已经降低误报。

从方法新颖性看，该工作属于清楚、轻量的组合式增量创新。若只有少量单 seed 实验，现实
定位是中文核心、EI 或 SCI 四区应用型期刊；若完成四数据集、多 seed/车辆五折、严格
shuffled 对照、低误报指标、显著性和效率分析，并在 BMS 上稳定降低误报且保持召回，可
按 SCI 三区或部分应用二区准备。更高水平通常还需要跨品牌/跨站泛化、在线漂移、理论
分析或真实部署证据。

## 6 数据集与预处理协议

### 6.1 MSL 与 SMAP

MSL 当前输入 55 维，SMAP 输入 25 维，均以第 0 维为目标遥测通道，其余变量提供上下文。
prototype-query C3 使用全部输入通道形成统计 Token；restricted C3 在 MSL/SMAP 上沿用
排除目标通道后的池化上下文作为对照。评分遵循目标通道口径。官方正常训练段用于模型
学习，测试标签只用于最终评价。必须保留独立序列边界，窗口不能跨段拼接。

### 6.2 清华 EV

七通道输入为

$$
[V_{\mathrm{pack}},I,SOC,V_{\max},V_{\min},T_{\max},T_{\min}].
$$

C3 使用完整七通道，评分重点使用五个电压/温度响应通道。数据按车辆划分，Brand3 论文
协议五折作为主结果；每折归一化、训练和阈值校准只能使用本折正常训练/验证车辆。车辆
分数采用固定 Top-5% 高风险窗口均值。

### 6.3 BMS

BMS 包含 35 个簇级、系统级和层级派生特征，C3 使用全部通道。六个簇各约 21.4 万训练点，
因此正式任务 104 预先固定窗口长度 100、stride=10；所有模型臂与 seed 使用相同采样，避免
stride=1 产生大量高度相关窗口而耗尽 T4 预算。当前私有发布包的
`test_label` 是全零占位，不能把 BMS 伪装成有故障标签的监督检测数据，也不报告 BMS
AUROC、AP 或故障 Recall。正式任务 104 采用每个 cluster 的时间切分：前 80% 训练，后
20% 独立已知正常测试；归一化只由训练段拟合，训练段内部 10% 仅用于早停验证。

BMS 的研究问题是“工况条件化是否降低真实电池正常运行中的告警负担”。固定训练阈值后，
比较 baseline、restricted C3 和 prototype-query C3 的 FPR、每万窗口误报、每簇/每时间块
误报波动，并按 `BMSnI` 派生的 idle/frequency-regulation 状态分层。状态标签只用于事后
分层，不进入 C3 输入；相邻窗口不能在 train/test 边界间随机拆分。只有后续获得人工确认
故障标签或受控注入协议，才可追加故障 Recall/F1，并单列为新实验。

### 6.4 SWAT 与 WADI

SWAT 和 WADI 采用官方正常训练段训练，测试标签只在最终评分阶段使用。预处理后的输入
维度由 release-specific schema 确定（当前本地默认分别为 51 和 93），prototype-query
使用全部通道 Token；窗口长度为 100、步长为 10。最新外部验证计划比较 baseline、
restricted C3 和 prototype-query C3，使用 seed 3407、3408、3409 共 18 个正式任务（配置
99，Kaggle 固定 NVIDIA T4）。

| 数据集 | 通道数 | 窗口长度 | 步长 | 划分单位 |
|---|---:|---:|---:|---|
| MSL | 55 | 100 | 1 | 官方独立序列 |
| SMAP | 25 | 100 | 1 | 官方独立序列 |
| 清华 EV | 7 | 127 | 现有片段协议 | 车辆 |
| BMS | 35 | 100 | 10 | 电池簇与连续时间段 |
| SWAT | 51（按预处理 schema） | 100 | 10 | 官方时间段 |
| WADI | 93（按预处理 schema） | 100 | 10 | 官方时间段 |

归一化和缺失值处理参数只从训练数据估计。训练集更新参数，早停验证集选择 epoch，正常
校准集确定阈值，测试集只进行最终评价。数据不足时早停和校准可共用正常验证集合，但测试
仍需完全独立。

## 7 训练参数与评价指标

| 参数 | MSL/SMAP | 清华 EV | BMS | SWAT/WADI |
|---|---:|---:|---:|---:|
| 优化器/学习率 | Adam/0.001 | Adam/0.001 | Adam/0.001 | Adam/0.001 |
| 最大 epoch | 30 | 30 | 30 | 30 |
| batch size | 128 | 64 | 64 | 64 |
| lookback / 步长 | 100 / 1 | 127 / 片段窗口 | 100 / 10 | 100 / 10 |
| patience | 3 | 0（固定车辆五折预算） | 3 | 3 |
| Query 数/宽度/头数 | 6/32/4 | 6/32/4 | 6/32/4 | 6/32/4 |
| 状态维数 / FiLM 幅度 | 8 / 0.1 | 8 / 0.1 | 8 / 0.1 | 8 / 0.1 |

MSL、SMAP、BMS 和 SWAT/WADI 使用 seed 3407、3408、3409；清华 EV Brand2/Brand3 使用
五折并在正式扩展计划中使用三 seed。通用主指标为未 point-adjust 的 AP、AUROC；同时报告固定或
验证阈值下的 Precision/Recall/F1、原始事件级指标，以及 TPR@FPR 1%、0.5%、0.1%。
point-adjusted F1 只能作为历史兼容指标，不能替代原始点级排序指标。

BMS 部署型误报为

$$
\mathrm{FPR}(\theta)=\frac{FP}{FP+TN},
$$

$$
N_{\mathrm{FA}}^{10k}=10^4\times\mathrm{FPR}(\theta).
$$

还应报告每簇每日误报事件、最大簇 FPR、跨簇 FPR 标准差/CV 和按工况分层的误报率。连续
超阈值窗口合并为一个事件。当前无故障标签时，不能使用“Recall 保持”作为 BMS 判据；
若后续补充故障标签，再增加“FPR 下降且 Recall 基本保持”的双重判据。

### 7.1 正式实验运行性能记录（99 号起）

从 99 号计划开始，每个实验目录同时保存 `runtime` 运行记录，不把运行性能只写在
Notebook 日志中。数据侧记录预处理/数据准备秒数；训练侧记录：GPU 型号与显存、总参数量、可训练参数量、最大 epoch、
实际完成 epoch 数、总训练秒数、平均/中位 epoch 秒数、训练峰值显存；推理侧记录：评分
窗口数、总推理秒数、窗口吞吐（windows/s）、单窗口毫秒数、批次数、批延迟 P50/P95 和
推理峰值显存。正式 Kaggle 任务固定使用 NVIDIA T4、相同 batch、AMP 和数据加载设置，
结果汇总时报告均值 ± 标准差。

epoch 不能跨模型机械地等同：内部 C3 的最大 epoch 和 early-stopping 预算固定为上表，
实际完成 epoch 只由训练验证损失决定；清华 EV 的 `patience=0` 是为了保持车辆五折的固定
30 epoch 论文协议。外部模型按模型类别记录自己的训练单位：PCA/SPE 是一次闭式/增量拟合，
没有 epoch；USAD 在正式公共计划 05 中固定 10 epochs、三 seed；若运行 TranAD、Anomaly
Transformer、GDN 或 DCdetector，则锁定其公开实现的预先指定 epoch，并报告总训练时间、
实际更新步数和推理吞吐，不能用测试标签挑选轮数。

FLOPs/MACs 不在训练过程中动态估计：当前模型包含预测和重构两次前向及可选分支，仅用
静态工具得到的数字容易与实际评分路径不一致。论文主表以参数量、峰值显存、吞吐和
P50/P95 延迟为主；如投稿要求 FLOPs，再对冻结 checkpoint 使用同一输入形状和同一评分
路径做离线 profile，并明确是否包含重构前向。

## 8 消融与贡献度分析

| 编号 | 实验臂 | 目的 |
|---|---|---|
| B0 | MTAD-GAT | 原始基线 |
| A1 | restricted C3 | 受限统计编码 + 状态辅助重建 + 融合 FiLM |
| A2 | prototype-query C3 | 全通道 Token + Query 交叉注意力 + 原型输出拼接 |
| A3 | prototype-query shuffled (farthest) | 批内按统计描述最远无自配对的状态错配负对照（103/104） |
| A4 | prototype-query w/o state auxiliary | 去掉统计描述重建约束（103/104） |

A3 保持模型规模和主干损失不变，只按统计描述最远无自配对地错配窗口与状态，是当前最重要的语义负对照；
正式主结果 99 号起先比较 B0、A1、A2，正式消融 103/104 再比较 A3、A4。
若 A2 不能稳定优于 A3，不能把提升归因于正确工况语义；若 A2 低于 B0/A1，应如实报告
为数据集相关的负贡献。

消融任务选择 MSL、SMAP 和 Brand3：前两者代表公开遥测，Brand3 直接使用完整五折提供
车辆级证据。A3/A4 使用三个 seed（3407、3408、3409），不再以 Brand3 fold1 代替正式五折。外部模型对比优先覆盖
MSL、SMAP，公共工业数据可再加入 SWAT/WADI；随机深度模型使用三个 seed，确定性经典
方法至少运行一次并固定同一预处理和评分口径。

嵌套贡献可写为

$$
\Delta_{restricted}=M(A1)-M(B0),
$$

$$
\Delta_{prototype}=M(A2)-M(A1).
$$

误报下降率为

$$
R_{FPR}=\frac{\mathrm{FPR}_{B0}-\mathrm{FPR}_{A2}}
{\mathrm{FPR}_{B0}+\varepsilon}\times100\%.
$$

贡献比例只作描述：

$$
C_i=\frac{|\Delta_i|}{\sum_j|\Delta_j|+\varepsilon}.
$$

若 $\Delta_i<0$ 必须标记负贡献，不能用绝对值包装为正向结果。由于模块有依赖，贡献度
不能解释成严格因果 Shapley 值。

## 9 对比实验与结果分析模板

代表性对比包括 PCA/SPE、Isolation Forest、One-Class SVM、LSTM-AE/VAE、USAD、
OmniAnomaly、TranAD、Anomaly Transformer、DCdetector 和 MTAD-GAT。最重要的是与同
代码、同划分的 MTAD-GAT 比较。所有方法使用相同输入、划分、原始评分、seed 和校准规则，
测试标签不得用于阈值或超参数选择。

| 方法 | MSL AP/AUROC | SMAP AP/AUROC | 清华车辆 AP/AUROC | BMS 每万窗口误报 | BMS FPR/稳定性 |
|---|---|---|---|---:|---:|
| 经典方法 | 待填 | 待填 | 待填 | 待填 | 待填 |
| LSTM-AE/VAE | 待填 | 待填 | 待填 | 待填 | 待填 |
| USAD/OmniAnomaly | 待填 | 待填 | 待填 | 待填 | 待填 |
| Transformer 方法 | 待填 | 待填 | 待填 | 待填 | 待填 |
| MTAD-GAT | 待填 | 待填 | 待填 | 待填 | 待填 |
| restricted C3 | 待填 | 待填 | 待填 | 待填 | 待填 |
| prototype-query C3 | 待填 | 待填 | 待填 | 待填 | 待填 |

当前 external 计划中的电池模型（DyAD、GDN、AE、Deep-SVDD、LSTM-AD、Isolation Forest）已
覆盖领域专用、图模型、重构模型、单类模型和经典树模型，作为 EV 外部对比已足够。公共数据
集的正式计划 05 已补充 PCA/SPE 和 USAD，覆盖 MSL/SMAP、每个方法三 seed；旧的 MSL 计划
包含 TranAD、Anomaly Transformer、GDN 和 DCdetector，若纳入正式主表还
必须完成同一数据划分、原始 AP/AUROC、运行性能和多 seed 的统一输出，不能仅凭旧仓库日志
混表。无需继续堆叠大量相近 Transformer。

结果依次分析整体有效性、多 seed/折/簇胜负、A2 相对 A3 的正确状态必要性、有标签数据集的
故障召回保持、BMS 低误报、效率收益和失败案例。使用配对 bootstrap 置信区间；独立单元充足时采用
Wilcoxon signed-rank，并在多模型比较时使用 Holm 校正，同时报告实际效应量。

### 9.1 当前已完成的 C3 结果

以下是开发阶段结果快照，均为原始点级 AP/AUROC；MSL/SMAP 来自 98 号计划，清华 EV 来自
97 号计划。它们只用于说明冻结架构的已知正/负结果，不进入 99 号起的正式主表。

| 数据集 | baseline | restricted C3 | prototype-query C3 |
|---|---:|---:|---:|
| MSL AUROC / AP | 0.6989 / 0.2445 | 0.7058 / 0.2461 | **0.7150 / 0.2510** |
| SMAP AUROC / AP | **0.6014 / 0.1538** | 0.6045 / 0.1512 | 0.4851 / 0.1288 |
| 清华 EV AUROC | 0.6845 ± 0.1050 | **0.7530 ± 0.0980** | 0.7092 ± 0.1247 |
| 清华 EV AP | 0.5029 ± 0.1261 | **0.5615 ± 0.0999** | 0.5338 ± 0.0882 |

SMAP 上 prototype-query 的原始排序能力下降，不能用 point-adjusted 或测试集 oracle
阈值 F1 掩盖；该结果应作为当前架构的负结果保留。清华 EV 上 restricted C3 胜出，说明
原型 Query 输出拼接并非对所有数据集稳定有效，跨数据集结论必须依赖多 seed 和独立协议。

## 10 参数敏感性

| 参数 | 候选值 | 默认值 |
|---|---|---:|
| 原型数 $K$ | 2、4、6、8、10 | 6 |
| Query 维数 | 16、32、64 | 32 |
| 注意力头 | 1、2、4、8 | 4 |
| 状态维数 | 4、8、16、32 | 8 |
| FiLM 幅度 | 0.05、0.1、0.2、0.5 | 0.1 |
| $\lambda_{state}$ | 0、0.01、0.05、0.1 | 0.05 |

先在 MSL 和一个清华开发折做单因素筛选，再冻结参数验证 SMAP、其余车辆折、BMS 和
SWAT/WADI。当前实现没有显式路由温度或原型损失，因此不再把 $K\times\tau$、路由熵、
winner rate 或 $\lambda_{proto}$ 列为正式敏感性因素；应记录 Query—通道注意力、参数量、
训练时间和推理时间。每个参数点至少三个 seed，论文强调稳定区间而非单点最优。

## 11 BMS 专项表与结论边界

| 方法 | 正常窗口数 | 校准阈值 | FPR | 每万窗口误报 | 每日误报事件 | 最大簇 FPR | 工况 FPR 波动 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MTAD-GAT | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 简单统计 FiLM | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| restricted C3 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| prototype-query C3 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

还应按 idle/frequency-regulation 状态分层统计正常误报，并绘制正常分数 P95/P99/P99.9、
分簇箱线图、分时间块曲线和告警时间轴。报告由 104 号任务额外输出
`bms_false_alarm_by_regime.csv`；工况标签只用于事后评价，不作为 C3 输入。

最低结论判据为：各有标签数据集至少完成三 seed；prototype-query 多数重复优于 restricted
或 baseline；BMS 在固定正常阈值下 FPR、每万窗口误报和工况/簇稳定性改善。当前 BMS 没有
故障标签，因此只能写“降低已知正常运行误报负担”，不能写“故障 Recall 提升”。若未来
补充故障标签，再加上 Recall 保持条件；Query—通道注意力没有明显退化；当前 3.67%–7.67%
参数增量对应可测收益。只满足整体指标时，结论应限定为“改善部分数据集检测排序”。

## 12 推荐执行顺序

97、98 号计划只保留为 99 号前的开发归档，不进入正式主表。99 号起进入正式实验：99 为
SWAT/WADI 三组×三 seed，100 为 MSL/SMAP 三组×三 seed，101/102 为 Brand3/Brand2 五折
×三 seed，103 为 MSL/SMAP + Brand3 完整五折核心消融，104 为私有 BMS 五组×三 seed
运行。正式公共外部对比使用计划 05 的 PCA/SPE 与 USAD 三 seed。当前不再增加路由或防
坍缩模块，最重要的工作是验证冻结 C3 的跨数据集稳定性，以及在没有故障标签的 BMS 上
用可审计的正常工况误报证据说明其工程价值。
