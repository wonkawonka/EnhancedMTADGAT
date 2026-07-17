# 方法、动机与文献依据

本文档只回答三件事：问题从哪里来、当前方法实际做了什么、哪些结论必须由本项目实验证明。

## 1. 方法定位

项目不是某篇论文的逐行复现，而是以 MTAD-GAT 为主干的两级方法：

- **C3：动态上下文条件化异常检测。** 从控制/慢状态通道学习连续上下文，在关系融合表示上进行 FiLM 调制。
- **C4：电—热响应一致性增强。** 在 C3 上增加物理状态描述、响应一致性损失和响应残差评分。

建议章节名：

- 第三章：基于动态上下文条件化的多变量时序异常检测方法。
- 第四章：电—热响应一致性增强的动力电池异常检测方法。

## 2. 动机与核心证据

清华团队的电池研究将电流、SOC 视为输入，将电压、温度视为动态响应，并指出罕见正常电流模式可能导致普通异常检测方法误报。这直接支持以下问题：**相同响应值在不同运行输入下含义不同，固定正常模式容易混淆罕见正常状态和真实异常。**

来源：Zhang et al., *Realistic fault detection of li-ion battery via dynamical deep learning*, Nature Communications, 2023，[论文](https://www.nature.com/articles/s41467-023-41226-5)。

该文不能证明本项目的 FiLM 结构有效，只能证明问题和“输入—响应”划分有依据。本项目的具体编码器、条件化位置和辅助任务均需消融验证。

## 3. 第三章模型

设窗口 (X\in\mathbb{R}^{L\times D})，条件通道为 (C=X[:,\mathcal C])，报警响应维度为 \(\mathcal R\)。

### 3.1 关系主干

```text
X -> Conv1D -> H_f (变量关系 GATv2)
            -> H_t (时间关系 GATv2)
H_rel = concat(X, H_f, H_t)
```

变量图的节点是变量，时间图的节点是窗口时刻，都是数据驱动的全连接候选关系，不是电池串并联拓扑。卷积、两类图注意力、GRU 以及预测/重构双任务来自 MTAD-GAT 主干。

来源：Zhao et al., *Multivariate Time-series Anomaly Detection via Graph Attention Network*, ICDM 2020，[预印本](https://arxiv.org/abs/2009.02040)。GATv2 来源：Brody et al., ICLR 2022，[论文](https://openreview.net/forum?id=F72ximsx7C1)。

### 3.2 连续上下文编码

```text
C -> temporal Conv -> dilated temporal Conv
  -> attention pooling + last state -> z ∈ R^32
```

电池数据的辅助任务预测电流活跃度、波动、方向切换率和 SOC 变化，防止 (z) 退化。MSL 缺少相应物理语义，因此关闭该辅助任务，只把非目标通道编码为潜在上下文。

编码器和四个描述目标是项目自定义设计。TS2Vec 等自监督表征文献只能支持“辅助目标可约束时序表征”的一般原则，不能作为该精确结构的来源：Yue et al., AAAI 2022，[论文](https://ojs.aaai.org/index.php/AAAI/article/view/20881)。

### 3.3 方案 3：关系融合表示上的 FiLM

状态向量不再只调 Transformer 残差比例，而是先调制 GRU 与 Transformer 共用的融合表示：

```text
[gamma, beta] = MLP(z)
H_cond = (1 + tanh(gamma)) ⊙ H_rel + tanh(beta)
h = GRU(H_cond) + Transformer(H_cond)
```

这样条件信息直接改变“当前运行状态下如何解释变量/时间关系”，同时保留两条序列建模路径。旧方案

```text
h = h_gru + sigmoid(Wz+b) ⊙ h_tr
```

只作为消融项，用来判断收益来自条件信息本身还是条件化位置。

FiLM 的基本思想来自 Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018，[论文](https://ojs.aaai.org/index.php/AAAI/article/view/11671)；将其用于当前 MTAD-GAT 融合表示是项目组合，不可写成直接复现。

### 3.4 预测、重构与响应评分

训练目标为：

```text
L3 = RMSE_forecast + RMSE_reconstruction + lambda_desc L_desc
```

各输出通道先融合预测误差和重构误差，再只对 \(\mathcal R\) 中的响应维度求均值得到全局异常分数。条件维度可以继续被模型预测，但不会仅因工况少见直接触发总报警。

质量感知融合依据正常校准集上两分支误差的稳健波动确定权重，测试时冻结。它是项目自定义评分策略，必须与固定权重比较。

## 4. 数据集变量语义

| 数据集 | 上下文编码输入 | 全局异常评分 |
| --- | --- | --- |
| MSL | 通道 1–54 | 通道 0 |
| 清华 EV | current、SOC | voltage、Vmax、Vmin、Tmax、Tmin |
| NASA Random Walk | current | voltage、temperature；原始 step code 不进入模型 |
| BMS | SYS_I、BMSnRSOC | BMSnI、簇电压/温度、单体离散度和层级差异 |

BMS 的设置针对“某个并联簇承担异常电流份额”。若异常目标是串联单体电芯过流，当前没有单体电流观测，只能检测伴随的电压/温度异常。

## 5. 第四章模型

C4 在 C3 上增加电流、电荷累计、SOC、电压相对位置、温度相对位置等状态描述，并计算：

- 电压变化率与电压极差；
- 温度变化率与温度极差；
- 累计电荷响应；
- SOC—电流变化耦合。

这些响应一方面形成训练正则，另一方面形成测试期物理残差，并由正常校准集限制其最大融合权重。它们是物理启发约束，不是完整等效电路。

一般依据：Karniadakis et al., *Physics-informed machine learning*, Nature Reviews Physics 2021，[论文](https://www.nature.com/articles/s42254-021-00314-5)。更强电池模型约束方法可参照 Cao et al., Nature Communications 2025，[论文](https://www.nature.com/articles/s41467-025-56832-8)，但不能把其模型能力归于本项目。

## 6. 最小实验闭环

第三章必须完成：MTAD-GAT 与 C3 主对比；无条件化、统计量编码、无辅助任务、旧残差门控和固定分数融合消融；MSL 与清华三种子结果；NASA 表征探针；BMS 工况误报与案例。

第四章必须完成：C3 到 C4 的加法递进；电压、温度、电荷、SOC—电流响应留一；三种子检测指标、响应 MAE 和效率统计。

## 7. 写作红线

- 文献证明问题存在，不等于证明当前模型有效。
- MSL 的匿名变量不能写成电池工况；NASA 工步标签只用于事后探针。
- 清华公开包没有车辆 ID，不能宣称车辆独立泛化。
- BMS 无可靠故障标签，不能报告或暗示监督检测精度。
- 未建立 OCV、内阻和 RC 参数模型，不能称为 ECM 或严格物理约束。
- 不以 point-adjusted F1 作为主结果。

## 8. 代码位置

- 数据集条件通道：`src/models/model_factory.py`
- 响应评分维度：`src/data/utils.py`
- 状态编码与 FiLM：`src/models/modules.py`、`src/models/mtad_gat.py`
- 训练损失：`src/engine/training.py`
- 异常评分：`src/engine/prediction.py`
- 物理响应：`src/models/physical_response.py`
