# 正式实验结果一：99–102

日期：2026-08-21
代码提交：`445fbf679b88c2506432b85ba361b39f89d5b3a8`
硬件：Kaggle NVIDIA Tesla T4
主指标：原始逐点 Average Precision（AP）和 AUROC

## 1. 实验状态与统计口径

| 计划 | 数据/账号 | Kaggle 状态 | 完成数 | 统计口径 |
|---|---|---|---:|---|
| 99 | SWAT/WADI，`daisychen2` | COMPLETE | 18/18 | 每个数据集、模型跨 3 个 seed，均值±样本标准差 |
| 100 | MSL/SMAP，`daisychen2` | COMPLETE | 18/18 | MSL 为原始结果；SMAP 为去掉重复 `P-2` 后的离线修正评估 |
| 101 | Brand3，`chenmjdaisy` | COMPLETE | 45/45 | 每个模型 5 折×3 seed，车辆级 AP/AUROC，均值±样本标准差 |
| 102 | Brand2，`chenmjdaisy` | CANCEL_ACKNOWLEDGED | 15/45 | 仅 baseline 的 5 折×3 seed 完成，restricted/prototype-query 未完成 |

Brand3 的车辆级 AP 使用 `vehicle_pr_auc`，其定义为 `average_precision_score`；梯形积分不是主指标。不同数据集的 AP 受异常比例和评价层级影响，不能直接按数值大小横向排序。

## 2. 正式结果

### 2.1 计划 100：MSL/SMAP

MSL 数据与本地 98 的原始数据一致，三 seed 结果为正式结果。SMAP 的当前代码把同一个 `P-2` 序列按标签表中的两条记录重复拼接；下面的 SMAP 数值是从已保存的 Kaggle 分数中删除两个重复测试段后的离线修正值。训练阶段仍经受了重复 `P-2` 的影响，因此 SMAP 应标为“修正评估”，不能等同于完全干净重训。

| 数据集 | 模块 | AP seed 3407 | AP seed 3408 | AP seed 3409 | AP 均值 | AP 样本标准差 | AUROC seed 3407 | AUROC seed 3408 | AUROC seed 3409 | AUROC 均值 | AUROC 样本标准差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MSL | baseline | 0.241877 | 0.238683 | 0.234781 | 0.2384 | 0.0036 | 0.690028 | 0.686655 | 0.662887 | 0.6799 | 0.0148 |
| MSL | restricted | 0.235344 | 0.236533 | 0.256787 | 0.2429 | 0.0121 | 0.668611 | 0.673761 | 0.726811 | 0.6897 | 0.0322 |
| MSL | prototype-query | 0.246917 | 0.226862 | 0.249174 | 0.2410 | 0.0123 | 0.716749 | 0.649739 | 0.701822 | 0.6894 | 0.0352 |
| SMAP（去重后修正评估） | baseline | 0.123898 | 0.139869 | 0.141474 | 0.1351 | 0.0097 | 0.460341 | 0.539916 | 0.556354 | 0.5189 | 0.0514 |
| SMAP（去重后修正评估） | restricted | 0.144607 | 0.122110 | 0.139638 | 0.1355 | 0.0118 | 0.569694 | 0.451854 | 0.537366 | 0.5196 | 0.0609 |
| SMAP（去重后修正评估） | prototype-query | 0.142915 | 0.156015 | 0.148138 | 0.1490 | 0.0066 | 0.550530 | 0.604169 | 0.539124 | 0.5646 | 0.0347 |

每个 seed 均单独列出；均值和样本标准差分别计算，不把 AP 与 AUROC 拼在同一单元格。

### 2.2 计划 99：SWAT/WADI

| 数据集 | 模块 | AP seed 3407 | AP seed 3408 | AP seed 3409 | AP 均值 | AP 样本标准差 | AUROC seed 3407 | AUROC seed 3408 | AUROC seed 3409 | AUROC 均值 | AUROC 样本标准差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SWAT | baseline | 0.711782 | 0.712271 | 0.703983 | 0.7093 | 0.0047 | 0.821064 | 0.817905 | 0.817097 | 0.8187 | 0.0021 |
| SWAT | restricted | 0.703380 | 0.713207 | 0.697923 | 0.7048 | 0.0077 | 0.814591 | 0.820486 | 0.817134 | 0.8174 | 0.0030 |
| SWAT | prototype-query | 0.708332 | 0.713007 | 0.707579 | 0.7096 | 0.0029 | 0.822471 | 0.828590 | 0.808484 | 0.8198 | 0.0103 |
| WADI | baseline | 0.054473 | 0.045649 | 0.045596 | 0.0486 | 0.0051 | 0.484923 | 0.425274 | 0.420585 | 0.4436 | 0.0359 |
| WADI | restricted | 0.071348 | 0.044953 | 0.047038 | 0.0544 | 0.0147 | 0.485988 | 0.410615 | 0.445057 | 0.4472 | 0.0377 |
| WADI | prototype-query | 0.047454 | 0.045359 | 0.046822 | 0.0465 | 0.0011 | 0.447598 | 0.416840 | 0.441681 | 0.4354 | 0.0163 |

### 2.3 计划 101：Brand3 五折三 seed

这是车辆级结果，统计 15 个折×seed 运行，而不是逐时间点结果。原始输出没有在本地保留按 seed/fold 展开的明细；因此下面保留独立的 seed 列，但不伪造缺失值。

| 数据集 | 模块 | AP seed 3407（原始明细缺失） | AP seed 3408（原始明细缺失） | AP seed 3409（原始明细缺失） | AP 总均值（15 折×seed） | AP 样本标准差 | AUROC seed 3407（原始明细缺失） | AUROC seed 3408（原始明细缺失） | AUROC seed 3409（原始明细缺失） | AUROC 总均值（15 折×seed） | AUROC 样本标准差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Brand3 | baseline | 缺失 | 缺失 | 缺失 | 0.5196 | 0.1108 | 缺失 | 缺失 | 缺失 | 0.7209 | 0.0795 |
| Brand3 | restricted | 缺失 | 缺失 | 缺失 | **0.5581** | 0.1078 | 缺失 | 缺失 | 缺失 | **0.7449** | 0.0916 |
| Brand3 | prototype-query | 缺失 | 缺失 | 缺失 | 0.5221 | 0.0834 | 缺失 | 缺失 | 缺失 | 0.7248 | 0.0595 |

Brand3 上 restricted 是三组中最好的一组；prototype-query 相对 baseline 没有稳定增益，且折间波动较大。

### 2.4 计划 102：Brand2 当前只能报告 partial baseline

Brand2 在完成 15 个 baseline 运行后，于 `restricted_f0_seed3407` 开始阶段被取消。先给出按 seed 汇总的结果；每个 seed 的均值是其 5 个 fold 的平均值。

| 数据集 | 模块 | AP seed 3407（5 折均值） | AP seed 3408（5 折均值） | AP seed 3409（5 折均值） | AP 总均值 | AP 样本标准差 | AUROC seed 3407（5 折均值） | AUROC seed 3408（5 折均值） | AUROC seed 3409（5 折均值） | AUROC 总均值 | AUROC 样本标准差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Brand2 | baseline | 0.858389 | 0.863496 | 0.861369 | 0.8611 | 0.0697 | 0.775641 | 0.803663 | 0.805861 | 0.7951 | 0.0895 |
| Brand2 | restricted | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 |
| Brand2 | prototype-query | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 |

AP 的每个 fold×seed 明细：

| 数据集 | 模块 | fold 0 / seed 3407 | fold 0 / seed 3408 | fold 0 / seed 3409 | fold 1 / seed 3407 | fold 1 / seed 3408 | fold 1 / seed 3409 | fold 2 / seed 3407 | fold 2 / seed 3408 | fold 2 / seed 3409 | fold 3 / seed 3407 | fold 3 / seed 3408 | fold 3 / seed 3409 | fold 4 / seed 3407 | fold 4 / seed 3408 | fold 4 / seed 3409 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Brand2 | baseline | 0.965351 | 0.973250 | 0.988588 | 0.847075 | 0.831048 | 0.816783 | 0.886144 | 0.874914 | 0.868833 | 0.844573 | 0.860323 | 0.832418 | 0.748804 | 0.777947 | 0.800223 |

AUROC 的每个 fold×seed 明细：

| 数据集 | 模块 | fold 0 / seed 3407 | fold 0 / seed 3408 | fold 0 / seed 3409 | fold 1 / seed 3407 | fold 1 / seed 3408 | fold 1 / seed 3409 | fold 2 / seed 3407 | fold 2 / seed 3408 | fold 2 / seed 3409 | fold 3 / seed 3407 | fold 3 / seed 3408 | fold 3 / seed 3409 | fold 4 / seed 3407 | fold 4 / seed 3408 | fold 4 / seed 3409 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Brand2 | baseline | 0.923077 | 0.935897 | 0.974359 | 0.703297 | 0.802198 | 0.747253 | 0.846154 | 0.782051 | 0.794872 | 0.703297 | 0.736264 | 0.703297 | 0.702381 | 0.761905 | 0.809524 |

Brand2 没有 restricted 和 prototype-query 结果，因此不能写成三组正式比较，也不能据此判断 C3 是否有效。

## 3. 为什么本地 SMAP C3 很差，而三 seed 结果看起来变好？

### 3.1 先区分“哪些结果变好”

三 seed 并不是所有模型都比本地 98 更好。下面把本地 98 的单 seed 与 100 号 SMAP 的修正后每个 seed 对齐；AP 和 AUROC 分列：

| 数据集 | 模块 | AP 本地98（seed 3407） | AP Kaggle100（seed 3407） | AP seed 3408 | AP seed 3409 | AP 修正后均值 | AUROC 本地98（seed 3407） | AUROC Kaggle100（seed 3407） | AUROC seed 3408 | AUROC seed 3409 | AUROC 修正后均值 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SMAP | baseline | 0.1538 | 0.1239 | 0.1399 | 0.1415 | 0.1351 | 0.6014 | 0.4603 | 0.5399 | 0.5564 | 0.5189 |
| SMAP | restricted | 0.1512 | 0.1446 | 0.1221 | 0.1396 | 0.1355 | 0.6045 | 0.5697 | 0.4519 | 0.5374 | 0.5196 |
| SMAP | prototype-query | 0.1288 | 0.1429 | 0.1560 | 0.1481 | 0.1490 | 0.4851 | 0.5505 | 0.6042 | 0.5391 | 0.5646 |

真正出现“本地很差、三 seed 均值变好”的主要是 prototype-query；baseline 和 restricted 的修正后三 seed 均值反而低于本地 98。

### 3.2 主要原因

1. **本地 98 只有一个 seed。** SMAP 的 seed 波动很大。prototype-query 的修正 AP 从 `0.1429` 到 `0.1560`，AUROC 从 `0.5391` 到 `0.6042`；单个 seed 不能代表正式均值。早停也会随 seed 改变实际训练轮数。

2. **本地与 Kaggle 曾经使用了不同的 SMAP 数据口径。** 本地旧缓存排除了 `P-2`，共有 53 条序列；Kaggle 100 按当前代码把 `P-2` 重复加入两次，形成 55 条序列。测试窗口数从本地的 `422317` 变成 Kaggle 的 `438535`。离线删除两个测试段后，测试评价口径已经对齐，但训练时重复的 `P-2` 仍然存在。

3. **3060 与 T4 的确定性并不完全等价。** 本地使用 RTX 3060、PyTorch `2.5.1+cu118`；Kaggle 使用 T4 和不同 CUDA 运行时。即使设置了 deterministic，GPU 算子、数据加载和早停路径仍可能带来小幅差异。SMAP 的困难程度使这种差异被放大。

因此，三 seed 结果不是“模型突然变强”，而是单 seed、数据口径和硬件/训练路径共同造成的混合效应。当前最稳妥的表述是：SMAP 的 prototype-query 在已完成的三 seed 修正评估中有一定平均增益，但由于训练阶段仍含重复 `P-2`，该结论只能作为暂定结果。

## 4. 为什么 SWAT/WADI 看起来没有 SMAP/MSL 好？

### 4.1 先纠正绝对性能比较

如果看绝对逐点 AP，SWAT 并不比 MSL/SMAP 差：SWAT 的 AP 约为 `0.705–0.710`，明显高于 MSL 的 `0.238–0.243` 和 SMAP 修正后的 `0.135–0.149`。真正表现较弱的是 WADI，其 AP 约为 `0.047–0.054`，AUROC 约为 `0.435–0.447`。

### 4.2 WADI 的 AP 已接近或低于随机基线

Kaggle 预处理日志记录 WADI 测试段为 93 个特征、172804 个点，异常比例为 `0.057736`。随机排序的 AP 大约就是异常比例，即约 `0.058`。WADI 三组结果都在这一水平附近或更低，且 AUROC 均低于 `0.5`，说明当前分数排序没有稳定区分异常和正常点，而不是单纯因为 AP 的数值尺度比较小。

SWAT 的测试异常比例为 `0.121402`，但 AP 约 `0.71`、AUROC 约 `0.82`，说明 SWAT 的绝对检测结果其实很好。

### 4.3 C3 在 SWAT/WADI 上没有带来明显增益的可能原因

正式结果只支持以下现象，不足以单独证明某一个机制原因：

- SWAT 中 baseline 已经很强，restricted 的 AP/AUROC 略降，prototype-query 与 baseline 基本持平，说明当前 C3 没有在强 baseline 上形成可见增益。
- WADI 的 restricted 只有小幅 AP 增加，prototype-query 反而下降；在 AUROC 低于 `0.5` 的背景下，这种小差值不能解释为可靠改进。
- SWAT/WADI 是工业控制系统数据，异常往往表现为传感器—执行器关系、控制回路或操作阶段的突变；当前 C3 主要从窗口统计量学习连续工况，并通过较小 FiLM 系数调制预测/重构主干。它未必能恢复控制因果关系，也可能把攻击状态变化混入工况表示。
- WADI 的低异常比例、长时间序列、复杂运行阶段和测试分布变化，会使通用预测/重构模型的残差排序更不稳定。此时增加工况旁路不一定能解决根本的控制回路建模问题。

### 4.4 C3 增益的直接对比

| 数据集 | 模块 | AP 相对 baseline | AUROC 相对 baseline |
|---|---|---:|---:|
| MSL | restricted | +0.0045 | +0.0098 |
| MSL | prototype-query | +0.0026 | +0.0095 |
| SMAP（修正评估） | restricted | +0.0004 | +0.0008 |
| SMAP（修正评估） | prototype-query | +0.0139 | +0.0457 |
| SWAT | restricted | -0.0045 | -0.0013 |
| SWAT | prototype-query | +0.0003 | +0.0012 |
| WADI | restricted | +0.0059 | +0.0036 |
| WADI | prototype-query | -0.0020 | -0.0082 |

这张表说明当前结果更像“数据集依赖的条件化收益”，而不是在所有数据集上都有效的统一增益。SMAP 的 prototype-query 增益还需要在去重后的干净训练协议下复核；SWAT/WADI 不能据此声称 C3 已经优于 baseline。

## 5. 当前可用于论文的结论边界

1. **可以正式报告：** 99 的 SWAT/WADI 三 seed 原始结果、101 的 Brand3 五折三 seed 结果、100 的 MSL 三 seed 结果。
2. **需要标注限制：** 100 的 SMAP 是重复 `P-2` 后的训练结果加去重后的离线测试评估，不能称为完全干净的正式主结果。
3. **不能正式下结论：** 102 的 Brand2 三组 C3 比较；目前只有 baseline partial 结果。
4. **方法结论应收敛：** 当前 C3 在 Brand3 restricted 和 SMAP prototype-query 上显示出潜在收益，但在 SWAT/WADI 上没有稳定增益；WADI 当前甚至没有超过随机排序基线。
5. **后续复现实验必须增加提交前检查：** 标签实体唯一性、原始文件与标签行一一对应、序列数、点数、窗口数和数据哈希都应写入 manifest；发现重复实体时直接中止训练。

## 6. 103 去重、实验协议与补充运行状态

### 6.1 103 的前置结果去重

历史 103 配置展开为 105 个条目，全部是 `dry_run`，没有产生新的训练结果。其中 63 个条目是 100/101 已完成的三组主结果，已从当前 103 配置删除；当前只保留 42 个真正新增的消融条目。

| 数据集 | 模块 | 103 原始条目 | 去重动作 | 已跑任务 | 当前 103 状态 |
|---|---|---:|---|---|---|
| MSL | baseline | 3 seeds | 删除 | 100 | 不重复跑 |
| MSL | restricted | 3 seeds | 删除 | 100 | 不重复跑 |
| MSL | prototype-query | 3 seeds | 删除 | 100 | 不重复跑 |
| SMAP | baseline | 3 seeds | 删除 | 100 | 不重复跑 |
| SMAP | restricted | 3 seeds | 删除 | 100 | 不重复跑 |
| SMAP | prototype-query | 3 seeds | 删除 | 100 | 不重复跑 |
| Brand3 | baseline | 5 folds×3 seeds | 删除 | 101 | 不重复跑 |
| Brand3 | restricted | 5 folds×3 seeds | 删除 | 101 | 不重复跑 |
| Brand3 | prototype-query | 5 folds×3 seeds | 删除 | 101 | 不重复跑 |
| MSL/SMAP | 早期 baseline/restricted/prototype-query 版本 | 历史条目 | 归档参考 | 95、98 | 不作为 103 新结果 |
| Brand3 | 早期 baseline/restricted/prototype-query 版本 | 历史条目 | 归档参考 | 96、97 | 不作为 103 新结果 |
| MSL/SMAP | prototype-query shuffled_state_farthest | 2 datasets×3 seeds | 保留 | — | dry-run，新消融 |
| Brand3 | prototype-query shuffled_state_farthest | 5×3 | 保留 | — | dry-run，新消融 |
| MSL/SMAP | prototype-query w/o state auxiliary | 2 datasets×3 seeds | 保留 | — | dry-run，新消融 |
| Brand3 | prototype-query w/o state auxiliary | 5×3 | 保留 | — | dry-run，新消融 |

95/96 的 `prototype_shuffled` 是早期开发版本，不能直接替代 103 的 `shuffled_state_farthest`：两者的 shuffle 定义、损失配置和训练协议不同。因此 103 只删除与 100/101 完全对应的 baseline、restricted、prototype-query，保留两个新消融方向。

### 6.2 当前 epoch/早停协议是否统一

当前并不统一：最大 epoch 都是 30，但 `patience=3` 会早停，`patience=0` 则按固定预算运行。因而不同任务的“实际 epoch 数”不能直接横向比较。

| 数据集 | 模块 | 最大 epoch | patience | 当前状态 | 说明 |
|---|---|---:|---:|---|---|
| 99 SWAT/WADI | baseline/restricted/prototype-query | 30 | 3 | 已运行 | 验证损失早停 |
| 100 MSL/SMAP | baseline/restricted/prototype-query | 30 | 3 | 已运行 | 验证损失早停；SMAP 另有数据去重限制 |
| 101 Brand3 | baseline/restricted/prototype-query | 30 | 0 | 已运行 | 固定车辆五折预算 |
| 102 Brand2 | baseline/restricted/prototype-query | 30 | 0 | baseline 已运行 | restricted/prototype-query 未完成 |
| 103 MSL/SMAP | 新消融 | 30 | 3 | dry-run | 未产生结果 |
| 103 Brand3 | 新消融 | 30 | 0 | dry-run | 未产生结果 |
| 104 BMS | 内部 C3 组 | 30 | 3 | dry-run | 未产生训练结果 |

因此，99–102 可以作为“按各自任务记录的正式/部分正式结果”报告，但不能宣称已经采用完全统一的训练预算。后续论文主实验建议预先冻结：`max_epochs=30`、`patience=3`、`min_delta=1e-4`，并恢复验证集最佳 checkpoint；所有模型和 seed 使用同一验证划分。若必须复现 Brand3 论文的固定 30 epoch 协议，则应把所有对比模型统一设为 `patience=0`，不能与早停结果混列。

### 6.3 正式 C3 外部对比计划（合并 04+05）

外部 04 和 05 统一登记为 `06_c3_formal_external_comparison_msl_smap`，不再把 04 的旧单次结果和 05 的 dry-run 当成两套正式结论。计划统一使用项目的 MSL/SMAP 数据读取、仅训练段归一化、原始逐点 AP/AUROC、禁止 point adjustment，并固定 seed `3407/3408/3409`。PCA/SPE 没有迭代 epoch；USAD 固定最大 10 epochs；TranAD、Anomaly Transformer、GDN、DCdetector 保留各自公开实现的训练预算，但必须记录实际 epoch 和运行时间。

正式入主表的门槛是：同一方法在同一数据集完成三个 seed，输出 AP、AUROC、训练/推理时间和显存信息。04 当前只有 MSL 单次结果，05 的 12 个任务全部是 `dry_run`，所以现在都不能作为正式外部对比结论。

| 数据集 | 模块 | 计划来源 | 正式 seed | 训练协议 | 当前状态 | 是否可入正式主表 |
|---|---|---|---|---|---|---|
| MSL | TranAD | 04 | 3407/3408/3409 | 原实现，当前记录 5 epoch | 旧结果仅单次 | 否，需三 seed 重跑 |
| MSL | Anomaly Transformer | 04 | 3407/3408/3409 | 原实现，10 epoch | 旧结果仅单次 | 否，需三 seed 重跑 |
| MSL | GDN | 04 | 3407/3408/3409 | 原实现，10 epoch | 旧结果仅单次 | 否，需三 seed 重跑 |
| MSL | DCdetector | 04 | 3407/3408/3409 | 原实现，10 epoch | 旧运行因缺少 `tsfresh` 失败 | 否，需补依赖后重跑 |
| MSL | PCA/SPE | 05 | 3407/3408/3409 | 无迭代 epoch | 3 个任务 dry-run | 否，需正式运行 |
| SMAP | PCA/SPE | 05 | 3407/3408/3409 | 无迭代 epoch | 3 个任务 dry-run | 否，需正式运行 |
| MSL | USAD | 05 | 3407/3408/3409 | 最大 10 epochs | 3 个任务 dry-run | 否，需正式运行 |
| SMAP | USAD | 05 | 3407/3408/3409 | 最大 10 epochs | 3 个任务 dry-run | 否，需正式运行 |
| SMAP | TranAD | 04 扩展 | 3407/3408/3409 | 原实现，需先完成数据适配 | 尚未配置/运行 | 否 |
| SMAP | Anomaly Transformer | 04 扩展 | 3407/3408/3409 | 原实现，需先完成数据适配 | 尚未配置/运行 | 否 |
| SMAP | GDN | 04 扩展 | 3407/3408/3409 | 原实现，需先完成数据适配 | 尚未配置/运行 | 否 |
| SMAP | DCdetector | 04 扩展 | 3407/3408/3409 | 原实现，需先完成数据适配 | 尚未配置/运行 | 否 |

已获得的 04 单次结果只作为开发记录，指标分列如下：

| 数据集 | 模块 | AP | AUROC | 训练 epoch | 结果性质 |
|---|---|---:|---:|---:|---|
| MSL | TranAD | 0.148542 | 0.528272 | 5 | 单次开发结果 |
| MSL | GDN | 0.137687 | 0.568382 | 10 | 单次开发结果 |
| MSL | Anomaly Transformer | 0.097489 | 0.444225 | 10 | 单次开发结果 |
| MSL | DCdetector | — | — | — | 缺少 `tsfresh`，运行失败 |

### 6.4 BMS 原始数据的运行日期

104 目前没有训练进程和训练结果。下面只是查看原始 `SYS_I`/运行信号后的运行日判断，不是检测指标：

| 数据集 | 模块 | 日期 | 原始信号判断 | 结论 |
|---|---|---|---|---|
| BMS | SYS_I | 7/8–7/12、7/15、7/18–7/20、7/23–7/24、7/27–7/28 | 持续有运行信号 | 明确运行 |
| BMS | SYS_I | 7/17、7/22 | 仅部分时段有运行信号 | 部分运行 |
| BMS | SYS_I | 7/7、7/14、7/16、7/21、7/29–7/31 | 零星有运行信号 | 仅零星运行 |
| BMS | SYS_I | 7/1–7/6、7/13、7/25–7/26 | 基本没有运行信号 | 基本未运行 |

PCS 功率在这些原始记录中全天为 0，因此目前只能确认 BMS 侧的运行信号，不能据此宣称 PCS 处于带载运行状态。
