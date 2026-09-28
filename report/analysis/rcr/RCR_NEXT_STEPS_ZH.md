# RCR 下一步结构研究与 Brand3 验证（2026-09-24）

状态：机制判断与任务协议已冻结；Brand3 两个 Kaggle 分片运行结果待补。此文不是论文最终四表。

## 当前判断

- 把论文主线转向“正常重构残差的联合依赖”是合理的，但不能把“残差上用 GAT”或“mask 一维预测另一维”单独宣称为首创。用户附文所列 SPAGD、GCAD 等相邻工作仍需核对原文与具体任务/监督假设，不能引用占位符作为文献证据。
- CH 上完整 RCR 相对共享 Raw 的 AUROC 在 LFP/NCM 分别提高 0.1618/0.0300，AP 提高 0.0848/0.0160；但 RCR-Net 相对 RCR-Lite 的 F1 分别为 +0.0377/-0.0043，故“关系神经头稳定优于线性关系”**尚未成立**。
- 先前 MTAD-GAT 冻结残差探针中，完整精度矩阵相对仅保留对角项的 AUROC 有明显增益（LFP 探索性结果 +0.1580；见 `report/analysis/20260901new/MODEL_BOTTLENECK_RESEARCH_REPORT_ZH.md`）。但换成当前 RCR backbone 和评分协议后，RCR-Lite 相对条件幅值 Scale 的 AUROC 在 LFP 为 -0.0149、NCM 为 +0.0023；不能把旧探针结果直接当作当前 Lite 的稳定增益。

## 三个改进方向的评估

| 方向 | 现有实现/证据 | 判断与下一步 |
| --- | --- | --- |
| 1. Masked residual relation learning | `MaskedResidualGraphNet.masked_loss` 已随机遮蔽残差坐标，验证/推理做逐变量 leave-one-out；不是待新增模块。LFP Net 比 Lite 好，NCM F1 则略差。 | 保留为方法核心，但必须证明它超越相同残差上的线性精度矩阵。下一轮先测关系-only 的 AUROC/AP、等 FPR 召回、弱故障分层；若优势不稳定，先改条件分布评分/训练目标，不应只堆深度。 |
| 2. 残差图与输入 Feature-GAT 分工 | NCM 去掉输入 feature graph 后，Net AUROC/F1 从 0.8231/0.6810 降到 0.7889/0.6227（Kaggle v3 评估日志）；但这只证明输入图有用，不能单独证明残差图与输入图互补。 | 做同一冻结 backbone 的残差 head 对照：现有 graph+attention、去残差图、等参数 masked MLP/attention；固定 z、mask、训练预算、阈值与评分，避免把容量差当关系收益。 |
| 3. 工况降级为辅助调制 | NCM 无 FiLM F1 0.6827，完整 0.6810；无全部条件 0.6776。条件在正常 LOO 任务中有用，但最终排序收益小。 | 主模型可保留轻量条件均值校准与可关的调制；不把“工况解耦”写成已证实的主要贡献。Brand3 同时跑 Lite 无条件与 Net 无 FiLM，检验是否可删。 |

## 条件校准弱收益：已经能排除与定位什么

1. **尺度路径退化是确定的实现层原因。** LFP 与 NCM 的 `lite_continuous/calibration.npz` 中，七个变量的 `scale_low=scale_high=0.05`；所有预测尺度经训练 P5/P95 clip 和绝对下限后成为同一个常数。因而当前“条件尺度”不能随工况变化；相对 Raw 的排序变化主要来自条件均值扣除、后续联合关系与 CDF。
2. **条件均值并非完全无信息。** 正常验证窗口的线性均值预测 R²：LFP 七变量约 -0.013～0.443，NCM 约 0.008～0.502，差异很大。说明简单的 CURRENT/SOC 八统计量对某些响应残差有解释力，对电流残差等变量弱；不能笼统说“条件没用”。
3. **FiLM 不是没有学动。** 在正常验证窗口抽样，其 `|gamma|` 均值 LFP/NCM 为 0.086/0.071，`|beta|` 均值 0.086/0.045（架构上限 0.1）。冻结关系头后打乱条件向量，正常验证 LOO MSE 分别上升约 13.4%/12.2%（各抽 128 窗口）。因此更准确的问题是：条件改善正常残差预测，却未稳定改善故障/正常的排序和阈值分离；优化目标与检测目标可能错位。这是诊断，不是故障因果解释。
4. **Lite/Net 接近不是实现成同一个模型。** 两者共享完全相同的幅值分支，最终固定 0.5 融合；NCM 片段 combined 分数 Spearman=0.9938，P95 判定仅 47/3402 条不同，F1 为 0.6853/0.6810。LFP Spearman=0.9841、判定 103/3593 条不同，F1 为 0.4505/0.4882。CDF 饱和比例很低（NCM 测试 <0.02%，LFP 测试 <0.24%），不是相近性能的主要解释。
5. 下一轮应在**不看测试标签选超参**前提下，分别测：训练正常上的未裁剪预测 sigma 分布、低于 0.05 的比例；仅改变训练正常可校准的尺度下限；真实/打乱条件的关系-only 排序；固定相同 FPR 时 L1/L2 与低动态故障召回。不要在测试集上按 F1 反调 `min_std` 或融合权重。

## Brand3 两任务：主结果与对比

目标账号与原始数据集均为 `chenmjdaisy`；输入 `chenmjdaisy/tsinghua-ev-brand3`。使用车辆隔离的 `strict_normal_validation`、训练折 MinMax、127 点窗口、每片段一窗、片段/车辆两级预设 Top-5%、seed3407；正常验证车辆 P95/P99 定阈值，故障标签只用于测试指标。与 CH 32 点窗口是跨数据集机制验证，不宣称完全相同采样粒度。

| Kaggle 分片 | 折 | 每折 RCR 结果 |
| --- | --- | --- |
| `rcr-v1-brand3-strict-folds-024` | 0/2/4 | Raw、Scale、joint-only、RCR-Lite、Lite 无条件、RCR-Net、Net 无 FiLM |
| `rcr-v1-brand3-strict-folds-13` | 1/3 | 相同七项；无重复 fold |

对比模型**复用已完成的同协议五折**，不在新任务里浪费 GPU 重训：MTAD-GAT 五折车辆 AUROC/AP 均值为 0.9039/0.8169，TranAD 为 0.7086/0.6807；两者逐折均有 `metrics.json`、`vehicle_scores.csv`、`scores.npz`，协议为 strict normal / minmax / Top-5% / seed3407。二者是 10 epoch 旧预算，RCR 最多 50 epoch backbone + 30 epoch head，须以 † 标记，不作等预算架构优劣归因。两任务完成后按 fold 合并 AUROC、AP、P99 FPR/Recall/F1，并保存车辆原始分数以复核；未完成前不填结果。
