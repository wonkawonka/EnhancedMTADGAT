# C4 消融与外部模型对比正式计划

## 1. 研究范围

本计划只使用 Brand3 和 BMS。C4 冻结为独立的控制量到响应量物理一致性分支，与 MTAD-GAT 主干并行训练；不与 C3 叠加。随机种子固定为 3407、3408、3409。

Brand3 使用 `strict_normal_validation` 车辆五折：训练、正常验证和测试车辆严格隔离；归一化器只在训练车辆拟合；车辆分数固定聚合 Top-5% 窗口。BMS 保持六簇边界和时间顺序，已有训练段的前 90% 用于建模、末尾 10% 用于正常校准，测试段不参与归一化、训练、早停、融合或阈值选择。

## 2. C4 消融

执行配置为 `configs/internal/105_c4_brand3_bms_formal_ablation_three_seed.json`，共 90 项：5 个实验臂 ×（Brand3 五折 + BMS）× 3 seeds。

| 实验臂 | 状态编码输入 | 状态 GRU | 一致性辅助训练 | 一致性分数融合 | 回答的问题 |
| --- | --- | --- | --- | --- | --- |
| Baseline | 无 | 无 | 无 | 无 | MTAD-GAT 主干基准 |
| C4 Full | 完整窗口 | 单向 | 有 | 正常校准，最大权重 0.35 | C4 总体是否有效 |
| C4 No-score | 完整窗口 | 单向 | 有 | 关闭 | 增益来自辅助训练还是独立评分 |
| C4 Control-only | 电流/SOC | 单向 | 有 | 正常校准 | 响应感知状态编码是否必要 |
| C4 Bidirectional | 完整窗口 | 双向 | 有 | 正常校准 | 单向因果状态编码是否优于历史双向版本 |

论文中的 C4 主方法只使用 `C4 Full`。其余三项只进入消融表，不在外部模型主表中作为独立方法重复计数。

## 3. 外部模型对比

外部方法固定为 PCA/SPE、USAD、Isolation Forest、AE、Deep SVDD、LSTM-AD、GDN、TranAD、Anomaly Transformer、DCdetector 和 GANF，共 11 个。C4 Full 与这 11 个模型形成 12 方法主表。

- seed 3407：复用 `configs/external/07_unified_external_all_models_msl_smap_brand3_bms_seed3407.json` 中 Brand3/BMS 的 66 项结果。
- seed 3408、3409：运行 `configs/external/08_unified_external_brand3_bms_seed3408_3409.json`，新增 132 项。
- C4：使用 105 中 `brand3_c4_full` 和 `bms_c4_full` 的 18 项结果。
- 匹配键：`dataset + fold + seed`。禁止将旧 01/06 的开发协议结果混入本表。

## 4. 指标和结论边界

Brand3 主指标为车辆级 Average Precision（AP，即本文采用的 PR-AUC），同时报告车辆级 AUROC，以及正常验证 P99 阈值冻结后的 F1、Precision、Recall。每个方法报告 5 folds × 3 seeds 的逐项结果、均值和标准差。

BMS 当前测试标签为全零占位，不报告 AP、AUROC、F1、Recall，也不声称验证了故障检出。只报告 P99 正常阈值下的 false alarm rate、每万窗口误报、分簇/分时间块/分工况误报波动以及阈值归一化分数稳定性。

同一数据集内，训练/验证/测试边界、seed、归一化拟合范围、阈值规则和聚合口径必须一致。窗口长度可以沿用各模型原始结构的原生设置，但必须在配置与结果中披露。

## 5. 执行顺序

```bash
python run.py internal \
  --plan configs/internal/105_c4_brand3_bms_formal_ablation_three_seed.json \
  --skip-existing

python run.py external \
  --plan configs/external/08_unified_external_brand3_bms_seed3408_3409.json \
  --skip-existing
```

正式汇总前必须确认 105 为 90/90 完成、08 为 132/132 完成，并且 07 中用于复用的 Brand3/BMS seed3407 项均存在完整的 `config.json`、`metrics.json`、`runtime.json`、`scores.npz` 和模型文件。
