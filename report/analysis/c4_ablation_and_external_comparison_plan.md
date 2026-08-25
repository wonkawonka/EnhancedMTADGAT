# C4 消融与外部模型对比正式计划

> 状态（2026-08-26）：C4 v1 在 Brand3 fold 0 开发筛查中未通过。当前固定正物理图偏置显著降低 Graph-only 与 Full 的 AP/AUROC，因此本计划暂停；仅在 C4 graph v2 通过同一单折门槛后恢复执行。结果和协议核对见 `c4_brand3_fold0_local_screen_20260826.md`。

## 1. 研究范围

本计划只使用 Brand3 和 BMS。C4 由两条互补路径组成：动态物理关系先验以软偏置进入 Feature-GAT logits；状态条件响应一致性分支与 MTAD-GAT 主干并行训练并提供独立异常证据。两者均不与 C3 叠加，也不增加 GAT 注意力到物理图的一致性损失。随机种子固定为 3407、3408、3409。

Brand3 全部方法统一使用 Zhang et al. DyAD 的 `paper_protocol`：车辆折随机种子固定为 0；第 i 折正常车单独进入最终测试，其余正常车训练；第 i 折故障车与训练正常车组成带标签 calibration，剩余故障车与第 i 折正常车组成最终测试。归一化使用前 200 条训练片段的 `paper_channel`；calibration 在 5%--95% 搜索车辆 Top-p，并按公开 notebook 的有标签片段排序规则冻结阈值。当前预处理后的片段固定为 128 点，因此 `lookback=127` 时每片段恰有一个合法窗口；片段与窗口聚合在这里等价。旧 `strict_normal_validation + MinMax + 正常验证 P99` 结果仅保留为开发历史，不得进入 Brand3 主表。BMS 保持六簇边界和时间顺序，已有训练段的前 90% 用于建模、末尾 10% 用于正常校准，测试段不参与归一化、训练、早停、融合或阈值选择。

## 2. C4 消融

执行配置为 `configs/internal/105_c4_brand3_bms_formal_ablation_three_seed.json`，共 72 项：4 个实验臂 ×（Brand3 五折 + BMS）× 3 seeds。

| 实验臂 | Feature-GAT 物理偏置 | 响应一致性分支 | 回答的问题 |
| --- | --- | --- | --- |
| Baseline | 无 | 无 | MTAD-GAT 主干基准 |
| Graph-only | 动态软偏置 | 无 | 变量物理关系先验是否有效 |
| Response-only | 无 | 辅助训练 + 正常校准融合 | 状态条件响应一致性是否有效 |
| C4 Full | 动态软偏置 | 辅助训练 + 正常校准融合 | 两类物理信息是否互补 |

动态图在训练折 scaler 元数据可用时（Brand3）先还原工程单位；随后根据电流能量、电压极差、温度极差和 SOC 变化对基础物理边动态加权，再以 `lambda_g=0.5` 加到注意力 logits。该图没有可训练参数，也没有对应的一致性损失。

论文中的 C4 主方法只使用 `C4 Full`。其余三项只进入消融表，不在外部模型主表中作为独立方法重复计数。No-score、control-only 和 bidirectional 可在主结果成立后作为二级机理分析另行执行。

## 3. 外部模型对比

外部方法固定为 PCA/SPE、USAD、Isolation Forest、AE、Deep SVDD、LSTM-AD、GDN、TranAD、Anomaly Transformer、DCdetector 和 GANF，共 11 个。C4 Full 与这 11 个模型形成 12 方法主表。

- seed 3407：复用 `configs/external/07_unified_external_all_models_msl_smap_brand3_bms_seed3407.json` 中 Brand3/BMS 的 66 项结果。
- seed 3408、3409：运行 `configs/external/08_unified_external_brand3_bms_seed3408_3409.json`，新增 132 项。
- C4：使用 105 中 `brand3_c4_full` 和 `bms_c4_full` 的 18 项结果。
- 匹配键：`dataset + fold + seed`。禁止将旧 01/06 的开发协议结果混入本表。

## 4. 指标和结论边界

Brand3 主指标为车辆级 Average Precision（AP，即本文采用的 PR-AUC），同时报告车辆级 AUROC，以及 DyAD 带标签 calibration 所选阈值下的 F1、Precision、Recall。每个方法报告 5 folds × 3 seeds 的逐项结果、均值和标准差，并保存所选 Top-p 与阈值来源。

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

正式汇总前必须确认 105 为 72/72 完成、08 为 132/132 完成，并且 07 中用于复用的 Brand3/BMS seed3407 项均存在完整的 `config.json`、`metrics.json`、`runtime.json`、`scores.npz` 和模型文件。
