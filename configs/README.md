# 实验计划说明

当前配置按“第三章动态状态条件化、第四章电-热响应一致性”的论文主线组织。

## 内部计划

- `configs/internal/00_kaggle_smoke.json`
  - 正式占用 GPU 前的两轮小样本链路检查，不进入论文表格。
- `configs/internal/01_ch3_main.json`
  - MSL 通用异常检测与清华 EV 充电片段级故障主结果。
- `configs/internal/02_ch3_regime_ablation.json`
  - 学习式动态状态编码、FiLM 条件化位置、统计量编码和分数融合消融。
- `configs/internal/03_ch4_tsinghua_main.json`
  - 第三章模型与第四章电-热物理响应增强模型的递进比较。
- `configs/internal/04_ch4_physics_ablation.json`
  - 第三章到第四章的加法递进，以及电压、温度、电荷流、SOC-电流响应的留一消融。
- `configs/internal/05_condition_validation.json`
  - NASA Random Walk 电流方向表征探针与 BMS 静置/调频案例，不进入故障检测主表。
- `configs/internal/06_kaggle_formal.json`
  - Kaggle 正式整合计划；完整数据和正式训练预算，只去重，主结果三种子、消融单种子。

清华公开包没有车辆 ID，因此不能报告车辆级泛化结果。CH-BATTERY 和 2024 EV Fault Dataset 不在当前论文实验计划中。NASA Random Walk 没有异常标签，不报告异常检测 F1；BMS 没有可靠告警标签，只报告运行稳定性和误报警统计。

正式 C3/C4 使用 `regime_condition_mode=fusion`：状态向量调制关系融合表示，再共同送入 GRU 和 Transformer。MSL 用非目标通道作为潜在上下文并关闭电池描述辅助任务；BMS 用 `SYS_I` 和簇 SOC 表示运行条件，簇电流仍参与异常评分。

## 外部计划

- `configs/external/01_ch3_msl_external.json`
  - MSL 上的 TranAD、Anomaly Transformer、GDN 和 DCdetector 对比。

## 运行顺序

```text
1. 00_kaggle_smoke.json
2. 06_kaggle_formal.json（正式整合入口）
3. 01_ch3_msl_external.json（外部基线，独立环境）
```

## 常用命令

```bash
.venv/bin/python -m src.runners.compare_experiments --plan configs/internal/06_kaggle_formal.json --resume --skip-existing
.venv/bin/python -m src.runners.run_external_baselines --plan configs/external/01_ch3_msl_external.json --skip-existing
```

完整的指标、数据集角色和论文表述见 `report/thesis_experiment_design.md`。
