# 对比实验计划总览

## 当前有效计划

当前建议实际运行的计划一共 `6` 个：

- `ch3_main_results.json`
  - 第三章主结果。
  - 覆盖：`SMAP`、`MSL`、`NASA_RANDOM_DISCHARGE(RW1/RW2/RW7/RW8)`、`BMS` 的 `baseline/c3`。

- `ch3_external_baselines.json`
  - 第三章外部基线。
  - 覆盖：仅 `SMAP`、`MSL` 的 `TranAD / Anomaly-Transformer / GDN / DCdetector`。

- `ch3_ablation.json`
  - 第三章标准消融。
  - 覆盖：`SMAP`、`MSL` 的 `no_transformer / no_regime / no_revin / fixed_fusion / no_event`。

- `ch3_battery_ablation.json`
  - 第三章电池侧补充消融。
  - 覆盖：`RW1` 与 `BMS` 的小规模 c3 消融。

- `ch4_bms_main.json`
  - 第四章电池主结果。
  - 覆盖：`NASA_RANDOM_DISCHARGE RW1/RW2/RW7/RW8` 的 `c3+physics`，以及 `BMS c3+physics / c4+physics`。

- `ch4_bms_ablation.json`
  - 第四章消融。
  - 覆盖：`BMS` 与代表性 `RW1` 的物理增强/结构增强消融。

## 暂不运行

- `ch4_bms_external_baselines.json`
  - 当前不作为主线计划运行。
  - 原因：第四章不再重复做 `BMS` 外部基线，主叙事改为“第三章对外部，第四章做内部递进”。

- `plan_template.json`
  - 仅作模板参考，不直接运行。

## 运行口径

如果你现在只想把论文主线先跑通，先跑这 `3` 个：

```text
configs/compare/ch3_main_results.json
configs/compare/ch3_external_baselines.json
configs/compare/ch4_bms_main.json
```

如果你要把论文当前规划的主结果 + 消融一次性补齐，就跑这 `6` 个：

```text
configs/compare/ch3_main_results.json
configs/compare/ch3_external_baselines.json
configs/compare/ch3_ablation.json
configs/compare/ch3_battery_ablation.json
configs/compare/ch4_bms_main.json
configs/compare/ch4_bms_ablation.json
```

## 对应命令

主仓计划：

```bash
python compare_experiments.py --plan configs/compare/ch3_main_results.json --skip-existing
python compare_experiments.py --plan configs/compare/ch3_ablation.json --skip-existing
python compare_experiments.py --plan configs/compare/ch3_battery_ablation.json --skip-existing
python compare_experiments.py --plan configs/compare/ch4_bms_main.json --skip-existing
python compare_experiments.py --plan configs/compare/ch4_bms_ablation.json --skip-existing
```

外部基线：

```bash
python run_external_baselines.py --plan configs/compare/ch3_external_baselines.json --skip-existing
```
