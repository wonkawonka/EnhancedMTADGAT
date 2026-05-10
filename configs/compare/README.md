# 对比实验计划总览

## 主线必跑

- `ch3_main_results.json`
  - 主仓主结果计划。
  - 用途：第三章主表。
  - 覆盖：`SMAP`、`MSL`、`NASA_RANDOM_DISCHARGE(RW1/RW2/RW7/RW8)`、`BMS`。

- `ch4_bms_main.json`
  - 第四章 `BMS` 主结果计划。
  - 用途：只跑第四章主结果，不重复跑 `BMS baseline/c3`。
  - 覆盖：`BMS c4` 主结果。

- `ch3_external_baselines.json`
  - 第三章外部基线计划。
  - 用途：补 `SMAP`、`MSL`、`NASA_RANDOM_DISCHARGE` 的外部对比。
  - 不含：`BMS`。

- `ch4_bms_external_baselines.json`
  - 第四章 `BMS-only` 外部基线计划。
  - 用途：专门补 `BMS` 的外部对比。
  - 覆盖：`TranAD`、`Anomaly-Transformer`、`OmniAnomaly`、`GDN`。

## 可选补充

- `ch3_ablation.json`
  - 第三章消融计划。
  - 用途：只有在你要补论文消融表时再跑。

- `ch4_bms_ablation.json`
  - 第四章消融计划。
  - 用途：第四章需要补机制分析和消融表时再跑。

- `plan_template.json`
  - 通用模板。
  - 用途：参考参数组织方式，不作为当前主线直接运行文件。

## 最小运行集合

如果你现在只想把论文主线先跑通，只需要这 4 个：

```text
configs/compare/ch3_main_results.json
configs/compare/ch4_bms_main.json
configs/compare/ch3_external_baselines.json
configs/compare/ch4_bms_external_baselines.json
```

## 对应命令

主仓主结果：

```bash
python compare_experiments.py --plan configs/compare/ch3_main_results.json --skip-existing
```

第四章 BMS 深化：

```bash
python compare_experiments.py --plan configs/compare/ch4_bms_main.json --skip-existing
```

第三章外部基线：

```bash
python run_external_baselines.py --plan configs/compare/ch3_external_baselines.json --skip-existing
```

第四章 BMS 外部基线：

```bash
python run_external_baselines.py --plan configs/compare/ch4_bms_external_baselines.json --skip-existing
```
