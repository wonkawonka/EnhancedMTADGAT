# 实验计划说明

当前配置已经按论文主线收口，只保留最需要重跑的计划。

## 内部计划

- `configs/internal/01_ch3_msl_main.json`
  - 第三章主结果，公开标准数据只保留 `MSL`
- `configs/internal/02_ch3_msl_ablation.json`
  - 第三章关键模块消融，仍只在 `MSL` 上完成
- `configs/internal/03_ch4_nasa_random_main.json`
  - 第四章主结果，核心电池数据集为 `NASA_RANDOM_DISCHARGE`
- `configs/internal/04_ch4_nasa_random_physics_ablation.json`
  - 第四章物理增强消融，只保留 `NASA_RANDOM_DISCHARGE` 上最必要的两组
- `configs/internal/05_chbattery_supplement.json`
  - `CH-BATTERY` 补充验证，集中保留 baseline、c3、c3+physics 三组

## 外部计划

- `configs/external/01_ch3_msl_external.json`
  - 第三章外部基线，只保留 `MSL`
- `configs/external/02_ch4_nasa_random_external.json`
  - 第四章外部基线，围绕 `RW1/RW2/RW7/RW8`

## 运行顺序

建议按下面顺序重跑：

```text
1. 01_ch3_msl_main.json
2. 01_ch3_msl_external.json
3. 02_ch3_msl_ablation.json
4. 03_ch4_nasa_random_main.json
5. 02_ch4_nasa_random_external.json
6. 04_ch4_nasa_random_physics_ablation.json
7. 05_chbattery_supplement.json
```

## 常用命令

主仓计划：

```powershell
.\.python312\python.exe -m src.runners.compare_experiments --plan configs/internal/01_ch3_msl_main.json --skip-existing
.\.python312\python.exe -m src.runners.compare_experiments --plan configs/internal/03_ch4_nasa_random_main.json --skip-existing
.\.python312\python.exe -m src.runners.compare_experiments --plan configs/internal/05_chbattery_supplement.json --skip-existing
```

外部基线：

```powershell
.\.python312\python.exe -m src.runners.run_external_baselines --plan configs/external/01_ch3_msl_external.json --skip-existing
.\.python312\python.exe -m src.runners.run_external_baselines --plan configs/external/02_ch4_nasa_random_external.json --skip-existing
```
