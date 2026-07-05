﻿## 项目说明

这是一个基于 `MTAD-GAT` 扩展的多变量时间序列异常检测项目，当前已经完成第一阶段目录整理，便于后续继续重构和重做实验。

- 主仓实现集中在 `src/`
- 内部实验计划放在 `configs/internal/`
- 外部基线计划放在 `configs/external/`
- 批量实验输出统一放在 `runs/`
- 分析报告统一沉淀在 `report/`
- 第三方基线代码保留在 `external_baselines/`

## 当前目录

```text
mtad-gat-pytorch/
 src/
   models/      # 模型与网络模块
   data/        # 数据工具与数据集辅助
   engine/      # 训练、预测、评估核心逻辑
   runners/     # 命令行入口实现
 configs/
   internal/    # 主仓实验计划
   external/    # 外部基线实验计划
 report/         # 分析报告、模板和汇总文档
 datasets/       # 原始与预处理数据
 runs/           # 批量实验输出
└─ external_baselines/
```

## 环境使用

如果你本地使用仓库内的解释器，直接调用：

```powershell
.\.python312\python.exe -m pip install -r requirements.txt
```

后续命令默认都可以写成：

```powershell
.\.python312\python.exe -m <module>
```

## 常用命令

### 数据预处理

```powershell
.\.python312\python.exe -m src.runners.preprocess --dataset MSL
.\.python312\python.exe -m src.runners.preprocess --dataset SMAP
.\.python312\python.exe -m src.runners.preprocess --dataset SMD --group 1-1
```

### 单次训练

```powershell
.\.python312\python.exe -m src.runners.train --dataset MSL
.\.python312\python.exe -m src.runners.train --dataset SMAP
.\.python312\python.exe -m src.runners.train --dataset BMS
```

### 单次预测

```powershell
.\.python312\python.exe -m src.runners.predict --dataset MSL --model_id -1
```

### 批量运行主仓计划

```powershell
.\.python312\python.exe -m src.runners.compare_experiments --plan configs/internal/01_ch3_msl_main.json --skip-existing
.\.python312\python.exe -m src.runners.compare_experiments --plan configs/internal/03_ch4_nasa_random_main.json --skip-existing
```

### 批量运行外部基线

```powershell
.\.python312\python.exe -m src.runners.run_external_baselines --plan configs/external/01_ch3_msl_external.json --skip-existing
.\.python312\python.exe -m src.runners.run_external_baselines --plan configs/external/02_ch4_nasa_random_external.json --skip-existing
```

### 数据分析与报告

```powershell
.\.python312\python.exe -m src.runners.analyze --dataset MSL
.\.python312\python.exe -m src.runners.analyze --dataset NASA_RANDOM_DISCHARGE --nasa_train_batteries RW1,RW2,RW7,RW8 --nasa_test_batteries RW1,RW2,RW7,RW8
.\.python312\python.exe -m src.runners.build_report
```

### 统一入口

```powershell
.\.python312\python.exe .\run.py internal --plan configs/internal/03_ch4_nasa_random_main.json --skip-existing
.\.python312\python.exe .\run.py analyze --dataset CH_BATTERY_LFP_DISCHARGE
.\.python312\python.exe .\run.py full --internal-plan configs/internal/03_ch4_nasa_random_main.json --external-plan configs/external/02_ch4_nasa_random_external.json --analysis-dataset NASA_RANDOM_DISCHARGE --nasa-train-batteries RW1,RW2,RW7,RW8 --nasa-test-batteries RW1,RW2,RW7,RW8
```

### 导出外部基线所需数据

```powershell
.\.python312\python.exe -m src.runners.prepare_external_baseline_data list --source-dataset MSL
.\.python312\python.exe -m src.runners.prepare_external_baseline_data export --source-dataset MSL --target tranad --output-dir .\tmp\tranad_msl
```

## 输出位置

当前仓库统一使用 `runs/`：

- 内部批量实验输出到 `runs/internal/`
- 外部基线批量实验输出到 `runs/external/`
- 直接训练和预测输出到 `runs/manual/`

## 主要代码位置

- 参数定义：`src/args.py`
- 训练入口：`src/runners/train.py`
- 预测入口：`src/runners/predict.py`
- 预处理入口：`src/runners/preprocess.py`
- 主模型：`src/models/mtad_gat.py`
- 基础模块：`src/models/modules.py`
- 模型构建：`src/models/model_factory.py`
- 训练逻辑：`src/engine/training.py`
- 预测与阈值：`src/engine/prediction.py`
- 数据工具：`src/data/utils.py`
- CH-BATTERY 工具：`src/data/ch_battery_utils.py`

## 相关说明

- 主仓实验计划说明见 `configs/README.md`
- 报告目录说明见 `report/README.md`
- 外部基线说明见 `external_baselines/BASELINES.md`
- 第三方引用许可见 `licences/`

## 当前状态

- 已完成阶段一：整理结构，不改算法逻辑
- 入口统一为 `python -m src.runners.*`
- 已补充数据分析入口、项目报告汇总入口和根目录 `run.py`
- 历史分析材料和大部分旧实验结果已清理，适合重新组织实验主线
