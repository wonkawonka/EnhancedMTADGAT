﻿## 项目说明

这是一个面向复杂运行状态下电池异常检测的 `MTAD-GAT` 扩展项目。第三章实现连续动态状态条件化模型，第四章在其上增加电-热响应一致性约束。

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

项目按操作系统使用仓库内的 .venv 解释器。Linux 路径为 .venv/bin/python，Windows 路径为 .venv\Scripts\python.exe；VS Code 已配置为自动从 .venv 选择当前系统的解释器。

Linux 下直接调用：

```powershell
.venv/bin/python -m pip install -r requirements.txt
```

Kaggle 正式内部实验和电池外部对照都保留平台预装的 CUDA/Torch，只安装主项目依赖：

```bash
pip install -r requirements-kaggle-main.txt
```

`requirements-kaggle-cu118.txt` 只留给未纳入正式计划的旧版第三方 DGL/PyG 仓库，不要在当前正式环境安装。

Windows 下将解释器替换为 .venv\Scripts\python.exe。

后续命令默认都可以写成：

```powershell
.venv/bin/python -m <module>
```

## 常用命令

### 数据预处理

```powershell
.venv/bin/python -m src.runners.preprocess --dataset MSL
.venv/bin/python -m src.runners.preprocess --dataset SMD --group 1-1
.venv/bin/python -m src.runners.preprocess --dataset BMS
```

### 单次训练

```powershell
.venv/bin/python -m src.runners.train --dataset MSL
.venv/bin/python -m src.runners.train --dataset BMS
.venv/bin/python -m src.runners.train_nc_battery --dataset TSINGHUA_EV --battery_brand 3 --battery_fold 0 --model_name mtad_gat_c3_regime --lookback 127
```

### 单次预测

```powershell
.venv/bin/python -m src.runners.predict --dataset MSL --model_id -1
```

### 批量运行主仓计划

```powershell
.venv/bin/python run.py preflight --tsinghua-ev-root datasets/TSINGHUA_EV
.venv/bin/python -m src.runners.compare_experiments --plan configs/internal/00_kaggle_smoke.json
.venv/bin/python -m src.runners.compare_experiments --plan configs/internal/01_ch3_main.json --skip-existing
.venv/bin/python -m src.runners.compare_experiments --plan configs/internal/06_kaggle_formal.json --resume --skip-existing
```

### 批量运行外部基线

```powershell
.venv/bin/python -m src.runners.run_external_baselines --plan configs/external/01_nc_battery_official.json --skip-existing
.venv/bin/python -m src.runners.run_external_baselines --plan configs/external/02_nc_battery_paper_protocol.json --skip-existing
```

`01` 是与内部模型共用的严格正常校准主对照；`02` 按 Zhang et al. 2023 Supplementary Note 2 使用带标签故障组校准阈值，只作论文协议复核。两套结果不能混表。外部计划使用五个通用基线（Isolation Forest、GDN、LSTM-AD、AE、Deep SVDD），并将同数据集专用 DyAD 单列为强基线；均由项目内入口运行，不需要另行克隆官方仓库。

### 数据分析与报告

```powershell
.venv/bin/python -m src.runners.analyze --dataset MSL
.venv/bin/python -m src.runners.analyze --dataset NASA_RANDOM_DISCHARGE --nasa_train_batteries RW1,RW2,RW7 --nasa_test_batteries RW8
.venv/bin/python -m src.runners.build_report
```

### 统一入口

```powershell
.venv/bin/python run.py internal --plan configs/internal/06_kaggle_formal.json --resume --skip-existing
.venv/bin/python run.py analyze --dataset NASA_RANDOM_DISCHARGE --nasa-train-batteries RW1,RW2,RW7 --nasa-test-batteries RW8
```

### 导出外部基线所需数据

```powershell
.venv/bin/python -m src.runners.prepare_external_baseline_data list --source-dataset MSL
.venv/bin/python -m src.runners.prepare_external_baseline_data export --source-dataset MSL --target tranad --output-dir .\tmp\tranad_msl
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
- 官方车辆级电池数据工具：`src/data/nc_battery.py`
- 官方车辆级电池训练入口：`src/runners/train_nc_battery.py`
- 工况标签推导：`src/data/regime_utils.py`
- 论文实验设计：`report/thesis_experiment_design.md`
- Kaggle GPU 执行手册：`report/kaggle_runbook.md`
- 第三/四章模型图：`report/figures/`

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
