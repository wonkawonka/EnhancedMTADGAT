# 外部基线说明

这里保留的是第三方基线仓与对齐说明，当前统一通过主仓脚本调度，不建议再手动分别到各个仓里临时敲命令。

## 当前分组

电池主数据使用五个通用基线：Isolation Forest、GDN、AE、Deep SVDD、LSTM-AD；同数据集论文提出的 DyAD 单列为领域强基线。详见 `BatteryFaultNC/README.md`。

下列通用时序仓只用于 MSL 等补充实验，不进入电池主表：

- `TranAD`
- `GDN`
- `DCdetector`
- `Anomaly-Transformer`
- `GANF`

## 统一调用

新电池正式外部基线通过统一运行脚本执行：

```bash
python -m src.runners.run_external_baselines --plan configs/external/01_nc_battery_official.json --dry-run
```

正式运行：

```bash
python -m src.runners.run_external_baselines --plan configs/external/01_nc_battery_official.json
```

只运行部分实验：

```bash
python -m src.runners.run_external_baselines --plan configs/external/01_nc_battery_official.json --only dyad
```

跳过已有输出：

```bash
python -m src.runners.run_external_baselines --plan configs/external/01_nc_battery_official.json --skip-existing
```

## 计划文件位置

外部基线计划统一放在 `configs/external/`。

每个计划文件主要包含：

- `plan_name`
  - 本次批量运行名称
- `common_env`
  - 所有实验共享环境变量
- `experiments`
  - 实验列表

每个实验常用字段：

- `name`
  - 实验名
- `baseline`
  - 基线名，对应 `external_baselines/` 下目录
- `script` 或 `command`
  - 实际执行入口
- `args`
  - 命令行参数
- `env`
  - 额外环境变量
- `cwd`
  - 可选工作目录
- `skip_if_exists`
  - 已有产物时的跳过标记

## 输出位置

当前外部基线批量输出统一写到：

```text
runs/external/<plan_name_timestamp>/
```

这样主仓内部实验和外部基线实验是分开的，后续清理结果也更方便。
