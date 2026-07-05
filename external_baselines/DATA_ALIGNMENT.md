# 外部基线数据对齐

本项目现在以主仓 `datasets/**/processed/*.pkl` 作为统一数据源。

也就是说：

- 主项目自己的 `MTAD-GAT / c3 / c4` 直接读取 `processed`。
- 外部 baseline 优先直接回退到主仓 `processed`。
- 只有在某些旧仓强依赖固定文件格式时，才把 `src.runners.prepare_external_baseline_data` 当作备用导出工具。

备用导出脚本：

```powershell
.\.python312\python.exe -m src.runners.prepare_external_baseline_data
```

## 主仓 processed 现状

当前 `processed` 数据是规则二维数组：

- `SMAP_train.pkl` -> `(N, 25)`
- `MSL_train.pkl` -> `(N, 55)`
- `NASA_B0005_train.pkl` -> `(N, 7)`
- `NASA_RANDOM_DISCHARGE_RW1_train.pkl` -> `(N, 4)`
- `BMS_B14_3_2_cluster1_train.pkl` -> `(N, 35)`

因此它很适合作为统一导出源。

## 支持矩阵

### 1. Anomaly-Transformer

当前仓源码只直接支持：

- `SMAP`
- `MSL`
- `SMD`
- `PSM`

对应读取格式：

- `SMAP_train.npy`
- `SMAP_test.npy`
- `SMAP_test_label.npy`

或：

- `MSL_train.npy`
- `MSL_test.npy`
- `MSL_test_label.npy`

当前策略：

- 优先直读主仓 `datasets/data/processed/SMAP_*.pkl` 与 `MSL_*.pkl`
- 如果 `data_path` 下已经存在原始 `npy`，仍按原仓逻辑读取

说明：

- `SMAP / MSL` 可以直接对齐。
- 当前仓内已经补了主仓 `processed` 直读分支，可读取 `NASA / NASA_RANDOM / BMS` 命名的数据集。
- `CALCE` 仍未接入该仓的自定义数据集分支。

### 2. TranAD

当前仓源码直接支持：

- `SMAP`
- `MSL`
- `SMD`
- `UCR`
- `NAB`

其中：

- `SMAP` 默认读 `processed/SMAP/P-1_train.npy`
- `MSL` 默认读 `processed/MSL/C-1_train.npy`
- `SMD` 默认读 `processed/SMD/machine-1-1_train.npy`

当前策略：

- 优先直读主仓 `datasets/data/processed/SMAP_*.pkl` 与 `MSL_*.pkl`
- 如果外部目录下已有它原生 `processed/*.npy`，仍可兼容原仓逻辑

说明：

- `SMAP / MSL / SMD` 可以直接对齐。
- 当前仓内已经补了 `NASA / NASA_RANDOM / BMS` 的主仓 `processed` 直读逻辑。
- `CALCE` 仍不在当前适配范围内。

### 3. OmniAnomaly

当前仓源码直接支持：

- `SMAP`
- `MSL`
- `machine-*` 形式的 `SMD`

它读取的是：

- `processed/<dataset>_train.pkl`
- `processed/<dataset>_test.pkl`
- `processed/<dataset>_test_label.pkl`

当前策略：

- 优先直读主仓 `datasets/data/processed/SMAP_*.pkl` 与 `MSL_*.pkl`
- `machine-*` 类型仍走 `datasets/ServerMachineDataset/processed`

说明：

- `SMAP / MSL / SMD` 最容易直接对齐。
- 当前仓内已经补了 `NASA / NASA_RANDOM / BMS` 的主仓 `processed` 直读与维度支持。
- `CALCE` 仍未纳入当前适配范围。

### 4. GDN

当前仓比前三个更灵活。

它读取：

- `data/<dataset>/train.csv`
- `data/<dataset>/test.csv`
- `data/<dataset>/list.txt`

其中：

- `train.csv` 是正常训练数据
- `test.csv` 需要带 `attack` 列
- `list.txt` 记录特征名

当前策略：

- 若 `./data/<dataset>/train.csv` 存在，优先走原仓逻辑
- 若不存在，则自动回退到主仓 `processed`，并在内存中构造 `DataFrame + attack列 + feature_map`

说明：

- `GDN` 是当前最适合扩展到 `BMS / NASA / RANDOM` 的外部 baseline。
- 因为它只依赖 `csv + list.txt`，不强绑定 `SMAP / MSL / SMD` 这几个名字。
- 但它的图结构现在是用 `list.txt` 自动构建的，不是你论文里的物理先验图。

### 5. LSTM-AE

当前公开实现更偏：

- `SWaT`
- `AMPds2`

因此和你主仓数据接口不一致。

当前仍建议用备用导出脚本先对齐：

```powershell
.\.python312\python.exe -m src.runners.prepare_external_baseline_data export ^
  --source-dataset SMAP ^
  --target lstm_ae ^
  --output-dir external_baselines/LSTM-AE/project_data/SMAP ^
  --overwrite
```

说明：

- 这一步只是完成数据对齐。
- 真正让 `LSTM-AE` 跑你的 `processed`，仍需要额外补一个数据读取薄封装。

## 建议优先级

如果你现在要尽快做论文主对比，建议按下面优先级：

1. `TranAD`：先做 `SMAP / MSL`
2. `Anomaly-Transformer`：做 `SMAP / MSL`
3. `OmniAnomaly`：做 `SMAP / MSL`
4. `GDN`：先做 `SMAP / MSL`，再考虑 `BMS`
5. `LSTM-AE`：后补薄封装

## 备用命令

列出一个数据集下可导出的序列：

```powershell
.\.python312\python.exe -m src.runners.prepare_external_baseline_data list --source-dataset BMS
```

如果后续某个仓必须使用显式导出文件，可用：

```powershell
.\.python312\python.exe -m src.runners.prepare_external_baseline_data export ^
  --source-dataset SMAP ^
  --target tranad ^
  --output-dir external_baselines/TranAD/processed ^
  --target-dataset-name SMAP ^
  --target-series-prefix P-1 ^
  --overwrite
```

导出 `MSL` 给 `Anomaly-Transformer`：

```powershell
.\.python312\python.exe -m src.runners.prepare_external_baseline_data export ^
  --source-dataset MSL ^
  --target anomaly_transformer ^
  --output-dir external_baselines/Anomaly-Transformer/dataset/MSL ^
  --target-dataset-name MSL ^
  --overwrite
```

导出 `BMS cluster1` 给 `GDN`：

```powershell
.\.python312\python.exe -m src.runners.prepare_external_baseline_data export ^
  --source-dataset BMS ^
  --series-name BMS_B14_3_2_cluster1 ^
  --target gdn ^
  --output-dir external_baselines/GDN/data/bms_cluster1 ^
  --overwrite
```
