# 外部基线说明

本目录用于存放外部对比模型源码，原则上：

- 保留外部仓原始结构，不把源码直接揉进 `MTAD-GAT` 主项目。
- 主项目只负责统一管理目录、启动命令、日志和实验登记。
- 外部基线的训练入口、数据格式、输出文件各不相同，因此不要强行共用主项目的 `train.py`。

## 当前目录

- `GDN`
- `Anomaly-Transformer`
- `USAD`
- `TranAD`
- `LSTM-AE`
- `OmniAnomaly`

## 官方性说明

- `GDN`
  - 来源：`https://github.com/d-ailin/GDN`
  - 性质：论文作者公开实现
  - 入口：`main.py`

- `Anomaly-Transformer`
  - 来源：`https://github.com/thuml/Anomaly-Transformer`
  - 性质：论文作者公开实现
  - 入口：`main.py`

- `USAD`
  - 来源：`https://github.com/robustml-eurecom/usad`
  - 性质：作者团队公开实现
  - 入口：仓库主体是 `usad.py`，官方仓更偏最小实现与 notebook 示例

- `TranAD`
  - 来源：`https://github.com/imperial-qore/TranAD`
  - 性质：论文作者公开实现
  - 入口：`main.py`
  - 注意：仓库里也带了一些其它模型的复现版本，但论文精确复现仍优先建议使用各模型原仓

- `OmniAnomaly`
  - 来源：`https://github.com/NetManAIOps/OmniAnomaly`
  - 性质：论文作者公开实现
  - 入口：`main.py`

- `LSTM-AE`
  - 来源：手动下载的公开实现
  - 性质：经典 LSTM-AE 参考实现，不建议在论文里写成“作者官方仓”
  - 入口：`main.py`
  - 说明：更适合作为经典重构型基线参考代码

## 统一调用

主项目新增了统一外部基线运行脚本：

```bash
python run_external_baselines.py --plan configs/compare/ch3_external_baselines.json --dry-run
```

正式运行：

```bash
python run_external_baselines.py --plan configs/compare/ch3_external_baselines.json
```

只运行部分实验：

```bash
python run_external_baselines.py --plan configs/compare/ch3_external_baselines.json --only gdn_cpu_demo,tranad_smd_demo
```

如果某个实验已经生成目标目录，可配合：

```bash
python run_external_baselines.py --plan configs/compare/ch3_external_baselines.json --skip-existing
```

## 计划文件格式

推荐在 `configs/compare/` 下新建你自己的计划文件，字段说明如下：

- `plan_name`
  - 本次批量运行名称

- `common_env`
  - 所有实验共享环境变量

- `experiments`
  - 实验列表

每个实验支持这些字段：

- `name`
  - 实验名

- `baseline`
  - 已知外部基线名，可选值：
  - `GDN`
  - `Anomaly-Transformer`
  - `USAD`
  - `TranAD`
  - `LSTM-AE`
  - `OmniAnomaly`

- `cwd`
  - 可选。若不填，则默认使用 `external_baselines/<baseline>`

- `script`
  - Python 入口脚本，相对 `cwd`

- `args`
  - 命令行参数字典
  - 如果键本身带 `-` 或 `--`，脚本会按原样使用
  - 如果键不带前缀，会自动转成 `--key`

- `flags`
  - 只带开关、不带取值的参数列表，例如 `retrain`

- `positional_args`
  - 位置参数列表

- `command`
  - 完整命令数组；如果提供该字段，则优先使用，不再拼接 `script/args`

- `env`
  - 当前实验单独环境变量

- `skip_if_exists`
  - 可选。用于 `--skip-existing` 判断的标记路径

## 当前入口速查

- `GDN`
  - 典型入口：
  ```bash
  python main.py -dataset msl -device cpu -epoch 10
  ```

- `Anomaly-Transformer`
  - 典型入口：
  ```bash
  python main.py --dataset SMD --mode train --data_path ./dataset/SMD --model_save_path ./checkpoints/demo_smd
  ```

- `TranAD`
  - 典型入口：
  ```bash
  python main.py --model TranAD --dataset SMD --retrain
  ```

- `LSTM-AE`
  - 典型入口：
  ```bash
  python main.py --save_path ./result/demo_swat --data_path D:/your_dataset/SWaT --seq_length 60 --epoch 20
  ```

- `OmniAnomaly`
  - 典型入口：
  ```bash
  python main.py --dataset='MSL' --max_epoch=20
  ```

- `USAD`
  - 当前仓库更像最小实现与 notebook 示例，不像其它仓一样提供完整标准训练入口
  - 建议先把它视为“代码参考仓”，后续如果你要正式纳入批量对比，再单独补它的数据适配和训练脚本
