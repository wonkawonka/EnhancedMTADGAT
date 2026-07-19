# Nature Communications battery baselines

电池主表改用论文 *Realistic fault detection of li-ion battery via dynamical deep learning* 的对照组：

- Isolation Forest（传统无监督）
- GDN
- AutoEncoder
- Deep SVDD
- LSTM-AD

DyAD 是该数据集原论文提出的专用模型，单列为领域强基线，不占上述五个通用基线名额。

官方源码：`https://github.com/962086838/Battery_fault_detection_NC_github`。

官方仓依赖 Python 3.6、PyTorch 1.5.1、CUDA 10.2，并且品牌路径和车辆折需要手工修改。本仓在 `src/models/nc_official_baselines.py` 集成公开架构和超参数的现代 PyTorch 兼容实现，由 `src/runners/train_nc_external.py` 统一训练和评估。正式运行不依赖外部克隆、旧版 PyG 或未发布的折文件；矩阵位于 `configs/external/01_nc_battery_official.json`。

```bash
python -m src.runners.run_external_baselines \
  --plan configs/external/01_nc_battery_official.json --dry-run
```

正式表使用项目内的 Isolation Forest、AE、Deep SVDD、GDN、LSTM-AD 和 DyAD。Isolation Forest 接收全部正常训练片段，并按算法标准让每棵树随机抽取至多 256 条样本；验证/测试仍全量计分。DyAD 保留三个品牌各自的隐藏维数、潜变量维数、训练轮数和损失权重。

这里不能表述为“原脚本原样运行”：官方代码要求旧版 CUDA/Python，并硬编码品牌路径；官方生成的 `ind_odd_dict*.npz.npy` 也未随仓库发布。准确表述是“基于公开代码架构和超参数的兼容实现，并在统一车辆划分和评价协议下重跑”。


公平性要求：三个品牌分别训练；同一车辆不得跨集合；缩放仅拟合正常训练车辆；所有模型复用同一 `brand/fold` 清单；主报车辆级 AUROC/AUPRC 和 TPR@FPR。原论文公布的数值可作为复现核对，不能和本仓不同划分的结果直接拼表。

MTAD-GAT 仍保留为本文内部主干基线，但 TranAD、Anomaly Transformer、DCdetector、GANF 不再作为电池主表的优先外部模型。
