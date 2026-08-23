# Kaggle 正式实验

## 1. 数据目录

Kaggle Input 中应存在已解压的数据目录（dataset slug 可以任意命名）：

- 新电池数据：父目录下包含 `battery_brand1/2/3`；
- BMS 原始数据：父目录下包含成组的 `BMS0Data`、`BMSnStatData`、
  `BMSnDetailTempData`、`BMSnDetailVoltData.xls` 文件。

代码会自动扫描 `/kaggle/input/<任意slug>/`、`TSINGHUA_EV/` 和
`datasets/TSINGHUA_EV/` 等常见层级，并通过 `label/` 与 `train/test/` 内容确认
真正的品牌目录。截图所示上传方式对应：

```text
/kaggle/input/dataset-bms/BMS/
/kaggle/input/tsinghua-ev/TSINGHUA_EV/
  battery_brand1/battery_brand1/{label,train,test,column.pkl}
  battery_brand2/battery_brand2/{label,train,test,column.pkl}
  battery_brand3/battery_brand3/{label,train,test,column.pkl}
```

双层 `battery_brandN` 已支持。实际 Kaggle slug 会转为小写或连字符也没关系，
以 Notebook 路径核对单元格打印的结果为准。若上传层级更特殊，可显式指定：

```bash
export MTAD_GAT_TSINGHUA_EV_ROOT=/kaggle/input/<slug>/<数据父目录>
export MTAD_GAT_BMS_ROOT=/kaggle/input/<slug>/<BMS父目录>
```

预处理与训练必须分开执行。新电池数据先建立经过校验的车辆/片段索引：

```bash
python run.py preprocess --dataset TSINGHUA_EV
```

索引写入可写的 `datasets/TSINGHUA_EV/processed/indices/`，不会尝试修改
Kaggle Input。原始片段已经是固定长度模型输入，因此无需重采样或另存一份张量；
每折归一化仍在训练入口中仅用该折训练车辆拟合，以避免数据泄漏。

BMS 上传的是原始 Excel，需要在训练 BMS 前单独执行一次：

```bash
python run.py preprocess --dataset BMS
```

Input 只负责读取原始文件；预处理结果写入当前仓库的
`datasets/BMS/processed/`（Kaggle 仓库位于 `/kaggle/working`，因此这里可写）。
后续 BMS 训练会自动优先读取这一目录，不需要重复预处理。

BMS 预处理完成后运行正式 C3 私有数据计划：

```bash
python run.py internal \
  --plan configs/internal/104_c3_bms_private_formal_three_seed.json \
  --resume --skip-existing
```

计划包含 baseline、restricted C3、prototype-query、farthest shuffled 和 no-aux 五个实验臂，
每臂三个 seed，共 15 个任务；窗口长度 100、stride=10。所有任务完成后会在批次目录自动生成
`bms_conditioning_comparison.json/csv`，并在每个实验目录生成按簇、时间块和
`BMSnI` 派生工况的误报报告。负差值表示条件化后的误报或波动更低；当前数据无故障标签，
因此不把这些结果解释为故障召回。

仓库或 Kaggle Dataset 中应存在：

```text
datasets/TSINGHUA_EV/
  battery_brand1/{data|train|test}/...
  battery_brand2/{data|train|test}/...
  battery_brand3/{data|train|test}/...
```

三个品牌包来自论文官方 Figshare。预处理阶段在项目可写缓存中生成
`battery_brandN_snippet_index.jsonl`；索引只保存路径、车辆 ID 和元数据，不复制数据。
后续五折直接复用。

## 2. 环境与冒烟

```bash
pip install -r requirements-kaggle-main.txt
python run.py internal --plan configs/internal/00_kaggle_smoke.json
```

冒烟每辆车只取 5 个片段，验证索引、车辆划分、训练、推理、车辆 top-5% 聚合和压缩包输出；不作为论文结果。

## 3. 正式运行

正式计划使用完整片段，不做采样：

```bash
python run.py internal \
  --plan configs/internal/40_final_c3_c4_clean_fivefold.json \
  --resume --skip-existing
```

矩阵名称为 `模型_b品牌_f折`。建议按品牌和折分会话，例如：

```bash
python run.py internal --plan configs/internal/40_final_c3_c4_clean_fivefold.json \
  --only baseline_mtad_gat_b3_f0,c3_four_regime_modules_b3_f0,c4_independent_physics_b3_f0 \
  --resume --skip-existing
```

主结果共有 3 个模型 × 3 个品牌 × 5 折 = 45 次。三个模型是原骨干、C3 四工况模块和 C4 独立物理分支；C3/C4 并列，不做叠加。C4 的有/无物理一致性评分由同一次推理报告，不重复训练。

原论文外部对照单独运行：

```bash
python run.py external \
  --plan configs/external/01_nc_battery_official.json \
  --skip-existing
```

外部矩阵为6种方法×3品牌×5折=90次训练与评估，其中包含五个通用基线和单列的 DyAD。输出稳定保存在 `runs/external/01_nc_battery_official/`，可用 `--only isolation_forest`、`--only dyad` 或具体名称分批运行。

六个模型均已集成到本项目，正式运行不需要克隆官方仓库，也不需要旧版 `torch-geometric` 或 `easydict`。统一入口使用相同的车辆折、数据路径和结果导出协议。

### 统一外部对比 07（seed=3407 首轮）

07 是论文后续统一外部主表的唯一首轮计划；旧 01/06 结果只保留为历史协议记录。它包含 11 个模型在 MSL、SMAP、BMS 和 Brand3 五折上的完整矩阵，共 88 项：

```bash
python run.py external \
  --plan configs/external/07_unified_external_all_models_msl_smap_brand3_bms_seed3407.json \
  --skip-existing
```

提交前可只展开命令，不训练：

```bash
python run.py external \
  --plan configs/external/07_unified_external_all_models_msl_smap_brand3_bms_seed3407.json \
  --dry-run --batch-tag submission-check
```

输出位于 `runs/external/07_unified_external_all_models_msl_smap_brand3_bms_seed3407/`。MSL/SMAP 写出逐点 AP、AUROC 和正常验证阈值下的原始 F1；Brand3 写车辆级 AP、AUROC、F1；BMS 只写正常工况误报、每万窗口误报以及分簇、时间块和工况稳定性，禁止把全零占位标签用于 AP/AUROC/F1。

### C4 Brand3/BMS 消融与外部对比

C4 正式消融使用 105：5 个实验臂 ×（Brand3 五折 + BMS）× 3 seeds，共 90 项。

```bash
python run.py internal \
  --plan configs/internal/105_c4_brand3_bms_formal_ablation_three_seed.json \
  --skip-existing
```

C4 外部主表的 seed3407 复用 07；其余两个 seed 使用 08，只运行 11 个外部模型的 Brand3/BMS，共 132 项：

```bash
python run.py external \
  --plan configs/external/08_unified_external_brand3_bms_seed3408_3409.json \
  --skip-existing
```

完整口径见 `report/analysis/c4_ablation_and_external_comparison_plan.md`。C4 主表只使用 105 的 `brand3_c4_full` 与 `bms_c4_full`；其他 C4 实验臂只进入消融表。

## 4. 每组实验的目的

| 比较 | 回答的问题 |
| --- | --- |
| MTAD-GAT-all vs response | 性能变化来自模型，还是仅来自去掉控制通道计分 |
| MTAD-GAT-response vs C3 | 电流/SOC工况条件化是否改善车辆故障排序 |
| MTAD-GAT-response vs C4 | 独立物理一致性分支是否增益 |
| C3历史反例（计划45/46） | FiLM/辅助任务与条件残差校准为何未通过；不在Kaggle重跑 |
| C4 response-only vs response+consistency | 独立物理分数的推理贡献 |
| Isolation Forest/GDN/AE/SVDD/LSTM-AD | 五类通用无监督基线的横向比较 |
| DyAD | 与同数据集原论文专用模型比较 |

车辆级报告统一把 PR-AUC 定义为 Average Precision（`average_precision_score`，并保留 `average_precision`/`auprc` 同值别名）；梯形积分另记为 `pr_auc_trapezoid`，不作为主指标。Zhang et al. 的鲁棒评分使用可调 Top-p 聚合，官方 notebook 在 5%–95% 间搜索，而不是固定 Top-5%；因此严格正常校准主表暂时锁定 Top-5% 以避免测试集调参，论文协议复核只在带标签校准折选择 p。验证集阈值 F1 只作辅助，不能用窗口级 F1 代替。

## 5. 时间与结果保存

完整数据超过 69 万个片段，正式实验以每个 128 点片段为一个样本（`lookback=127`），不再把一个片段展开成大量重叠窗口。配置上限 10 epoch、3 epoch 早停；论文 DyAD 本身按品牌只训练 3–5 epoch。先用 brand3 fold0 的首轮日志实测耗时，再按“首轮秒数 × 剩余轮数 × 剩余运行数”估算，不能沿用旧数据的时间。

每个实验输出 `model.pt`、`metrics.json`、`vehicle_scores.csv` 和检查点。每个实验
仍有自己的 zip，批处理结束还会在批次目录旁生成同名的完整批次 zip，包含
`logs/`、`output/`、`run_registry.json` 及所有实验产物：

```text
runs/internal/40_final_c3_c4_clean_fivefold/
runs/internal/40_final_c3_c4_clean_fivefold.zip
runs/external/01_nc_battery_official/
runs/external/01_nc_battery_official.zip
```

Notebook 最后的打包单元格会为任何尚无完整 zip 的旧批次/续跑批次补建同名外层
压缩包；不会再把所有批次重复套入一个超大的总 zip，以免浪费 Kaggle 20GB 工作区。
每次使用 `--resume --skip-existing`，会话结束前执行打包单元格。
