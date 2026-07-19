# 实验计划

- `internal/00_kaggle_smoke.json`：官方电池数据小样本链路检查，不进论文表。
- `internal/01_ch3_main.json`：MSL 上 MTAD-GAT 与 C3 的通用性补充。
- `internal/05_condition_validation.json`：NASA Random 表征探针，以及BMS正常调频数据上条件化前后的误报与稳定性对照，不进故障主表。
- `internal/06_kaggle_formal.json`：电池正式主入口；完整三品牌、品牌独立、车辆级五折、主模型和核心消融。
- `external/01_nc_battery_official.json`：严格正常校准协议下的五个通用基线（Isolation Forest、GDN、AE、Deep SVDD、LSTM-AD）和同数据集专用 DyAD。
- `external/02_nc_battery_paper_protocol.json`：按 Zhang et al. 2023 Supplementary Note 2 的带标签校准协议复核 DyAD、GDN、AE 和 Deep SVDD；不得与严格协议结果混表。

公平主对照按统一的严格车辆折运行；DyAD 单列为同数据集专用强基线，其余五项覆盖传统、图、重构、单分类和循环时序方法。论文协议复核使用一组故障车辆校准阈值，单独汇总。说明见 `external_baselines/BatteryFaultNC/README.md`。

```bash
python run.py preprocess --dataset TSINGHUA_EV
python run.py internal --plan configs/internal/00_kaggle_smoke.json
python run.py internal --plan configs/internal/06_kaggle_formal.json --resume --skip-existing
python run.py external --plan configs/external/01_nc_battery_official.json --skip-existing
python run.py external --plan configs/external/02_nc_battery_paper_protocol.json --skip-existing
```

执行顺序为：路径核对 → Tsinghua 索引预处理 → GPU/data/model preflight → smoke →
内部正式计划 → 外部严格协议 → 论文协议复核 → 完整批次归档。Tsinghua 原始片段
无需复制或重采样；按折训练集拟合的归一化不属于可提前合并的全局预处理。
