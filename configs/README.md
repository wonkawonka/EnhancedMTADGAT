# 实验计划

- `internal/00_kaggle_smoke.json`：官方电池数据小样本链路检查，不进论文表。
- `internal/01_ch3_main.json`：MSL 上 MTAD-GAT 与 C3 的通用性补充。
- `internal/05_condition_validation.json`：NASA Random 表征探针，以及BMS正常调频数据上条件化前后的误报与稳定性对照，不进故障主表。
- `internal/06_kaggle_formal.json`：电池正式主入口；完整三品牌、品牌独立、车辆级五折、主模型和核心消融。
- `internal/07_c4_outer_gate_diagnostic.json`：C4 开发诊断；仅 brand3/fold0 的外层门控 C3/C4 配对，不进入论文主结果。
- `internal/08_c4_state_gate_diagnostic.json`：C4 主开发诊断；仅对物理状态残差加零初始化门控，保持 C3 内层 fusion，不进入论文主结果。
- `internal/09_c4_outer_state_gate_confirmation.json`：C4 开发确认；外层门控下的可控物理状态残差，仅 brand3/fold0，不进入论文主结果。
- `internal/10_c4_outer_direct_state_confirmation.json`：C4 开发确认；外层门控下直接物理状态注入的公平复核，仅 brand3/fold0，不进入论文主结果。
- `internal/11_c4_control_state_diagnostic.json`：C4 开发诊断；仅使用电流/SOC 等控制量构造物理状态，仅 brand3/fold0，不进入论文主结果。
- `internal/12_c4_shared_physical_feature_exploration.json`：C4 主方向探索；响应物理特征在 C3 主干内的三种融合方式，仅 brand3/fold0。
- `internal/13_c4_control_response_physics_exploration.json`：按 Zhang et al. 控制—响应动机比较响应目标、共享 FiLM 与物理 Feature-GAT 边权；在 brand3/fold0 用带标签校准折按 AUROC 选择 Top-p，再固定到测试折。
- `internal/14_c4_brand3_paper_fivefold_development.json`：仅 brand3 的五折开发复核；统一论文归一化、论文车辆划分及 PR-AUC=AP，比较 MTAD-GAT、C3 和两种 C4。
- `internal/15_c4_control_response_decoder_development.json`：仅 brand3 的五折开发；把物理信息改为电流/SOC控制的响应解码器及可选控制—响应辅助约束。
- `external/03_dyad_brand3_paper_fivefold_development.json`：计划14配套的 brand3 DyAD 五折开发对照。
- `external/01_nc_battery_official.json`：严格正常校准协议下的五个通用基线（Isolation Forest、GDN、AE、Deep SVDD、LSTM-AD）和同数据集专用 DyAD。
- `external/02_nc_battery_paper_protocol.json`：按 Zhang et al. 2023 Supplementary Note 2 的带标签校准协议复核 DyAD、GDN、AE 和 Deep SVDD；不得与严格协议结果混表。

公平主对照按统一的严格车辆折运行；DyAD 单列为同数据集专用强基线，其余五项覆盖传统、图、重构、单分类和循环时序方法。论文协议复核使用一组故障车辆校准阈值，单独汇总。说明见 `external_baselines/BatteryFaultNC/README.md`。

```bash
python run.py preprocess --dataset TSINGHUA_EV
python run.py internal --plan configs/internal/00_kaggle_smoke.json
python run.py internal --plan configs/internal/06_kaggle_formal.json --resume --skip-existing
python run.py internal --plan configs/internal/07_c4_outer_gate_diagnostic.json --resume --skip-existing
python run.py internal --plan configs/internal/08_c4_state_gate_diagnostic.json --resume --skip-existing
python run.py internal --plan configs/internal/09_c4_outer_state_gate_confirmation.json --resume --skip-existing
python run.py internal --plan configs/internal/10_c4_outer_direct_state_confirmation.json --resume --skip-existing
python run.py internal --plan configs/internal/11_c4_control_state_diagnostic.json --resume --skip-existing
python run.py internal --plan configs/internal/12_c4_shared_physical_feature_exploration.json --resume --skip-existing
python run.py internal --plan configs/internal/13_c4_control_response_physics_exploration.json --resume --skip-existing
python run.py external --plan configs/external/01_nc_battery_official.json --skip-existing
python run.py external --plan configs/external/02_nc_battery_paper_protocol.json --skip-existing
```

执行顺序为：路径核对 → Tsinghua 索引预处理 → GPU/data/model preflight → smoke →
内部正式计划 → 外部严格协议 → 论文协议复核 → 完整批次归档。Tsinghua 原始片段
无需复制或重采样；按折训练集拟合的归一化不属于可提前合并的全局预处理。
