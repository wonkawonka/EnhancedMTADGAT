# 项目报告总览

## 实验计划盘点

| plan_type | plan_name | file_name | thesis_role | experiment_count | comment |
| --- | --- | --- | --- | --- | --- |
| internal | 00_kaggle_smoke | 00_kaggle_smoke.json | 只检查官方完整数据的索引、车辆级划分、训练、推理和结果打包链路；不作为论文结果。 | 2 |  |
| internal | 01_ch3_msl_generality | 01_ch3_main.json | 第三章非电池通用性补充；电池主结果统一由06计划执行 | 6 |  |
| internal | 05_condition_validation | 05_condition_validation.json | 动态状态表征有效性与真实BMS案例，不参与故障检测主表 | 3 | NASA Random Walk不是已删除的经典NASA数据；其原始工步码不进入模型，仅按电流方向做事后弱表征探针与跨电池稳定性检查。BMS测试段确认全程正常，仅比较状态条件化前后的经验误报率和跨簇/跨时段稳定性。 |
| internal | 06_kaggle_formal | 06_kaggle_formal.json | 正式主实验：完整三品牌数据，品牌独立建模，车辆级五折评估 | 60 |  |
| internal | 07_c4_outer_gate_diagnostic | 07_c4_outer_gate_diagnostic.json | 开发诊断，不进入正式主结果或消融表。用于检验以工况门控的 Transformer 外层残差，是否能缓解 C4 物理状态直接注入 Transformer 输入造成的表征干扰。 | 2 |  |
| internal | 08_c4_state_gate_diagnostic | 08_c4_state_gate_diagnostic.json | 开发诊断，不进入正式主结果。只检验物理状态的直接注入方式：将 C4 的物理状态残差改为零初始化的可学习门控，同时保持 C3 的 fusion 条件化、GRU/Transformer 融合与计分不变。 | 1 |  |
| internal | 09_c4_outer_state_gate_confirmation | 09_c4_outer_state_gate_confirmation.json | 开发确认，不进入正式主结果。复核外层 GRU–Transformer 门控下，零初始化门控物理状态残差是否优于对应 C3 外层参照。 | 1 |  |
| internal | 10_c4_outer_direct_state_confirmation | 10_c4_outer_direct_state_confirmation.json | 开发确认，不进入正式主结果。公平复核外层 GRU–Transformer 门控下的直接物理状态注入，排除新增物理编码器改变共享路径随机初始化的影响。 | 1 |  |
| internal | 11_c4_control_state_diagnostic | 11_c4_control_state_diagnostic.json | 开发诊断，不进入正式主结果。物理状态仅使用可控量（相位、电流、累计电荷、SOC），避免把待检测的电压/温度响应重复作为状态输入。 | 1 |  |
| internal | 12_c4_shared_physical_feature_exploration | 12_c4_shared_physical_feature_exploration.json | 第四章开发探索，不进入正式主结果。保持 C3 主干，比较三种响应物理特征的网络内融合方式。 | 3 |  |
| internal | 13_c4_control_response_physics_exploration | 13_c4_control_response_physics_exploration.json | 第四章开发探索：按 Zhang et al. 的控制—响应动机，在同一 C3 主干中比较响应目标与物理注意力。 | 3 |  |
| internal | 14_c4_brand3_paper_fivefold_development | 14_c4_brand3_paper_fivefold_development.json | 第四章小规模本机开发验证：仅 brand3 五折，不扩展到其他品牌。 | 20 |  |
| internal | 15_c4_control_response_decoder_development | 15_c4_control_response_decoder_development.json | 第四章 brand3 五折开发：不再注入响应统计量，改为仅由电流/SOC历史条件化响应解码器。 | 10 |  |
| external | 01_nc_battery_official | 01_nc_battery_official.json | 新电池正式外部对照：五种通用无监督基线加同数据集专用DyAD | 90 |  |
| external | 02_nc_battery_paper_protocol | 02_nc_battery_paper_protocol.json | 协议复核：按Zhang等人Supplementary Note 2划分车辆，并使用带标签校准集选择阈值 | 60 |  |
| external | 03_dyad_brand3_paper_fivefold_development | 03_dyad_brand3_paper_fivefold_development.json | 与内部开发计划14配套的brand3 DyAD五折对照，不扩展其他品牌。 | 5 |  |

## 消融设计矩阵

| plan_name | experiment_name | comment | dataset | model_name | use_transformer | use_regime_condition | regime_encoder_type | regime_aux_lambda | regime_condition_mode | score_fusion_mode | use_event_consistency | use_physical_state_encoding | physical_state_injection_mode | physical_state_feature_mode | use_physical_response_features | physical_feature_fusion_mode | battery_response_only_training | use_physical_regularization | use_physical_response_score | physical_response_terms | nasa_train_batteries | nasa_test_batteries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06_kaggle_formal | battery_mtadgat_all | 忠实MTAD-GAT：训练和计分均覆盖全部通道。 | TSINGHUA_EV | mtad_gat |  |  | temporal | 0.0 | fusion | fixed |  |  |  |  |  |  |  |  |  |  |  |  |
| 06_kaggle_formal | battery_c3 | 第三章主模型：电流/SOC工况编码及融合表示FiLM条件化。 | TSINGHUA_EV | mtad_gat_c3_regime |  |  | temporal | 0.05 | fusion | quality_aware |  |  |  |  |  |  |  |  |  |  |  |  |
| 06_kaggle_formal | battery_c4 | 第四章主模型：C3加物理状态、响应一致性损失及物理响应计分。 | TSINGHUA_EV | mtad_gat_c4_physics |  |  | temporal | 0.05 | fusion | quality_aware |  |  |  |  |  |  |  |  |  |  |  |  |
| 06_kaggle_formal | battery_c3_no_condition | 核心消融仅在brand3完整五折：去掉工况条件化。 | TSINGHUA_EV | mtad_gat_c3_regime |  | False | temporal | 0.0 | fusion | quality_aware |  |  |  |  |  |  |  |  |  |  |  |  |
| 06_kaggle_formal | battery_c3_no_aux | 核心消融仅在brand3完整五折：去掉状态辅助任务。 | TSINGHUA_EV | mtad_gat_c3_regime |  |  | temporal | 0.0 | fusion | quality_aware |  |  |  |  |  |  |  |  |  |  |  |  |
| 06_kaggle_formal | battery_c4_state_only | 核心消融仅在brand3完整五折：保留物理状态编码，去掉物理损失与计分。 | TSINGHUA_EV | mtad_gat_c4_physics |  |  | temporal | 0.05 | fusion | quality_aware |  |  |  |  |  |  |  | False | False |  |  |  |
| 07_c4_outer_gate_diagnostic | battery_c3_outer_gate | 外层门控 C3 参照：h_end = h_gru + gate(regime) * h_transformer。 | TSINGHUA_EV | mtad_gat_c3_regime |  |  | temporal | 0.05 | transformer_residual | quality_aware |  |  |  |  |  |  |  |  |  |  |  |  |
| 07_c4_outer_gate_diagnostic | battery_c4_outer_gate_state_only | 仅相对外层门控 C3 增加物理状态编码；关闭物理损失与物理响应计分。 | TSINGHUA_EV | mtad_gat_c4_physics |  |  | temporal | 0.05 | transformer_residual | quality_aware |  |  |  |  |  |  |  | False | False |  |  |  |

## 已生成分析报告

_暂无数据_

## 说明

- `plan_inventory.csv` 汇总当前 internal/external 计划。
- `ablation_matrix.csv` 汇总当前消融设计的模块开关矩阵。
- `analysis_inventory.csv` 汇总已生成的数据分析报告。
