# 项目报告总览

## 实验计划盘点

| plan_type | plan_name | file_name | thesis_role | experiment_count | comment |
| --- | --- | --- | --- | --- | --- |
| internal | 00_kaggle_smoke | 00_kaggle_smoke.json | 只检查官方完整数据的索引、车辆级划分、训练、推理和结果打包链路；不作为论文结果。 | 2 |  |
| internal | 01_ch3_msl_generality | 01_ch3_main.json | 第三章非电池通用性补充；电池主结果统一由06计划执行 | 6 |  |
| internal | 05_condition_validation | 05_condition_validation.json | 动态状态表征有效性与真实BMS案例，不参与故障检测主表 | 3 | NASA Random Walk不是已删除的经典NASA数据；其原始工步码不进入模型，仅按电流方向做事后弱表征探针与跨电池稳定性检查。BMS测试段确认全程正常，仅比较状态条件化前后的经验误报率和跨簇/跨时段稳定性。 |
| internal | 06_kaggle_formal | 06_kaggle_formal.json | 正式主实验：完整三品牌数据，品牌独立建模，车辆级五折评估 | 60 |  |
| external | 01_nc_battery_official | 01_nc_battery_official.json | 新电池正式外部对照：五种通用无监督基线加同数据集专用DyAD | 90 |  |
| external | 02_nc_battery_paper_protocol | 02_nc_battery_paper_protocol.json | 协议复核：按Zhang等人Supplementary Note 2划分车辆，并使用带标签校准集选择阈值 | 60 |  |

## 消融设计矩阵

| plan_name | experiment_name | comment | dataset | model_name | use_transformer | use_regime_condition | regime_encoder_type | regime_aux_lambda | regime_condition_mode | score_fusion_mode | use_event_consistency | use_physical_state_encoding | use_physical_regularization | use_physical_response_score | physical_response_terms | nasa_train_batteries | nasa_test_batteries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06_kaggle_formal | battery_mtadgat_all | 忠实MTAD-GAT：训练和计分均覆盖全部通道。 | TSINGHUA_EV | mtad_gat |  |  | temporal | 0.0 | fusion | fixed |  |  |  |  |  |  |  |
| 06_kaggle_formal | battery_c3 | 第三章主模型：电流/SOC工况编码及融合表示FiLM条件化。 | TSINGHUA_EV | mtad_gat_c3_regime |  |  | temporal | 0.05 | fusion | quality_aware |  |  |  |  |  |  |  |
| 06_kaggle_formal | battery_c4 | 第四章主模型：C3加物理状态、响应一致性损失及物理响应计分。 | TSINGHUA_EV | mtad_gat_c4_physics |  |  | temporal | 0.05 | fusion | quality_aware |  |  |  |  |  |  |  |
| 06_kaggle_formal | battery_c3_no_condition | 核心消融仅在brand3完整五折：去掉工况条件化。 | TSINGHUA_EV | mtad_gat_c3_regime |  | False | temporal | 0.0 | fusion | quality_aware |  |  |  |  |  |  |  |
| 06_kaggle_formal | battery_c3_no_aux | 核心消融仅在brand3完整五折：去掉状态辅助任务。 | TSINGHUA_EV | mtad_gat_c3_regime |  |  | temporal | 0.0 | fusion | quality_aware |  |  |  |  |  |  |  |
| 06_kaggle_formal | battery_c4_state_only | 核心消融仅在brand3完整五折：保留物理状态编码，去掉物理损失与计分。 | TSINGHUA_EV | mtad_gat_c4_physics |  |  | temporal | 0.05 | fusion | quality_aware |  |  | False | False |  |  |  |

## 已生成分析报告

_暂无数据_

## 说明

- `plan_inventory.csv` 汇总当前 internal/external 计划。
- `ablation_matrix.csv` 汇总当前消融设计的模块开关矩阵。
- `analysis_inventory.csv` 汇总已生成的数据分析报告。
