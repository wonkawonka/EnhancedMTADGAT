# 项目报告总览

## 实验计划盘点

| plan_type | plan_name | file_name | thesis_role | experiment_count | comment |
| --- | --- | --- | --- | --- | --- |
| internal | 01_ch3_msl_main | 01_ch3_msl_main.json | 第三章公开基准主结果 | 2 | 第三章主结果计划：只保留 MSL 作为公开标准数据集，聚焦 baseline 与 c3 的主表对比。该计划对应论文第三章最核心的公开数据结果。 |
| internal | 02_ch3_msl_ablation | 02_ch3_msl_ablation.json | 第三章模块消融 | 5 | 第三章消融计划：只在 MSL 上完成关键模块消融，保证消融结论集中、篇幅可控且便于答辩展开。 |
| internal | 03_ch4_nasa_random_main | 03_ch4_nasa_random_main.json | 第四章主结果 | 3 | 第四章主结果计划：NASA_RANDOM_DISCHARGE 作为核心电池主线，使用 RW1、RW2、RW7、RW8 四个随机工况序列比较 baseline、c3 与 c3+physics 三个层级。 |
| internal | 04_ch4_nasa_random_physics_ablation | 04_ch4_nasa_random_physics_ablation.json | 第四章物理增强消融 | 2 | 第四章物理增强消融计划：只保留 NASA_RANDOM_DISCHARGE 上最关键的两个物理模块拆分，控制实验规模同时保证可解释性。 |
| internal | 05_chbattery_supplement | 05_chbattery_supplement.json | 电池补充验证 | 3 | 补充验证计划：CH-BATTERY 用于证明窗口级模型向样本级电池故障判别迁移时的有效性。该计划不再单独拆分多个文件，而是集中保留最需要写进论文的三组结果。 |
| external | 01_ch3_msl_external | 01_ch3_msl_external.json | 第三章外部基线对比 | 4 | 第三章外部基线计划：只保留 MSL 上最常见、最能支撑论文主表的四类对比模型。 |
| external | 02_ch4_nasa_random_external | 02_ch4_nasa_random_external.json |  | 12 | 第四章 NASA_RANDOM_DISCHARGE 外部基线计划。围绕 RW1、RW2、RW7、RW8 四个随机工况序列，保留 TranAD、GDN、DCdetector 三类代表性基线，兼顾可比性与重跑成本。 |

## 消融设计矩阵

| plan_name | experiment_name | comment | dataset | model_name | use_transformer | use_regime_condition | use_revin | score_fusion_mode | use_event_consistency | use_physical_state_encoding | use_physical_regularization | nasa_train_batteries | nasa_test_batteries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 02_ch3_msl_ablation | msl_no_transformer | 关闭 Transformer 残差分支，并将状态注入方式切回 fusion。 | MSL | mtad_gat | False | True | True | quality_aware | True | False | False |  |  |
| 02_ch3_msl_ablation | msl_no_regime | 关闭状态感知调制。 | MSL | mtad_gat | True | False | True | quality_aware | True | False | False |  |  |
| 02_ch3_msl_ablation | msl_no_revin | 关闭 RevIN。 | MSL | mtad_gat_c3 | True | True | False | quality_aware | True | False | False |  |  |
| 02_ch3_msl_ablation | msl_fixed_fusion | 将 quality-aware 融合退回 fixed。 | MSL | mtad_gat_c3 | True | True | True | fixed | True | False | False |  |  |
| 02_ch3_msl_ablation | msl_no_event | 关闭事件一致性。 | MSL | mtad_gat_c3 | True | True | True | quality_aware | False | False | False |  |  |
| 04_ch4_nasa_random_physics_ablation | nasa_random_phys_encoding_only | 仅启用物理状态编码。 | NASA_RANDOM_DISCHARGE | mtad_gat_c3 |  |  |  |  |  | True | False | RW1,RW2,RW7,RW8 | RW1,RW2,RW7,RW8 |
| 04_ch4_nasa_random_physics_ablation | nasa_random_phys_reg_only | 仅启用物理正则化。 | NASA_RANDOM_DISCHARGE | mtad_gat_c3 |  |  |  |  |  | False | True | RW1,RW2,RW7,RW8 | RW1,RW2,RW7,RW8 |

## 已生成分析报告

_暂无数据_

## 说明

- `plan_inventory.csv` 汇总当前 internal/external 计划。
- `ablation_matrix.csv` 汇总当前消融设计的模块开关矩阵。
- `analysis_inventory.csv` 汇总已生成的数据分析报告。
