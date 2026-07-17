# 项目报告总览

## 实验计划盘点

| plan_type | plan_name | file_name | thesis_role | experiment_count | comment |
| --- | --- | --- | --- | --- | --- |
| internal | 00_kaggle_smoke | 00_kaggle_smoke.json | Kaggle GPU、数据和完整链路冒烟检查，不进入论文表格 | 2 | 先运行本计划确认CUDA、Train.zip读取、训练、批量评分和报告输出，再运行正式50 epoch计划。 |
| internal | 01_ch3_main | 01_ch3_main.json | 第三章动态状态条件化时序异常检测主结果 | 12 | MSL验证通用时序异常检测能力；清华公开包按有标签充电片段评估，因缺少车辆ID不宣称车辆级泛化。 |
| internal | 02_ch3_regime_ablation | 02_ch3_regime_ablation.json | 第三章动态状态编码与条件化消融 | 6 | 统计量编码器仅作为消融基线；完整方法使用控制/状态通道上的时序卷积与注意力池化。 |
| internal | 03_ch4_tsinghua_main | 03_ch4_tsinghua_main.json | 第四章电-热物理响应一致性主结果 | 3 | 第四章严格继承第三章模型，只增加物理状态编码、响应一致性损失和自适应物理残差融合。 |
| internal | 04_ch4_physics_ablation | 04_ch4_physics_ablation.json | 第四章相对第三章的递进、模块消融与响应项消融 | 8 | 同一数据划分和随机种子下先做C3到C4的加法式递进，再对完整C4按电压、温度、电荷流和SOC-电流响应组做留一消融。 |
| internal | 05_condition_validation | 05_condition_validation.json | 动态状态表征有效性与真实BMS案例，不参与故障检测主表 | 2 | NASA Random Walk的原始工步码不进入模型，仅按电流方向做事后弱表征探针与跨电池稳定性检查；BMS用于静置/调频告警率和切换稳定性，不报告故障F1。 |
| internal | 06_kaggle_formal | 06_kaggle_formal.json | Kaggle正式实验最小闭环：完整数据、主结果三种子、消融单种子 | 28 | 只去除01-05计划之间的重复运行，不减少数据、不增大正式窗口步长、不降低训练上限。18个实验定义按主结果局部展开为28次运行。 |
| external | 01_ch3_msl_external | 01_ch3_msl_external.json | 第三章外部基线对比 | 4 | 第三章外部基线计划：只保留 MSL 上最常见、最能支撑论文主表的四类对比模型。 |

## 消融设计矩阵

| plan_name | experiment_name | comment | dataset | model_name | use_transformer | use_regime_condition | regime_encoder_type | regime_aux_lambda | regime_condition_mode | score_fusion_mode | use_event_consistency | use_physical_state_encoding | use_physical_regularization | use_physical_response_score | physical_response_terms | nasa_train_batteries | nasa_test_batteries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 02_ch3_regime_ablation | c3_full |  | TSINGHUA_EV | mtad_gat_c3_regime | True | True | temporal | 0.05 | fusion | quality_aware | False |  |  |  |  |  |  |
| 02_ch3_regime_ablation | c3_no_condition |  | TSINGHUA_EV | mtad_gat_c3_regime | True | False | temporal | 0.0 | fusion | quality_aware | False |  |  |  |  |  |  |
| 02_ch3_regime_ablation | c3_statistics_encoder |  | TSINGHUA_EV | mtad_gat_c3_regime | True | True | statistics | 0.0 | fusion | quality_aware | False |  |  |  |  |  |  |
| 02_ch3_regime_ablation | c3_no_regime_aux |  | TSINGHUA_EV | mtad_gat_c3_regime | True | True | temporal | 0.0 | fusion | quality_aware | False |  |  |  |  |  |  |
| 02_ch3_regime_ablation | c3_transformer_residual_gate |  | TSINGHUA_EV | mtad_gat_c3_regime | True | True | temporal | 0.05 | transformer_residual | quality_aware | False |  |  |  |  |  |  |
| 02_ch3_regime_ablation | c3_fixed_score_fusion |  | TSINGHUA_EV | mtad_gat_c3_regime | True | True | temporal | 0.05 | fusion | fixed | False |  |  |  |  |  |  |
| 04_ch4_physics_ablation | c3_reference |  | TSINGHUA_EV | mtad_gat_c3_regime | True | True | temporal | 0.05 | fusion |  | False | False | False | False | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 04_ch4_physics_ablation | c4_state_only |  | TSINGHUA_EV | mtad_gat_c4_physics | True | True | temporal | 0.05 | fusion |  | False | True | False | False | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 04_ch4_physics_ablation | c4_state_and_loss |  | TSINGHUA_EV | mtad_gat_c4_physics | True | True | temporal | 0.05 | fusion |  | False | True | True | False | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 04_ch4_physics_ablation | c4_full |  | TSINGHUA_EV | mtad_gat_c4_physics | True | True | temporal | 0.05 | fusion |  | False | True | True | True | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 04_ch4_physics_ablation | c4_without_voltage_response |  | TSINGHUA_EV | mtad_gat_c4_physics | True | True | temporal | 0.05 | fusion |  | False | True | True | True | temperature_rate,charge_flow,temperature_spread,soc_current_coupling |  |  |
| 04_ch4_physics_ablation | c4_without_temperature_response |  | TSINGHUA_EV | mtad_gat_c4_physics | True | True | temporal | 0.05 | fusion |  | False | True | True | True | voltage_rate,charge_flow,voltage_spread,soc_current_coupling |  |  |
| 04_ch4_physics_ablation | c4_without_charge_flow |  | TSINGHUA_EV | mtad_gat_c4_physics | True | True | temporal | 0.05 | fusion |  | False | True | True | True | voltage_rate,temperature_rate,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 04_ch4_physics_ablation | c4_without_soc_current_coupling |  | TSINGHUA_EV | mtad_gat_c4_physics | True | True | temporal | 0.05 | fusion |  | False | True | True | True | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread |  |  |
| 06_kaggle_formal | msl_mtadgat | 第三章通用基线；给C3提供同数据、同训练预算参照。 | MSL | mtad_gat |  |  | temporal | 0.0 | fusion |  | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | msl_c3_full | 验证C3在非电池数据上的通用性，报告三种子raw指标。 | MSL | mtad_gat_c3_regime |  |  | temporal | 0.0 | fusion |  | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_mtadgat | 清华电池主基线；用于判断C3/C4是否真正改善片段级检测。 | TSINGHUA_EV | mtad_gat |  |  | temporal | 0.05 | fusion |  | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c3_full | 第三章完整方法；报告三种子片段级AUPRC/AUROC及校准阈值指标。 | TSINGHUA_EV | mtad_gat_c3_regime |  |  | temporal | 0.05 | fusion |  | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c3_no_condition | 隔离状态条件化贡献；与C3 full之差回答状态向量是否有用。 | TSINGHUA_EV | mtad_gat_c3_regime |  | False | temporal | 0.0 | fusion |  | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c3_statistics_encoder | 检验学习式时序状态编码是否优于简单窗口统计量。 | TSINGHUA_EV | mtad_gat_c3_regime |  |  | statistics | 0.0 | fusion |  | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c3_no_aux | 检验动态描述辅助任务是否防止状态表征退化。 | TSINGHUA_EV | mtad_gat_c3_regime |  |  | temporal | 0.0 | fusion |  | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c3_residual_gate | 与fusion-FiLM对比，回答条件化位置是否比只调Transformer残差权重更合适。 | TSINGHUA_EV | mtad_gat_c3_regime |  |  | temporal | 0.05 | transformer_residual |  | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c3_fixed_score | 检验质量感知预测/重构分数融合是否优于固定权重。 | TSINGHUA_EV | mtad_gat_c3_regime |  |  | temporal | 0.05 | fusion | fixed | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c4_state_only | C3加物理状态编码，回答前向物理状态表征本身的贡献。 | TSINGHUA_EV | mtad_gat_c4_physics |  |  | temporal | 0.05 | fusion |  | False |  | False | False | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c4_state_loss | 再加响应一致性损失，主要观察响应MAE与检测指标是否改善。 | TSINGHUA_EV | mtad_gat_c4_physics |  |  | temporal | 0.05 | fusion |  | False |  |  | False | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c4_full | 完整第四章；三种子验证物理响应增强是否稳定有效。 | TSINGHUA_EV | mtad_gat_c4_physics |  |  | temporal | 0.05 | fusion |  | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c4_no_voltage | 留一消融电压变化率与电压极差响应。 | TSINGHUA_EV | mtad_gat_c4_physics |  |  | temporal | 0.05 | fusion |  | False |  |  |  | temperature_rate,charge_flow,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c4_no_temperature | 留一消融温度变化率与温度极差响应。 | TSINGHUA_EV | mtad_gat_c4_physics |  |  | temporal | 0.05 | fusion |  | False |  |  |  | voltage_rate,charge_flow,voltage_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c4_no_charge_flow | 留一消融累计电荷响应。 | TSINGHUA_EV | mtad_gat_c4_physics |  |  | temporal | 0.05 | fusion |  | False |  |  |  | voltage_rate,temperature_rate,voltage_spread,temperature_spread,soc_current_coupling |  |  |
| 06_kaggle_formal | tsinghua_c4_no_soc_current | 留一消融SOC-电流耦合响应。 | TSINGHUA_EV | mtad_gat_c4_physics |  |  | temporal | 0.05 | fusion |  | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread |  |  |
| 06_kaggle_formal | nasa_random_condition | 完整RW1/RW2/RW7训练、RW8跨电池弱探针；只验证状态嵌入，不评价故障检测。 | NASA_RANDOM_DISCHARGE | mtad_gat_c3_regime |  |  | temporal | 0.05 | fusion |  | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling | RW1,RW2,RW7 | RW8 |
| 06_kaggle_formal | bms_idle_frequency_regulation | 完整6簇数据；无标签案例只检查静置/调频、切换告警稳定性和高分案例。 | BMS | mtad_gat_c3_regime |  |  | temporal | 0.05 | fusion |  | False |  |  |  | voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling |  |  |

## 已生成分析报告

_暂无数据_

## 说明

- `plan_inventory.csv` 汇总当前 internal/external 计划。
- `ablation_matrix.csv` 汇总当前消融设计的模块开关矩阵。
- `analysis_inventory.csv` 汇总已生成的数据分析报告。
