# 最终冻结的 C3/C4 路线

状态：冻结（2026-08-16）。后续实验只允许复现、补折和补数据集，不再改变结构定义。

## 1. 共同基线

Baseline 保持原始 MTAD-GAT：`Conv1D + Feature-GAT + Temporal-GAT + GRU + Forecast/Reconstruction`。C3 与 C4 都以该基线为对照，二者是平行路线，不能在同一个模型中同时开启。

## 2. C3：受约束状态 FiLM 三模块

C3 固定为以下三个不可替换的模块：

1. `RestrictedStateEncoder`：只编码允许的控制量或上下文状态；
2. fusion-level `FiLMConditioner`：只调制三路特征拼接后的表示，幅度由 `regime_film_scale` 限制；
3. 状态语义辅助损失：用 Smooth-L1 约束状态嵌入保留窗口统计语义。

正式四臂是 `baseline / film_no_aux / film_true / film_shuffled`。其中 shuffled 是批内错配状态通道的负对照，不是新模型路线。

数据集状态映射：清华使用电流和 SOC；MSL/SMAP 使用目标通道之外的匿名上下文并做置换不变池化；SWaT/WADI 使用当前全部通道。工业数据缺少明确控制—响应映射，因此其结果只说明该映射下的外推能力。

## 3. C4：独立控制量→响应量物理一致性分支

C4 固定为清华实验中生效的独立旁路：双向状态编码器读取正常窗口，经低维 VAE 信息瓶颈重建五个响应通道，训练目标为 Smooth-L1 一致性损失加小权重 KL。推理使用控制条件下的响应一致性残差，并限制其融合权重。

C4 的响应通道固定为 `[0, 3, 4, 5, 6]`，控制语义为电流/SOC；当前只定义于 `TSINGHUA_EV`。它不向 MTAD-GAT 共享骨干注入物理特征，也不包含 C3。

## 4. 明确排除的旧逻辑

正式代码和计划不再启用关系变化/关系原型、C3 joint NLL、Transformer/多尺度/RevIN 候选、共享物理注入、手工物理响应分数、控制响应 decoder、条件校准/条件图/路由 adapter、物理正则、稀疏图、normal-tail 或 Group-DRO。历史实现、计划和诊断报告统一放在 `archive/c3_c4_legacy_20260816/`，不进入自动执行入口。

## 5. 唯一活动计划

- `configs/internal/85_c3_restricted_state_public_quick_validation.json`：MSL/SMAP C3 四臂；
- `configs/internal/86_c3_restricted_state_tsinghua_quick_validation.json`：清华 C3 四臂；
- `configs/internal/89_swat_wadi_frozen_c3_restricted_film.json`：SWaT/WADI C3 四臂；
- `configs/internal/31_c4_physical_only_brand3_paper_fivefold_development.json`：清华 C4 五折。

这些沿用已跑计划，没有另建重复编号。

## 6. 权重兼容边界

模块属性名和 `state_dict` 键保留已有冻结 checkpoint 的布局。2026-08-16 已用 `strict=True` 验证 SWaT baseline、SWaT C3 film_true 和清华 C4 checkpoint；严格加载均无 missing/unexpected key，并完成前向。兼容指“旧模型可原样用于评分或续算”，不表示不同实验臂之间可以互相加载权重。
