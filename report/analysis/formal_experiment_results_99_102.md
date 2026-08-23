# 任务 99–102 正式实验结果及运行边界报告

生成日期：2026-08-23

代码提交：`445fbf679b88c2506432b85ba361b39f89d5b3a8`

硬件：Kaggle NVIDIA Tesla T4
主指标：原始逐点 AP、AUROC；Brand2/Brand3 使用车辆级 Vehicle PR-AUC 与 AUROC。

## 1. 结论摘要

| 任务 | 实验内容 | 结果状态 | 可正式使用范围 |
|---|---|---|---|
| 99 | SWAT/WADI，三模块 × 3 seeds | 18/18 完成 | 两数据集三 seed 原始结果 |
| 100 | MSL/SMAP，三模块 × 3 seeds | 18/18 完成 | MSL 正式结果；SMAP 仅去重后离线修正评估 |
| 101 | Brand3，三模块 × 5 folds × 3 seeds | 45/45 完成 | 三模块车辆级 15 次运行汇总 |
| 102 | Brand2，三模块 × 5 folds × 3 seeds | 15/45 后取消 | 仅 baseline 的 partial 结果 |

Brand3 的 restricted 汇总均值最高；MSL 的 restricted 三 seed 均值略高。SMAP 的 prototype-query 修正评估均值最高，但训练阶段含重复 `P-2`，不能作为干净重训的正式主结果。SWAT 没有稳定增益，WADI 的 AP 接近随机基线。

说明：表内 **粗体** 是同一数据集、同一可比口径下的最优值；不同数据集和评价层级不横向比较。

## 2. 实验协议

### 2.1 任务 99：SWAT/WADI

- 配置：[99_swat_wadi_c3_three_group_three_seed.json](../../configs/internal/99_swat_wadi_c3_three_group_three_seed.json)
- baseline、restricted、prototype-query；seeds 3407/3408/3409；最大 30 epochs，patience=3。
- 报告原始逐点 AP/AUROC 的均值和样本标准差。

### 2.2 任务 100：MSL/SMAP

- 配置：[100_c3_prototype_query_msl_smap_three_group_three_seed.json](../../configs/internal/100_c3_prototype_query_msl_smap_three_group_three_seed.json)
- 模块、seeds 与训练预算同任务 99。
- MSL 为原始正式结果。SMAP 训练时重复拼接 `P-2`；结果从保存分数中去除两个重复测试段，仅为去重后离线修正评估。

### 2.3 任务 101/102：Brand3/Brand2

- 配置：[101 Brand3](../../configs/internal/101_c3_prototype_query_brand3_fivefold_three_seed.json)、[102 Brand2](../../configs/internal/102_c3_prototype_query_brand2_fivefold_three_seed.json)。
- 5 folds × 3 seeds、最大 30 epochs、patience=0；指标为车辆级 Vehicle PR-AUC/AUROC。
- 101 完成 45 个运行，但逐 seed/fold 明细未保留；102 在 `restricted_f0_seed3407` 开始时取消，仅完成 baseline 15 个运行。

## 3. 任务 99：SWAT/WADI 按数据集、seed 结果

### 3.1 SWAT（seed=3407）

| 模块 | AP | AUROC |
|---|---:|---:|
| baseline | **0.711782** | 0.821064 |
| restricted | 0.703380 | 0.814591 |
| prototype-query | 0.708332 | **0.822471** |

### 3.2 SWAT（seed=3408）

| 模块 | AP | AUROC |
|---|---:|---:|
| baseline | 0.712271 | 0.817905 |
| restricted | **0.713207** | 0.820486 |
| prototype-query | 0.713007 | **0.828590** |

### 3.3 SWAT（seed=3409）

| 模块 | AP | AUROC |
|---|---:|---:|
| baseline | 0.703983 | 0.817097 |
| restricted | 0.697923 | 0.817134 |
| prototype-query | **0.707579** | **0.808484** |

### 3.4 WADI（seed=3407）

| 模块 | AP | AUROC |
|---|---:|---:|
| baseline | 0.054473 | 0.484923 |
| restricted | **0.071348** | **0.485988** |
| prototype-query | 0.047454 | 0.447598 |

### 3.5 WADI（seed=3408）

| 模块 | AP | AUROC |
|---|---:|---:|
| baseline | **0.045649** | **0.425274** |
| restricted | 0.044953 | 0.410615 |
| prototype-query | 0.045359 | 0.416840 |

### 3.6 WADI（seed=3409）

| 模块 | AP | AUROC |
|---|---:|---:|
| baseline | 0.045596 | 0.420585 |
| restricted | **0.047038** | **0.445057** |
| prototype-query | 0.046822 | 0.441681 |

### 3.7 SWAT/WADI 三 seed 均值

| 数据集 | 模块 | AP 均值±SD | AUROC 均值±SD |
|---|---|---:|---:|
| SWAT | baseline | 0.7093±0.0047 | 0.8187±0.0021 |
| SWAT | restricted | 0.7048±0.0077 | 0.8174±0.0030 |
| SWAT | prototype-query | **0.7096±0.0029** | **0.8198±0.0103** |
| WADI | baseline | 0.0486±0.0051 | 0.4436±0.0359 |
| WADI | restricted | **0.0544±0.0147** | **0.4472±0.0377** |
| WADI | prototype-query | 0.0465±0.0011 | 0.4354±0.0163 |

## 4. 任务 100：MSL/SMAP 按数据集、seed 结果

### 4.1 MSL（seed=3407）

| 模块 | AP | AUROC |
|---|---:|---:|
| baseline | 0.241877 | 0.690028 |
| restricted | 0.235344 | 0.668611 |
| prototype-query | **0.246917** | **0.716749** |

### 4.2 MSL（seed=3408）

| 模块 | AP | AUROC |
|---|---:|---:|
| baseline | 0.238683 | 0.686655 |
| restricted | 0.236533 | 0.673761 |
| prototype-query | 0.226862 | 0.649739 |

### 4.3 MSL（seed=3409）

| 模块 | AP | AUROC |
|---|---:|---:|
| baseline | 0.234781 | 0.662887 |
| restricted | **0.256787** | **0.726811** |
| prototype-query | 0.249174 | 0.701822 |

### 4.4 SMAP（seed=3407，去重后离线修正评估）

| 模块 | AP | AUROC |
|---|---:|---:|
| baseline | 0.123898 | 0.460341 |
| restricted | **0.144607** | **0.569694** |
| prototype-query | 0.142915 | 0.550530 |

### 4.5 SMAP（seed=3408，去重后离线修正评估）

| 模块 | AP | AUROC |
|---|---:|---:|
| baseline | 0.139869 | 0.539916 |
| restricted | 0.122110 | 0.451854 |
| prototype-query | **0.156015** | **0.604169** |

### 4.6 SMAP（seed=3409，去重后离线修正评估）

| 模块 | AP | AUROC |
|---|---:|---:|
| baseline | 0.141474 | **0.556354** |
| restricted | 0.139638 | 0.537366 |
| prototype-query | **0.148138** | 0.539124 |

### 4.7 MSL/SMAP 三 seed 均值

| 数据集 | 模块 | AP 均值±SD | AUROC 均值±SD |
|---|---|---:|---:|
| MSL | baseline | 0.2384±0.0036 | 0.6799±0.0148 |
| MSL | restricted | **0.2429±0.0121** | **0.6897±0.0322** |
| MSL | prototype-query | 0.2410±0.0123 | 0.6894±0.0352 |
| SMAP（修正评估） | baseline | 0.1351±0.0097 | 0.5189±0.0514 |
| SMAP（修正评估） | restricted | 0.1355±0.0118 | 0.5196±0.0609 |
| SMAP（修正评估） | prototype-query | **0.1490±0.0066** | **0.5646±0.0347** |

## 5. 任务 101：Brand3 五折三 seed 汇总

| 数据集 | 模块 | 实验数 | Vehicle PR-AUC 均值±SD | Vehicle AUROC 均值±SD |
|---|---|---:|---:|---:|
| Brand3 | baseline | 15 | 0.5196±0.1108 | 0.7209±0.0795 |
| Brand3 | restricted | 15 | **0.5581±0.1078** | **0.7449±0.0916** |
| Brand3 | prototype-query | 15 | 0.5221±0.0834 | 0.7248±0.0595 |

restricted 的汇总最高。101 没有保留逐 seed/fold 原始明细，不能补造明细或做配对显著性检验。

## 6. 任务 102：Brand2 partial baseline 结果

### 6.1 按 seed 汇总（每个 seed 为 5 folds 均值）

| 模块 | Vehicle PR-AUC（3407 / 3408 / 3409） | 总均值±SD | Vehicle AUROC（3407 / 3408 / 3409） | 总均值±SD |
|---|---|---:|---|---:|
| baseline | 0.858389 / 0.863496 / 0.861369 | 0.8611±0.0697 | 0.775641 / 0.803663 / 0.805861 | 0.7951±0.0895 |
| restricted | 未完成 | 未完成 | 未完成 | 未完成 |
| prototype-query | 未完成 | 未完成 | 未完成 | 未完成 |

### 6.2 baseline 按 fold、seed 明细

| Fold | Seed | Vehicle PR-AUC | Vehicle AUROC |
|---:|---:|---:|---:|
| 0 | 3407 | 0.965351 | 0.923077 |
| 0 | 3408 | 0.973250 | 0.935897 |
| 0 | 3409 | **0.988588** | **0.974359** |
| 1 | 3407 | 0.847075 | 0.703297 |
| 1 | 3408 | 0.831048 | 0.802198 |
| 1 | 3409 | 0.816783 | 0.747253 |
| 2 | 3407 | 0.886144 | 0.846154 |
| 2 | 3408 | 0.874914 | 0.782051 |
| 2 | 3409 | 0.868833 | 0.794872 |
| 3 | 3407 | 0.844573 | 0.703297 |
| 3 | 3408 | 0.860323 | 0.736264 |
| 3 | 3409 | 0.832418 | 0.703297 |
| 4 | 3407 | 0.748804 | 0.702381 |
| 4 | 3408 | 0.777947 | 0.761905 |
| 4 | 3409 | 0.800223 | 0.809524 |

粗体仅标出 baseline 15 个运行中的最高分，不构成模块对照。restricted/prototype-query 无结果，不能据此判断 C3 在 Brand2 是否有效。

## 7. 结果解释与论文使用边界

### 7.1 SMAP

SMAP 对随机 seed 和早停路径敏感；且本地旧缓存排除 `P-2`（53 条序列），Kaggle 100 训练重复加入 `P-2`（55 条序列）。当前仅在评价端去重，训练影响仍存在。因此 prototype-query 的 SMAP 均值增益只能作为暂定发现，需在唯一 channel loader 的干净训练协议下复核。

### 7.2 SWAT/WADI

SWAT 的 AP 约 0.705–0.710、AUROC 约 0.817–0.820，绝对检测性能并不弱，但 C3 没有稳定差异。WADI 的异常比例约 0.057736，随机排序 AP 接近该比例；当前 AP 为 0.0465–0.0544、AUROC 为 0.4354–0.4472，不能视为可靠增益。

### 7.3 可用于论文的结论

1. 可正式报告：99 的 SWAT/WADI 三 seed 原始结果、100 的 MSL 三 seed 结果、101 的 Brand3 15 次运行汇总结果。
2. 必须标注限制：100 的 SMAP 是“重复 `P-2` 训练 + 去重后离线评估”。
3. 不能正式下结论：102 的 Brand2 三模块比较；目前只有 baseline partial 结果。
4. 方法结论应收敛为数据集依赖的条件化收益，不能声称 C3 在所有数据集上优于 baseline。

任务 103 的新增机制消融与任务 06 的外部对比结果，请参见最新报告：[实验结果_103_06.md](实验结果_103_06.md)。
