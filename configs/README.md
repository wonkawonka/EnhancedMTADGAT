# C3/C4 活动实验计划

99 号起进入正式实验；97、98 只保留为开发归档，不进入正式主表：

- `internal/99_swat_wadi_c3_three_group_three_seed.json`：SWAT/WADI 正式主结果
- `internal/100_c3_prototype_query_msl_smap_three_group_three_seed.json`：MSL/SMAP 正式主结果
- `internal/101_c3_prototype_query_brand3_fivefold_three_seed.json`：Brand3 正式五折主结果
- `internal/102_c3_prototype_query_brand2_fivefold_three_seed.json`：Brand2 正式五折主结果
- `internal/103_c3_core_ablation_msl_smap_brand3_fold1_three_seed.json`：正式核心消融；已删除 100/101 中重复的 baseline、restricted、prototype-query，只保留 shuffled_state_farthest 和 no_aux 两个新消融；文件名保留历史 fold1 命名，但 `plan_name` 已改为完整 Brand3 五折
- `internal/104_c3_bms_private_formal_three_seed.json`：私有 BMS 正常工况误报与稳定性

外部公共数据统一登记为 `external/06_c3_formal_external_comparison_msl_smap.json`，它合并 04 的 MSL 深度基线和 05 的 PCA/SPE、USAD 矩阵。04 的旧单次输出仍只能作开发记录；完成三 seed 和统一运行时输出后，才进入正式主表。原 04/05 文件保留为来源计划，历史 C3/C4 计划不得与 99 号后的正式结果混合。

示例：

```bash
.venv/bin/python run.py internal --plan configs/internal/89_swat_wadi_frozen_c3_restricted_film.json --dry-run
```
