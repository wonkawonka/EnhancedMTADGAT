# C3/C4 活动实验计划

99 号起进入正式实验；97、98 只保留为开发归档，不进入正式主表：

- `internal/99_swat_wadi_c3_three_group_three_seed.json`：SWAT/WADI 正式主结果
- `internal/100_c3_prototype_query_msl_smap_three_group_three_seed.json`：MSL/SMAP 正式主结果
- `internal/101_c3_prototype_query_brand3_fivefold_three_seed.json`：Brand3 正式五折主结果
- `internal/102_c3_prototype_query_brand2_fivefold_three_seed.json`：Brand2 正式五折主结果
- `internal/103_c3_core_ablation_msl_smap_brand3_fold1_three_seed.json`：正式核心消融；文件名保留历史 fold1 命名，但 `plan_name` 已改为完整 Brand3 五折
- `internal/104_c3_bms_private_formal_three_seed.json`：私有 BMS 正常工况误报与稳定性

外部公共数据正式对比为 `external/05_public_baselines_msl_smap.json`；旧的 `external/04_msl_external_baselines.json` 仅作仓库兼容性开发计划，未完成统一多 seed 输出前不进入主表。历史 C3/C4 计划保留在原位置或 `archive/`，不得与 99 号后的正式结果混合。

示例：

```bash
.venv/bin/python run.py internal --plan configs/internal/89_swat_wadi_frozen_c3_restricted_film.json --dry-run
```
