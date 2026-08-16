# C3/C4 活动实验计划

当前只保留已经执行过且与最终冻结结构一致的四个内部计划：

- `internal/85_c3_restricted_state_public_quick_validation.json`
- `internal/86_c3_restricted_state_tsinghua_quick_validation.json`
- `internal/89_swat_wadi_frozen_c3_restricted_film.json`
- `internal/31_c4_physical_only_brand3_paper_fivefold_development.json`

85、86、89 统一比较 C3 的 `baseline / film_no_aux / film_true / film_shuffled`；31 只运行清华 C4 独立物理一致性分支。不要新建内容重复的“冻结计划”。

其余 C3/C4 历史计划已可恢复地归档到 `archive/c3_c4_legacy_20260816/configs/`。C3/C4 之外的 external 配置不属于本次清理范围。

示例：

```bash
.venv/bin/python run.py internal --plan configs/internal/89_swat_wadi_frozen_c3_restricted_film.json --dry-run
```
