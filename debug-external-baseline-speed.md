# [OPEN] external-baseline-speed

## 目标
- 修复第三章外部基线 1 epoch 测速时的批量失败问题。

## 现象
- `ch3_external_baselines` 的 11 个实验全部结束，但 `success_count = 0`。
- 当前测速结果只能反映报错前耗时，不能作为真实训练速度。

## 初始假设
- 假设 1：`.python312` 环境缺少外部基线公共依赖，导致多个模型启动即失败。
- 假设 2：部分外部基线使用的 CLI 参数或数据集名字与当前适配层不一致，导致参数解析或数据加载失败。
- 假设 3：外部基线对 GPU / CUDA / torch 版本有额外要求，当前环境不满足。
- 假设 4：第三方仓库中的本地补丁不完整，输出目录或统一接口改造后仍有兼容性问题。
- 假设 5：不同模型存在不同根因，需要分别修复，但至少会有 1-2 个公共失败点。

## 证据记录
- `TranAD` 初始失败：`dgl` 在 torch 2.7.1 下缺少 graphbolt dll。
- `Anomaly-Transformer` 初始失败：`batch_size=256` 在 6GB GPU 上 OOM。
- `GDN` 初始失败：缺少 `pytz`；修复后又暴露 Windows 保存路径包含 `|` 和 `:`。
- `DCdetector` 初始失败：依次缺少 `statsmodels`、`tsfresh`、`hurst`、`arch`，随后又暴露 `np.Inf` 与 NumPy 2 不兼容。
- 代表验证结果：
  - `TranAD + SMAP + 1 epoch` 已跑通。
  - `GDN + SMAP + 1 epoch` 已跑通。
  - `Anomaly-Transformer + SMAP + 1 epoch` 已跑通。
  - `DCdetector + SMAP + 1 epoch` 已跑通。

## 修复记录
- `TranAD`
  - 将 `dgl` 改为仅在 `MTAD_GAT/GDN` 类初始化时懒加载，避免 `TranAD` 本体被无关依赖卡死。
  - 修复单标签与多维分数的维度兼容。
  - 修复 `pandas` 新版本移除 `DataFrame.append()` 的兼容问题。
- `Anomaly-Transformer`
  - 将第三章外部计划的 `batch_size` 从 `256` 下调到 `32`，适配 6GB GPU。
- `GDN`
  - 安装 `pytz`。
  - 修复模型保存时间戳格式，避免 Windows 非法文件名。
- `DCdetector`
  - 安装 `statsmodels`、`tsfresh`、`hurst`、`arch`。
  - 将 `np.Inf` 替换为 `np.inf`，兼容 NumPy 2。

## 验证记录
- 已完成 4 个代表实验的 post-fix 验证。
- 第三章外部 11 个实验的整套 1-epoch 复测已重新启动，待批量台账写出后做最终汇总。
