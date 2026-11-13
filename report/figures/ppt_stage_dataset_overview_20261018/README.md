# PPT 阶段汇报：LFP/NCM 数据集概况图

本目录中的 01—04 图由现有全量审计汇总表生成；05 图只读取已登记的 96 个代表性原始片段。所有图均合并充电和放电片段，仅按 LFP/NCM 数据集分面。

## 图表

1. `01_class_distribution_lfp_ncm.png/pdf`：样本数量与类别分布。
2. `02_segment_length_distribution_lfp_ncm.png/pdf`：片段长度箱线图，纵轴为对数刻度。
3. `03_pca_segment_summary_lfp_ncm.png/pdf`：28 维段级统计量的分数据集 PCA。
4. `04_feature_correlation_heatmap_lfp_ncm.png/pdf`：七个变量段均值的 Pearson 相关性。
5. `05_control_response_raw_trends_lfp_ncm.png/pdf`：上排为 LFP/NCM 的控制量 `SUM_CURRENT`，下排为响应量 `MIN_CELL_VOLT`。

## 统计口径

- 源表：`report/analysis/ch_batterygen_full_data_audit/sample_summary.csv`。
- 有效片段：LFP 10,000；NCM 9,478。
- 充电、放电片段合并；故障类别仍区分正常、高内阻、低容量和自放电。
- PCA 在 LFP/NCM 内分别进行中位数补全和 RobustScaler，并将标准化值裁剪到 [-20, 20] 后做可视化 PCA；不可直接比较两个面板的坐标值。
- 热力图使用每个片段的七变量均值；它描述样本级共同变化，不是原始时间点相关，也不表示因果。
- 原始趋势图每个“化学体系 × 故障类别”含 6 个充电和 6 个放电片段。细线为单个片段，粗线为合并后 12 条轨迹在 256 个相对进度点上的逐点中位数。

## PPT 表述边界

- 类别柱状图可以支持“正常样本显著多于各类异常样本”。
- 长度图可以支持“样本为变长片段，长度跨度大”，不能据此断言长度差异就是故障机理。
- PCA 的极端值裁剪只为防止少量异常值压缩主体点云；PCA 只用于展示总体分布与重叠，不用于证明类别天然可分。
- 相关性热力图说明多变量依赖存在；合并充放电后，相关性同时包含工况差异。
- 原始趋势图的横轴是片段内相对进度，不是真实同步时间。`SUM_CURRENT` 的正负自然暴露运行方向；其合并中位线只表示混合分布的中心，不应解释为一条实际电流工况。

## PCA 方差解释率

- LFP：PC1 56.22%，PC2 15.97%。
- NCM：PC1 44.92%，PC2 14.79%。

配套统计 CSV 与 JSON 与图片同目录保存。
