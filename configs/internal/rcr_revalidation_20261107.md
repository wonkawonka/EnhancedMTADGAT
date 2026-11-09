# RCR 从零训练复验计划

机器计划：[rcr_revalidation_20261107.json](rcr_revalidation_20261107.json)。

执行器：`src.runners.run_rcr_revalidation`。这份计划由专用执行器读取，不传给面向 MTAD-GAT 的旧 `compare_experiments`。

## 固定协议

- 数据：LFP、NCM、Brand3 五折，共7单元，完整窗口。
- 数据划分固定：CH split seed=3407，Brand3 fold seed=0；模型种子变化不改变划分和训练归一化。
- 模型种子：3407、3408、3409。
- 所有网络从零初始化；只有同一任务被中断时恢复该任务的检查点，不加载历史预训练权重，不做续训实验。
- 原骨干：7变量嵌入→VariableGraph→32维融合→2层/4头时间Transformer→32→32→7 Decoder。
- 预算：统一最多30轮、patience=8、AdamW lr=0.001、weight_decay=0.0001、batch=128、Huber重构损失，仅正常验证损失选最佳点。
- 评分：原RCR Scale，带符号残差的线性工况均值和log-RMS尺度；默认下限0.05。全量正常训练拟合校准，正常验证P95确定阈值。
- 不使用本轮Memory实验的cal_abs、额外20%校准VIN留出或隐空间评分；只有去校准消融使用未校准重构误差。
- 窗口→片段Top5%；Brand3再片段→车辆Top5%，全部向上取整且至少一个。
- 这是已多轮探索数据上的开发复验，不宣称独立确认。30轮上限不自动意味着与历史外部模型完整协议一致。

## 预登记实验矩阵

| 编号 | 变化 | 种子 | 数据单元 | 新训练数 |
| --- | --- | --- | ---: | ---: |
| R1 | 完整RCR | 3407/3408/3409 | 7 | 21 |
| R4 | 去VariableGraph | 3407/3408/3409 | 7 | 21 |
| R5 | 去时间Transformer | 3407/3408/3409 | 7 | 21 |
| R6 | 去校准，复用R1同权重 | 3407/3408/3409 | 7 | 0 |
| P1 | lr=0.0003/0.001/0.003 | 3407 | 7 | 14 |
| P2 | hidden=16/32/64 | 3407 | 7 | 14 |
| P3 | dropout=0/0.1/0.2 | 3407 | 7 | 14 |
| P4 | min_std=0.005/0.05/0.1，复用R1同权重 | 3407/3408/3409 | 7 | 0 |

共105个训练单元。P1—P3只改变一个参数，默认值复用R1的3407，不重复训练。
105个主评分＋21个去校准评分＋42个非默认尺度下限评分＝168个评价单元。
R6和P4均独立按自己的正常验证分数确定P95阈值；不复用不兼容分数的阈值。
不得按测试最高AP选择敏感性最优值后替换正式默认配置。

## 排队与执行

```bash
.venv/bin/python -m src.runners.run_rcr_revalidation \
  --plan configs/internal/rcr_revalidation_20261107.json --dry-run
```

CUDA执行必须在沙箱外；后台会话固定为`rcr_revalidation_20261107`，单GPU串行。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
MPLCONFIGDIR=/tmp/rcr-revalidation-mpl \
.venv/bin/python -u -m src.runners.run_rcr_revalidation \
  --plan configs/internal/rcr_revalidation_20261107.json \
  --execute --wait-predecessor
```

等待当前Memory/Decoder实验21个训练单元及28个评分单元全部完成，并核验每单元的指标、窗口分数、样本映射分数和阈值后开始。
前置任务失败则停止接续；运行器使用排他锁避免重复队列。
正式启动时固定plan、展开任务和实现源码哈希，运行中不得修改；协议改变必须新建输出目录。

## 产物与结论规则

输出：`runs/internal/rcr_revalidation/rcr_revalidation_20261107_v1/`。
报告：`report/analysis/rcr_revalidation/RCR_REVALIDATION_20261107.md`。

- 每单元保存初始权重哈希、最佳/最终权重、完整续跑检查点、训练曲线、实际epoch/优化步数、峰值显存和耗时。
- 保存全量窗口分数、片段/车辆分数、样本映射、校准参数、P95阈值、metrics.csv与metrics.json。
- 主指标AP、AUROC、P、R、F1、FPR；所有方法严格使用相同评分与聚合，去校准仅改变后处理。
- Brand3先求每个种子五折平均，再跨三种子报告均值±样本标准差；逐折结果完整保留。故障车辆跨折重复，不把五折当作五组独立故障车。
- 完成情况逐单元审核；未完整覆盖种子/折的结果在汇总表标待补。
- 历史结果只作注明协议的参考，不覆盖旧报告、不直接写入论文最终四张表。
