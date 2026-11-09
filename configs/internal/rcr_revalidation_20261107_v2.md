# RCR规范复验 V2：当前执行计划

机器计划：[rcr_revalidation_20261107_v2.json](rcr_revalidation_20261107_v2.json)。
V1配置、源码快照及已运行产物保留。V2仅恢复同配置、同种子的已运行从零训练任务；不加载历史RCR权重，不增加续训实验。

| 实验 | 参数 | 种子 | 数据单元 | 新训练数 |
| --- | --- | --- | ---: | ---: |
| 完整RCR | 原变量图＋Transformer×2＋Decoder | 3407/3408/3409 | 7 | 21 |
| 去变量图 | feature_graph=false | 3407 | 7 | 7 |
| 去Transformer | temporal=false | 3407 | 7 | 7 |
| 去校准 | 同R1权重，直接重构误差 | 3407 | 7 | 0 |
| 学习率敏感性 | 0.0003 / 0.001 / 0.003 | 3407 | 7 | 14 |
| 隐维度敏感性 | 16 / 32 / 64 | 3407 | 7 | 14 |
| Transformer层数敏感性 | 1 / 2 / 3 | 3407 | 7 | 14 |
| 尺度下限敏感性 | 0.005 / 0.05 / 0.1 | 3407/3408/3409 | 7 | 0 |

7个数据单元为LFP、NCM、Brand3五折。共77个训练单元、126个评价单元（77个主评分＋7个去校准＋42个非默认下限）。
Dropout固定0.1，不做其敏感性；Transformer头数固定4，FFN维度固定64。除目标参数外均保持默认值，三档中间值复用完整RCR的3407结果。
结构消融仅与完整RCR的3407结果配对；不能拿单种子消融直接与完整模型三种子均值作结构归因。
去Transformer检查0层；层数敏感性检查1/2/3层，两者任务不同。

## 固定训练及评分

- 模型从零训练，最多30轮、patience=8、AdamW lr=0.001、batch=128、Huber损失；正常验证最低损失选择检查点。
- 数据划分始终固定：CH split seed=3407，Brand3 fold seed=0，完整正常训练窗口，训练正常数据的MinMax归一化。
- 唯一主评分：原RCR Scale（带符号重构残差减工况均值、除工况尺度、平方平均）。全训练正常窗口拟合校准，正常验证P95阈值。
- 去校准只作消融；尺度下限只作方法参数敏感性。二者复用网络权重，但各自用正常验证确定阈值。
- 不跑cal_abs、额外校准留出、旧结果复算、Decoder续训和整体续训。
- 每项敏感性预先固定全部取值，不根据测试最高AP选择配置。
- Brand3五折先在每种子内平均，再对三种子结果求均值±标准差；单种子消融只报告该种子结果。跨折重复故障车辆不视作独立车辆。
- 新实验只作为当前固定划分上的开发复验，未更改历史外部模型协议，不能仅凭同为30轮就宣称完整协议相同。

## 运行

```bash
.venv/bin/python -m src.runners.run_rcr_revalidation \
  --plan configs/internal/rcr_revalidation_20261107_v2.json --dry-run
```

正式运行必须在沙箱外，单GPU串行、tmux后台：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
MPLCONFIGDIR=/tmp/rcr-revalidation-mpl \
.venv/bin/python -u -m src.runners.run_rcr_revalidation \
  --plan configs/internal/rcr_revalidation_20261107_v2.json \
  --execute --wait-predecessor
```

结果目录：`runs/internal/rcr_revalidation/rcr_revalidation_20261107_v2/`。
报告：`report/analysis/rcr_revalidation/RCR_REVALIDATION_20261107_v2.md`。
运行器保存plan、源码、队列展开清单、数据映射与归一化来源哈希；最佳/最终权重、可恢复检查点、训练曲线和逐窗口/样本分数均保留。
历史V1保存为被本版替代，已完成/中断任务在V2记录复用来源，不重复已完成训练，不覆盖旧报告，不改论文最终四表。
