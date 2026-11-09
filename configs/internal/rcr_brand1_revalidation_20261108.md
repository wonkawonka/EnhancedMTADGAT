# Brand1：原RCR规范复验

机器可执行计划：`configs/internal/rcr_brand1_revalidation_20261108.json`。

继承上一轮RCR复验。Brand1完整五折，固定分折种子0；正常车辆分成互斥训练、验证、测试集合，全部故障车辆只在测试集合，每折重复故障对象不当作独立样本。

| 项目 | 配置 | 训练数 | 评估数 |
| --- | --- | ---: | ---: |
| 完整RCR | 3407/3408/3409 × 五折 | 15 | 15 |
| 去变量图 | 3407 × 五折 | 5 | 5 |
| 去Transformer | 3407 × 五折 | 5 | 5 |
| 去校准 | 复用完整RCR的3407权重 | 0 | 5 |
| 学习率 | 0.0003 / 0.001 / 0.003；默认复用 | 10 | 10 |
| 隐维度 | 16 / 32 / 64；默认复用 | 10 | 10 |
| Transformer层数 | 1 / 2 / 3；默认复用 | 10 | 10 |
| 尺度下限敏感性 | 按要求取消 | 0 | 0 |
| 合计 | | **55** | **60** |

每次从零训练，最多30轮、patience=8，正常验证Huber最低点选权重。主评分仍为原RCR Scale，固定尺度下限0.05，阈值取正常验证车辆分数P95；不新增cal_abs或mean-only。Brand1按车辆评价，与本轮Brand3一致。五折先在每种子内平均，再计算三种子均值和样本标准差。

仅下载原论文Brand1文件：https://ndownloader.figshare.com/files/41520222 ，大小1191991306字节，官方MD5 `8d32af34063c0475587c07f6021861a1`。

输出目录：`runs/internal/rcr_revalidation/rcr_brand1_revalidation_20261108`。

独立报告：`report/analysis/rcr_revalidation/BRAND1_REVALIDATION_20261108.md`。不覆盖LFP/NCM/Brand3报告，不修改论文最终四张表。完成后自动整理全部敏感性、逐折逐种子指标和训练轮数。

启动顺序：下载并校验 → 空间检查及解压 → CPU准备五折输入 → GPU串行执行55个训练单元 → 完整报告。RTX3060 Laptop 6GiB，GPU0，batch128。监控实际D盘空间，剩余不足2GiB时停止启动新单元，保留已完成结果。
