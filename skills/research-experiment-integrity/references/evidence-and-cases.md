# 依据与针对性检查

这是审查参考，不是统一参数标准。下列来源支持透明记录、独立选择和方法核验，不保证某个数值适合本项目。查阅日期：2026-11-11。

## 一手依据

- [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist)，第 4–8 项：方法与基线可复现、实验细节及参数选择过程、误差统计和计算资源需披露。没有规定所有方法采用同一学习率、窗口或 30 轮。
- [TSB-AD，NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_Track.pdf)，§5.1：基于原论文/开源实现建议构建每算法搜索空间，在独立 tuning 集选择后评估。不能据此授权本项目使用测试故障标签调参，也不能照搬其点级指标到只有片段标签的数据。
- [Anomaly Detection in Time Series: A Comprehensive Evaluation，PVLDB 2022](https://vldb.org/pvldb/vol15/p1779-wenig.pdf)，§4.1.3：从论文/参考实现默认值出发，区分固定、数据依赖、共享和需优化参数。这是一种明确报告的实验设计，不是所有领域必须照搬的规则。
- [Google Research Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)，Designing the next round of experiments：公平比较需要考虑不同方法对学习率等配套参数的不同需求。该文主要面向监督/类似监督训练，不能推导出无标签异常检测的可靠最优选择指标。
- [TimeEval 配置文档](https://timeeval.readthedocs.io/en/latest/user/usage-timeeval.html)：算法版本、每算法参数及资源限制显式配置。框架能执行协议，不能证明实现符合论文。
- [Anthropic Skill Authoring Best Practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)：指令应简明、按需提供细节并通过真实任务观察效果。技能格式校验不等于行为验证，更不是科学正确性认证。

## 本项目失败案例及所需证据

| 场景 | 应检查的证据与处理 |
| --- | --- |
| 128 点片段，window=10、stride=127 | `range(0, 128-10+1, 127)` 仅产生 start=0。即使片段数量齐全，也只覆盖前 10 点。分别计算输入覆盖和分数目标覆盖；不能因一片段一分数就通过。 |
| plan 写 window=32，模型内部覆盖成 10 | 查看实例化后的有效值和实际输入/目标映射；声明配置不等于运行配置。 |
| 换成作者默认长窗口导致短片段消失 | 比对前后样本 ID、长度和排除原因；补齐、变窗口或改队列都是待披露选择，不静默缩小测试集合。 |
| 数据原有 128 点，旧 RCR 只用 127 点 | 明确有效时间区间；不把原始数据完整等同于旧模型已全覆盖，也不偷偷重跑 RCR 来改变对照。 |
| 拼接实体后直接滑窗 | 用边界附近样本核对同一窗口输入与目标的实体/片段 ID。数组连续存储可以合法，跨实体建模不能悄悄发生。 |
| 预测下一点变为重构输入最后一点 | 用递增序列检查目标时刻及可见输入；这是任务改变，不是 shape 适配。 |
| 自写 GRU-VAE 命名 OmniAnomaly | 对照原文随机递归建模、normalizing flow 和概率评分。不能用“同为 VAE”证明忠实；缺模块则标变体，不作为作者原模型。 |
| 把 ML4ITS MTAD-GAT 当作者实现 | [README](https://github.com/ML4ITS/mtad-gat-pytorch) 明示其不是原论文结果复现，默认 GATv2、GRU 解码替代 VAE。必须披露身份，不能用通用模型名掩盖实现区别。 |
| TranAD 被通用 Transformer/固定双项 MSE 替换 | 核对参考实现的层定义、位置编码、分阶段目标和随 epoch 变化的损失权重；默认窗口 10 不授权每片段只看前 10 点。[作者代码](https://github.com/imperial-qore/TranAD) |
| AT/DC 评分少温度或只取最后点 | 与参考公式核对 temperature、softmax 轴、时间维度及回填，不能因输出都是标量就认为等价。 |
| Min-max 的标量验证值恒零 | 检查 detach/梯度方向和验证量有效性；前向相消不意味着无训练梯度。无效验证值不能静默替换成另一目标，也不能声称 early stopping 已有效。 |
| 原仓库按测试集最优 F1 选阈值 | 区分模型分数与测试标签参与的阈值。项目固定阈值结果不能与 oracle/point-adjusted F1 混列。 |
| 原样协议已获确认，再次查询进度 | 直接核对产物与进度，不要求用户重复批准，也不重新启动训练。 |

这些案例用于人工走查及后续针对性测试。仅把案例写进文档不算已运行测试；审计报告只列实际取得的证据。
