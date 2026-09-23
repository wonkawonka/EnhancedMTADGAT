# RCR V1 实施与验证协议

状态：实现与八个冒烟单元审计通过；LFP全量backbone已早停（21轮、正常验证选择第13轮），全量Lite拟合/评估完成，Net关系头训练中，其余实验未完成。源方案为用户附件
`043e87fa-1ce1-4345-b169-15b381c5f543/pasted-text-1.txt`。
本文件在任何本轮测试检测指标产生前建立。不修改既有正式模型逻辑或最终四张实验矩阵。

## 必须完成的范围

1. 共用轻量 reconstruction backbone：数值投影及变量embedding、独立全局特征图1层、时间Transformer2层、轻量均值decoder；不做forecasting、概率重构、多尺度或联合微调。
2. 只用CURRENT/SOC提取附件列出的8维窗口条件；训练正常数据拟合线性均值/对数scale，冻结后用于所有后续阶段。
3. RCR-Lite：全局LedoitWolf、magnitude/joint训练正常经验CDF、固定0.5融合。
4. 先验证Lite，再实现RCR-Net：独立残差图、1层2头attention、8→16→16条件MLP、受限FiLM、30%随机mask、仅mask坐标MSE；冻结backbone与calibration。推理逐变量leave-one-out。
5. 核心消融：raw、condition scaling、LedoitWolf、RCRNet、without condition、without feature graph。附件其他诊断：backbone无图/无时间、CDF幅值单支、无FiLM神经关系、无条件/充放电one-hot/连续条件线性调制/条件MLP；alpha=1/.75/.5/.25/0全部报告而不选最优。
6. LFP与NCM充放电混合setting，seed3407，复用原VIN隔离与train-normal输入归一化。与MTAD-GAT、TranAD、Anomaly Transformer、DCdetector、PatchAD、OmniAnomaly对照；旧预算结果明确标记，不冒充等预算架构比较。
7. 总体AUROC/AP/P/R/F1/FPR，正常验证P95/P99，并保存故障类型×severity、动态低/中/高、充放电和low-dynamic HR结果。分层不得重新调阈值。
8. 保存checkpoint、正常calibration/CDF、阈值、全部样本映射、逐窗口各分支分数、逐变量残差/z/LOO误差；复算审计和中文报告。

## 在实现前明确的空白细节

- 窗口32、stride4，沿用当前混合协议。默认不裁剪测试窗口。
- 变量token `[B,L,D,32]` 经图后按变量展平，再线性投影到32维时间token；保留变量身份，不用对称均值导致变量互换。时间位置用固定正弦编码。图为`softmax(UU^T/sqrt(8))`及残差消息传递。
- 条件取train-normal MinMax后的CURRENT/SOC，8维统计再用train-normal均值/标准差标准化；响应变量不进入条件路径。CURRENT/SOC也不被宣称为严格外生因果变量。
- 线性校准：OLS拟合每窗口残差时间均值；对扣除拟合均值后的窗口RMS取log，再OLS拟合log_sigma。每变量clip到训练预测sigma的P5–P95，绝对下限0.05；数值指数保护不替代正常数据clip。
- Lite严格按附件`z^T Omega z / D`评分，保存LedoitWolf的location供审计，不偷偷切换为再次减location的Mahalanobis。
- 点级平方误差/二次型先按窗口时间均值，再建立训练窗口经验CDF；片段取最高5%窗口均值。CDF右连续，允许饱和，报告饱和率而不看测试标签更换融合。alpha固定0.5，其他alpha只作预设敏感性。
- FiLM为`(1+0.1*tanh(gamma))*h+0.1*tanh(beta)`，输出层零初始化，以共享关系为初始状态。所有残差在进入图/attention前mask，且只做同时间变量关系，不引入lag。
- Bernoulli mask概率0.3；若某点没有mask，随机补遮一个坐标，防止空loss；推理始终逐坐标遮一个。relation验证采用固定LOO任务，不依赖故障标签。
- Backbone最多50epoch、relation最多30epoch，正常验证loss早停patience8，AdamW lr=.001、weight_decay=.0001、batch128、Huber delta1。选择最佳正常验证epoch，不能改为测试最优epoch；不以此前10epoch小预算替代附件预算。
- `without condition`同时移除条件scale和FiLM，改用全局正常残差均值/scale；另设仅移除FiLM的干净对照，避免混淆两种干预。one-hot按已有窗口所属充放电元数据，不从响应变量推断。

## 进度与证据

- `src/models/rcr.py`：backbone、线性calibration、Lite、masked residual graph实现。
- `scripts/test_rcr.py`：8项CPU测试已通过：输入/梯度条件隔离、backbone形状/反传、评分公式与序列化复现、CDF/常数工况边界、mask前置防泄漏、无条件不变性、分块统计与直接sklearn LedoitWolf一致性、先融合后Top5%聚合及混淆计数。
- `src/runners/run_rcr.py`：正常验证Huber早停、完整断点/RNG恢复、样本映射、正常VIN隔离核验、带checkpoint哈希的磁盘残差缓存。
- `scripts/audit_rcr_lite_smoke.py`：真实数据缓存（256训练窗口、各128验证/测试窗口）上的Lite公式复算和保存加载精确一致性已通过。输出`runs/internal/rcr_v1_smoke/lite_audit.json`；没有计算检测指标或据此调参。
- 正式LFP输出：`runs/internal/rcr_v1/LFP/full`，全量窗口、最多50轮、patience8；21轮后早停，最佳正常验证Huber为2.9930738301809957e-08（epoch13）。全部残差已导出，不能把正常拟合loss当作检测效果。
- 全量LFP实际窗口数：训练1543911、验证226466、测试1130562；正常训练5600段、验证800段，共同测试3593段。无训练窗口抽样。
- `src/models/rcr_fitting.py`与`src/runners/run_rcr_lite.py`：分块全量OLS与LedoitWolf充分统计量、CDF、逐窗口分支评分、逐变量z及标准化条件磁盘缓存，支持continuous/none/cycle。真实数据冒烟完整执行通过；LFP full/continuous正式拟合及独立审计已完成，其余正式单元按队列推进。
- `src/runners/run_rcr_relation.py`：仅缓存驱动的30轮关系训练、正常验证LOO早停、断点/RNG恢复、全量LOO平方误差、训练CDF与融合；对backbone和校准文件做前后哈希检查。真实数据缓存的CPU两轮完整冒烟已通过，未使用故障标签。
- `scripts/evaluate_rcr.py`：先窗口内固定alpha融合，再片段Top5%聚合；正常验证P95/P99、全部预设分支及alpha均报告。故障分层的AUROC/AP使用同充放电全部正常作对照，动态分档在VIN/工况/故障类型内按原始平均绝对相邻电流差排序三分，不能混称物理dI/dt。
- `scripts/run_rcr_queue.py`：单GPU锁、存活进程核对、串行56个阶段（含8个backbone、12个神经关系头及校准/评估），错误即停止。等待已启动LFP进程结束后自动推进，状态与具体PID见`runs/internal/rcr_v1/queue_status.json`；完成标记不能代替最终审计。
- `scripts/run_rcr_reference.py`：独立复用现有六基线模型/训练实现，支持本轮LFP/NCM精确划分核对，不修改正式loader。预算沿用旧参考10epoch/100k窗口，正常验证重新定P95/P99；与RCR全量预算不等，只能作为明确标注的参考比较。尚未启动补跑。
- 关系头128窗口×32点的CUDA单批前向、反向与LOO推理已通过，峰值分配约197MiB。LFP的3593个片段原始电流动态元数据已准备。
- `scripts/audit_rcr.py`：正常验证最佳epoch、正常车辆隔离、冻结源哈希、全部CDF及融合公式、固定窗口z/LOO误差、阈值和指标复算；在完整冒烟产物上通过。不同BLAS批形仅允许float64末位数值容差，不修改评分公式。
- 旧LFP参考再校准：`scripts/export_rcr_legacy_references.py`读取原始分数并验证种子/划分/特征/标签与实体映射。TranAD原生窗口10（其他为32），额外测试片段`LFP_high_resistance_vin_52_vin_52_charge_7`长度不足32，原始文件核验后仅对共同3593段重算比较，明确保存排除项。不得直接拼接原指标。
- 六个LFP参考已完成重新校准，输出`runs/internal/rcr_v1/references/LFP/`；旧指标文件与checkpoint未改动。NCM六参考在RCR主队列成功完成后串行补跑。
- `scripts/finish_rcr_queue.py`已启动，等待同一GPU锁，随后补跑NCM参考、执行8个backbone目录的完整产物审计、生成报告；队列失败则停止，不跳过缺失实验。
- `scripts/report_rcr.py`生成[中文对比报告](RCR_REPORT_ZH.md)与comparison.csv；当前明确标为未完成，只填已核验结果。最终84行（含两个阈值）的比较记录和全部分层产物必须齐全。
- 报告同步汇总Raw/Scale/Lite/Net的放电低动态HR、全HR/LC/SD结果、完整severity分层CSV、训练/验证/测试CDF饱和比例与实际/最佳epoch；不只报告最高F1。
- continuous、none、cycle三类Lite及相应神经关系头已完成真实数据缓存的小样本CPU流程；这些仅验证实现，不作为模型效果结果。
- 所有五类关系头（continuous+MLP/linear/none、cycle+MLP、none+none）及三类Lite共八个冒烟单元均已通过产物审计。
- `src/models/rcr_detector.py`提供冻结模型独立加载和窗口推理接口，已验证Lite/Net共享raw分数和LOO误差公式。`scripts/archive_rcr_input_scaler.py`只读取训练正常记录，以相同缺失值处理重建并保存MinMax参数；最终审计增加原始窗口→归一化→残差抽查，不更改训练数据或模型。
- LFP输入MinMax已从5600条训练正常记录归档。真实128窗口、同CUDA/同批大小复算与已保存残差逐位一致；CPU/CUDA内核产生约3.1e-5的float32差异，因此正式原始窗口审计要求沙箱外CUDA，CPU接口用于便携推理但不声称与GPU分数逐位一致。输入推理统一连续数组布局。
- 正式LFP训练/验证/测试各128窗口的原始输入→残差及条件特征重建均逐位一致。Lite全量拟合使用49405152个残差点，训练拟合序列化复算通过；七变量scale均近似常数0.05。因此Raw→Scale增益不能解释为条件方差建模收益，保留预设下限不根据测试结果修改。
- 正式LFP `lite_continuous` 单元已通过独立产物审计（`artifact_audit_lite_continuous.json`），包括原始窗口CUDA复算、全量有限性/CDF、正常阈值和指标重算。CSV分数使用`float_precision='round_trip'`读回，避免末位舍入拆分CDF并列分数；没有修改原评分或放宽指标误差。该单元通过不等于整个目标完成。
- 待完成：LFP正式关系头、其余Lite/关系头及backbone消融、NCM训练与六模型对照、完整分层报告及最终完整性核验。LFP full/continuous Lite和LFP六模型旧预算参考的重新校准已完成，不列入待跑项。

完成标准是按方案执行并报告，不以性能必须领先作为结束条件；阴性结果不得触发测试集驱动的结构修改。
