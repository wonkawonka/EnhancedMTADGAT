# Kaggle GPU 执行手册

## 1. 首次检查

```bash
pip install -r requirements-kaggle-main.txt
python run.py preflight --tsinghua-ev-root datasets/TSINGHUA_EV
python run.py internal --plan configs/internal/00_kaggle_smoke.json
```

MSL 首次使用前运行：

```bash
python run.py preprocess --dataset MSL
```

`preflight` 检查 CUDA、数据包和模型前向/反向。正式配置使用 `require_cuda=true`，不会静默切到 CPU。冒烟结果只证明链路可用，不能作为论文性能。

## 2. 推荐顺序

```text
环境预检
  -> C3/C4 冒烟
  -> 06 正式计划的 seed 3407 主结果
  -> 单种子消融
  -> 主结果 seed 2024/2025
  -> NASA Random 与 BMS
  -> 外部基线
```

`06_kaggle_formal.json` 不缩数据、不改变正式步长和训练上限，只去掉原计划之间的重复实验。先检查训练/验证损失、输出目录、峰值显存和片段级指标是否合理，再继续后续运行。每次使用 `--resume --skip-existing`，并及时保存 `runs/internal/06_kaggle_formal/`。

## 3. 正式命令示例

正式整合计划包含 18 个实验定义，主结果局部展开三种子后共 28 次运行：

```bash
python run.py internal --plan configs/internal/06_kaggle_formal.json --resume --skip-existing
```

整套计划不可能在一个会话跑完。建议一个清华模型一次会话，先跑三个核心模型的 seed 3407：

```bash
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_mtadgat_seed3407 --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c3_full_seed3407 --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c4_full_seed3407 --resume --skip-existing
```

再逐个运行 C3/C4 消融；每个名称都是独立正式实验：

```bash
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c3_no_condition --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c3_statistics_encoder --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c3_no_aux --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c3_residual_gate --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c3_fixed_score --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c4_state_only --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c4_state_loss --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c4_no_voltage --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c4_no_temperature --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c4_no_charge_flow --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only tsinghua_c4_no_soc_current --resume --skip-existing
```

MSL、NASA Random 和 BMS 同样使用完整数据：

```bash
python run.py internal --plan configs/internal/06_kaggle_formal.json --only msl_mtadgat_seed3407,msl_mtadgat_seed2024,msl_mtadgat_seed2025,msl_c3_full_seed3407,msl_c3_full_seed2024,msl_c3_full_seed2025 --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only nasa_random_condition --resume --skip-existing
python run.py internal --plan configs/internal/06_kaggle_formal.json --only bms_idle_frequency_regulation --resume --skip-existing
```

最后补清华主结果的 seed 2024/2025。命名规则是 `实验名_seed种子`，可按上面的 seed 3407 命令替换。

4 个 MSL 外部基线使用第三方仓和旧依赖，建议另开 Kaggle 会话：

```bash
pip install -r requirements-kaggle-cu118.txt
python run.py external --plan configs/external/01_ch3_msl_external.json --skip-existing
```

不要在已经运行内部模型的主环境中安装 `requirements-kaggle-cu118.txt`；它会把 PyTorch 切换到外部 GDN/DGL/PyG 使用的 CUDA 11.8 兼容栈。

## 4. 规模与时间基线

- 清华：训练 797,472 窗口，正常校准 170,880，测试 204,480。
- 本机 RTX 3060 Laptop 6 GB 的 24,000 窗口冒烟约 25 秒/epoch；线性估计清华完整 epoch 约 13–15 分钟。
- 单模型 20–35 epoch 约 5–9 小时，跑满 50 epoch 约 11–13 小时；Kaggle T4/P100 需用首个完整 epoch 重新估算。
- 正式整合计划展开后为：MSL 6 次、清华 20 次、NASA Random 1 次、BMS 1 次，共 28 次内部运行。
- 清华 20 次占主要耗时：若早停在 20–35 epoch，约 100–180 GPU 小时；若都跑满 50 epoch，约 220–260 GPU 小时。
- 连同 MSL、NASA Random、BMS，内部正式计划先按约 110–210 GPU 小时准备。该估计来自本机吞吐外推，Kaggle GPU、I/O 和实际早停轮数会造成较大偏差。
- 外部 4 基线另估 1–3 小时，但首次环境适配可能比训练本身更久，不计入内部时间。

正式计划需要跨多个 Kaggle 会话保存并续跑。优先保证 MTAD-GAT、C3、C4 的 seed 3407 和关键消融，再补主结果另外两个种子。

首个清华实验跑完一轮后，用日志中的 `[Epoch ... xx.xs]` 重新估算：

```text
剩余训练时间约 = 首轮秒数 × 剩余epoch数 × 剩余同规模模型数
```

## 5. 每组实验要回答什么

| 组别 | 关键比较 | 希望看到的证据 |
| --- | --- | --- |
| 第三章主结果 | MTAD-GAT vs C3 | 清华 AUPRC/F1 提升或正常 FPR 降低；MSL 至少验证方法不是只对单一电池数据有效 |
| C3 状态消融 | full vs no-condition/statistics/no-aux | 学习式状态编码和辅助约束确实提供增益，而不是单纯多参数 |
| C3 位置与评分 | fusion vs residual-gate；quality-aware vs fixed | FiLM 条件化位置与分数融合设计分别有独立贡献 |
| 第四章递进 | C3 -> state -> state+loss -> full | 响应 MAE 和检测指标逐步改善，且正常 FPR 不明显恶化 |
| 物理项留一 | full vs 去掉四类响应 | 判断各响应项是否有效；不强求每项都同幅度下降 |
| NASA Random | RW1/2/7训练，RW8探针 | 冻结状态嵌入能跨电池区分电流方向；不是故障检测结论 |
| BMS | 静置/调频/切换统计和高分案例 | 工况切换不过度误报；可能发现簇电流份额异常，但无标签时不能给检测准确率 |
| 外部基线 | C3 vs 4种公开模型 | 给第三章提供横向参照；必须统一数据、raw指标与阈值口径 |

`06` 本身就是正式口径：完整数据、50 epoch 上限、主结果三种子。若某项不符合预期，应先查收敛和指标方差，不能只因结果不好就删除消融。

## 6. 结果检查

- 确认配置中 `regime_condition_mode=fusion`、`use_revin=false`。
- MSL 的 `regime_aux_lambda=0`；清华/BMS 保留动态描述辅助任务。
- 检查校准集与测试集没有交叉，阈值不使用测试标签。
- 清华保存片段级指标；BMS 不应生成监督 F1/AUROC 结论。
