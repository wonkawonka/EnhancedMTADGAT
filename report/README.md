# 报告阅读入口

论文与实验按下面顺序梳理即可：

1. [thesis_experiment_design.md](thesis_experiment_design.md)：先看研究问题、两章递进关系和完整实验流程。
2. [dataset_inventory.md](dataset_inventory.md)：再确认每个数据集的条件变量、响应变量、划分和评价口径。
3. [model_design_and_literature_audit.md](model_design_and_literature_audit.md)：需要写方法与相关工作时，查公式、创新边界和核心文献。
4. [kaggle_runbook.md](kaggle_runbook.md)：实际运行实验时使用。
5. [figures/](figures/)：第三章和第四章模型结构图。

## 一句话主线

- 第三章：用相对可信的运行条件形成连续上下文，并在关系融合表示上进行 FiLM 条件化，使正常响应随运行状态变化。
- 第四章：在第三章上增加电压、温度、电荷流和 SOC—电流响应一致性。

## 统一执行入口

```bash
.venv/bin/python run.py preflight --tsinghua-ev-root datasets/TSINGHUA_EV
.venv/bin/python run.py internal --plan configs/internal/00_kaggle_smoke.json
.venv/bin/python run.py internal --plan configs/internal/01_ch3_main.json --resume --skip-existing
```

生成项目汇总：

```bash
.venv/bin/python -m src.runners.build_report
```

`generated/` 是自动汇总，`analysis/` 是数据分析产物，`templates/` 仅放复用模板；论文口径以上述三个核心 Markdown 为准。
