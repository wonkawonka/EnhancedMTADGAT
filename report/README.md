<!-- 说明：记录项目分析报告、生成报告和模板的统一目录。 -->

# 报告目录说明

`report/` 用来沉淀分析文档，不再把分析结论散落在聊天记录和临时截图里。

## 建议结构

- `report/analysis/`
  - 各数据集的数据探索分析输出，包含 `csv/json/md/png`
- `report/generated/`
  - 项目级汇总报告，如实验计划盘点、消融矩阵、分析报告索引
- `report/templates/`
  - 报告模板，后续写论文或中期材料时直接复用

## 生成方式

数据分析：

```powershell
.\.python312\python.exe -m src.runners.analyze --dataset MSL
.\.python312\python.exe -m src.runners.analyze --dataset NASA_RANDOM_DISCHARGE --nasa_train_batteries RW1,RW2,RW7,RW8 --nasa_test_batteries RW1,RW2,RW7,RW8
```

项目总报告：

```powershell
.\.python312\python.exe -m src.runners.build_report
```

统一入口：

```powershell
.\.python312\python.exe .\run.py full --internal-plan configs/internal/03_ch4_nasa_random_main.json --analysis-dataset NASA_RANDOM_DISCHARGE
```

