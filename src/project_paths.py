"""集中定义项目根目录和常用路径。"""


from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIGS_ROOT = PROJECT_ROOT / "configs"

DATASETS_ROOT = PROJECT_ROOT / "datasets"

RUNS_ROOT = PROJECT_ROOT / "runs"

INTERNAL_RUNS_ROOT = RUNS_ROOT / "internal"

EXTERNAL_RUNS_ROOT = RUNS_ROOT / "external"

MANUAL_RUNS_ROOT = RUNS_ROOT / "manual"

REPORT_ROOT = PROJECT_ROOT / "report"

