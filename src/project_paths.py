"""集中定义项目根目录和常用路径。"""


import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _platform_name() -> str:
    """Return the supported platform key used by the path configuration."""
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    return "linux"


CURRENT_PLATFORM = _platform_name()

# Keep platform-specific defaults in one place. Environment variables can
# override machine-specific dataset and run locations without code changes.
PLATFORM_PATHS = {
    "linux": {
        "python": PROJECT_ROOT / ".venv" / "bin" / "python",
        "datasets": PROJECT_ROOT / "datasets",
        "runs": PROJECT_ROOT / "runs",
    },
    "windows": {
        "python": PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        "datasets": PROJECT_ROOT / "datasets",
        "runs": PROJECT_ROOT / "runs",
    },
}


def _platform_path(name: str) -> Path:
    platform_key = CURRENT_PLATFORM.upper()
    env_name = f"MTAD_GAT_{platform_key}_{name.upper()}_ROOT"
    configured = os.getenv(env_name)
    if configured:
        return Path(configured).expanduser()
    return PLATFORM_PATHS[CURRENT_PLATFORM][name]


PYTHON_EXECUTABLE = _platform_path("python")

KAGGLE_INPUT_ROOT = Path("/kaggle/input")

CONFIGS_ROOT = PROJECT_ROOT / "configs"

DATASETS_ROOT = Path(os.getenv("MTAD_GAT_DATASETS_ROOT", _platform_path("datasets")))

KAGGLE_DATASET_SLUGS = {
    "CH-BATTERY": ("ch-battery", "CH-BATTERY"),
    "BMS": ("bms", "BMS"),
    "NASA-RANDOM-DISCHARGE": ("nasa-random-discharge", "NASA_RANDOM_DISCHARGE"),
    "NASA-RANDOM-CHARGE": ("nasa-random-charge", "NASA_RANDOM_CHARGE"),
    "DATA": ("smap-msl", "data"),
    "TSINGHUA-EV": ("tsinghua-ev", "TSINGHUA_EV"),
}


def resolve_dataset_root(dataset_name: str | None = None, default_relative: str | None = None) -> Path:
    dataset_key = str(dataset_name).upper().replace("_", "-") if dataset_name else ""
    if dataset_key:
        dataset_env_name = "MTAD_GAT_" + dataset_key.replace("-", "_") + "_ROOT"
        dataset_env_root = os.getenv(dataset_env_name)
        if dataset_env_root:
            return Path(dataset_env_root)

    env_root = os.getenv("MTAD_GAT_DATASETS_ROOT")
    if env_root:
        base = Path(env_root)
    elif KAGGLE_INPUT_ROOT.exists():
        base = KAGGLE_INPUT_ROOT
    else:
        base = DATASETS_ROOT

    if not dataset_name:
        return base

    local_name = default_relative or str(dataset_name)
    candidates = []
    if base == KAGGLE_INPUT_ROOT:
        slug_names = KAGGLE_DATASET_SLUGS.get(dataset_key, ())
        for slug in slug_names:
            candidates.extend([base / slug, base / slug / local_name])
    candidates.append(base / local_name)
    candidates.append(PROJECT_ROOT / local_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def dataset_path(*parts: str) -> Path:
    return resolve_dataset_root() / Path(*parts)


def processed_dataset_path(dataset_name: str, local_name: str | None = None) -> Path:
    return DATASETS_ROOT / (local_name or dataset_name) / "processed"

RUNS_ROOT = Path(os.getenv("MTAD_GAT_RUNS_ROOT", _platform_path("runs")))

INTERNAL_RUNS_ROOT = RUNS_ROOT / "internal"

EXTERNAL_RUNS_ROOT = RUNS_ROOT / "external"

MANUAL_RUNS_ROOT = RUNS_ROOT / "manual"

REPORT_ROOT = PROJECT_ROOT / "report"
