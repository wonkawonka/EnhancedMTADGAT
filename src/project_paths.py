"""集中定义项目根目录和常用路径。"""


import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

KAGGLE_INPUT_ROOT = Path("/kaggle/input")

CONFIGS_ROOT = PROJECT_ROOT / "configs"

DATASETS_ROOT = PROJECT_ROOT / "datasets"

KAGGLE_DATASET_SLUGS = {
    "CH-BATTERY": ("ch-battery", "CH-BATTERY"),
    "BMS": ("bms", "BMS"),
    "NASA-RANDOM-DISCHARGE": ("nasa-random-discharge", "NASA_RANDOM_DISCHARGE"),
    "NASA-RANDOM-CHARGE": ("nasa-random-charge", "NASA_RANDOM_CHARGE"),
    "DATA": ("smap-msl", "data"),
    "SMAP-MSL": ("smap-msl", "data"),
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

RUNS_ROOT = PROJECT_ROOT / "runs"

INTERNAL_RUNS_ROOT = RUNS_ROOT / "internal"

EXTERNAL_RUNS_ROOT = RUNS_ROOT / "external"

MANUAL_RUNS_ROOT = RUNS_ROOT / "manual"

REPORT_ROOT = PROJECT_ROOT / "report"

