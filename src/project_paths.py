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

KAGGLE_INPUT_ROOT = Path(os.getenv("KAGGLE_INPUT_ROOT", "/kaggle/input"))

CONFIGS_ROOT = PROJECT_ROOT / "configs"

DATASETS_ROOT = Path(os.getenv("MTAD_GAT_DATASETS_ROOT", _platform_path("datasets")))

KAGGLE_DATASET_SLUGS = {
    "CH-BATTERY": ("ch-battery", "CH-BATTERY"),
    "BMS": ("bms", "BMS", "bms-data"),
    "NASA-RANDOM-DISCHARGE": ("nasa-random-discharge", "NASA_RANDOM_DISCHARGE"),
    "NASA-RANDOM-CHARGE": ("nasa-random-charge", "NASA_RANDOM_CHARGE"),
    "DATA": ("smap-msl", "data"),
    "TSINGHUA-EV": (
        "tsinghua-ev",
        "TSINGHUA_EV",
        "tsinghua-ev-battery",
        "realistic-battery-fault-detection",
    ),
}


def _dataset_root_matches(candidate: Path, dataset_key: str) -> bool:
    """Return whether a directory contains the requested dataset, not merely exists."""
    if not candidate.is_dir():
        return False
    if dataset_key == "TSINGHUA-EV":
        return all((candidate / f"battery_brand{brand}").is_dir() for brand in (1, 2, 3))
    if dataset_key == "BMS":
        processed = candidate if candidate.name == "processed" else candidate / "processed"
        if processed.is_dir() and any(processed.glob("BMS_*_train.pkl")):
            return True
        # Raw BMS uploads consist of four workbooks per acquisition bundle.
        # Recognising the suffixes here lets preprocessing consume a read-only
        # Kaggle Input without requiring a fixed dataset slug.
        raw_suffixes = (
            "_BMS0Data.xls",
            "_BMS0Data.xlsx",
            "_BMSnStatData.xls",
            "_BMSnStatData.xlsx",
            "_BMSnDetailTempData.xls",
            "_BMSnDetailTempData.xlsx",
            "_BMSnDetailVoltData.xls",
            "_BMSnDetailVoltData.xlsx",
        )
        found_types = {
            suffix
            for path in candidate.iterdir()
            if path.is_file()
            for suffix in raw_suffixes
            if path.name.endswith(suffix)
        }
        return any(suffix.startswith("_BMS0Data") for suffix in found_types) and all(
            any(suffix.startswith(prefix) for suffix in found_types)
            for prefix in ("_BMSnStatData", "_BMSnDetailTempData", "_BMSnDetailVoltData")
        )
    return True


def _candidate_dataset_roots(base: Path, dataset_key: str, local_name: str):
    """Yield common local/Kaggle layouts while preserving deterministic priority."""
    yield base / local_name
    aliases = {local_name.lower(), *(slug.lower() for slug in KAGGLE_DATASET_SLUGS.get(dataset_key, ()))}
    if dataset_key in {"TSINGHUA-EV", "BMS"} or base.name.lower() in aliases:
        yield base
    for slug in KAGGLE_DATASET_SLUGS.get(dataset_key, ()):
        slug_root = base / slug
        yield slug_root
        yield slug_root / local_name
        yield slug_root / "datasets" / local_name
    if base.is_dir():
        # Kaggle dataset slugs are user-defined. Search one level below
        # /kaggle/input and support uploads made from either the dataset folder
        # itself or the repository's datasets/ directory.
        for mounted_dataset in sorted(path for path in base.iterdir() if path.is_dir()):
            if dataset_key in {"TSINGHUA-EV", "BMS"} or mounted_dataset.name.lower() in aliases:
                yield mounted_dataset
            yield mounted_dataset / local_name
            yield mounted_dataset / "datasets" / local_name


def resolve_dataset_root(dataset_name: str | None = None, default_relative: str | None = None) -> Path:
    dataset_key = str(dataset_name).upper().replace("_", "-") if dataset_name else ""
    if dataset_key:
        dataset_env_name = "MTAD_GAT_" + dataset_key.replace("-", "_") + "_ROOT"
        dataset_env_root = os.getenv(dataset_env_name)
        if dataset_env_root:
            return Path(dataset_env_root)

    if not dataset_name:
        return Path(os.getenv("MTAD_GAT_DATASETS_ROOT", DATASETS_ROOT))

    local_name = default_relative or str(dataset_name)
    bases = []
    env_root = os.getenv("MTAD_GAT_DATASETS_ROOT")
    if env_root:
        bases.append(Path(env_root))
    bases.extend([DATASETS_ROOT, KAGGLE_INPUT_ROOT])

    seen = set()
    for base in bases:
        for candidate in _candidate_dataset_roots(base, dataset_key, local_name):
            normalized = str(candidate.resolve()) if candidate.exists() else str(candidate.absolute())
            if normalized in seen:
                continue
            seen.add(normalized)
            if _dataset_root_matches(candidate, dataset_key):
                return candidate
    return DATASETS_ROOT / local_name


def dataset_path(*parts: str) -> Path:
    return resolve_dataset_root() / Path(*parts)


def processed_dataset_path(
    dataset_name: str,
    local_name: str | None = None,
    *,
    for_write: bool = False,
) -> Path:
    """Resolve processed data for reading; preprocessing writes stay project-local."""
    name = local_name or dataset_name
    project_processed = DATASETS_ROOT / name / "processed"
    if for_write:
        return project_processed
    # Raw Kaggle inputs are read-only and may be selected through a
    # dataset-specific MTAD_GAT_*_ROOT.  Prefer a preprocessing result already
    # produced in the writable datasets root before considering mounted input.
    if project_processed.is_dir():
        return project_processed
    root = resolve_dataset_root(dataset_name, name)
    return root if root.name == "processed" else root / "processed"

RUNS_ROOT = Path(os.getenv("MTAD_GAT_RUNS_ROOT", _platform_path("runs")))

INTERNAL_RUNS_ROOT = RUNS_ROOT / "internal"

EXTERNAL_RUNS_ROOT = RUNS_ROOT / "external"

MANUAL_RUNS_ROOT = RUNS_ROOT / "manual"

REPORT_ROOT = PROJECT_ROOT / "report"
