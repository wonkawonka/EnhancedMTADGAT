import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KAGGLE_DIR = ROOT / "kaggle离线output"


def iter_zip_files(root_dir):
    for zip_path in sorted(root_dir.glob("**/*.zip")):
        if "analysis" in zip_path.parts:
            continue
        yield zip_path


def extract_one(zip_path, overwrite=False):
    target_dir = zip_path.with_suffix("")
    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        return {
            "zip": str(zip_path),
            "target": str(target_dir),
            "status": "skipped_existing",
        }

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
        return {
            "zip": str(zip_path),
            "target": str(target_dir),
            "status": "extracted",
            "file_count": len(zf.namelist()),
        }


def main():
    parser = argparse.ArgumentParser(description="Extract all Kaggle offline result zips in place.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(KAGGLE_DIR),
        help="Root directory that contains plan folders with zip files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if the target directory already exists and is non-empty.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    zip_paths = list(iter_zip_files(root_dir))
    if not zip_paths:
        print(f"No zip files found under: {root_dir}")
        return

    extracted = 0
    skipped = 0
    for index, zip_path in enumerate(zip_paths, 1):
        result = extract_one(zip_path, overwrite=args.overwrite)
        print(f"[{index}/{len(zip_paths)}] {result['status']}: {zip_path.name} -> {result['target']}")
        if result["status"] == "extracted":
            extracted += 1
        else:
            skipped += 1

    print(f"Done. Extracted={extracted}, Skipped={skipped}, Total={len(zip_paths)}")


if __name__ == "__main__":
    main()
