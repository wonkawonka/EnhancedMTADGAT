"""将项目数据转换为外部基线所需的输入格式。"""


import argparse

import json

import os

import pickle

from pathlib import Path


import numpy as np

import pandas as pd


from src.project_paths import processed_dataset_path


DATASET_ROOTS = {

    "MSL": processed_dataset_path("data"),

    "SMD": processed_dataset_path("ServerMachineDataset"),

    "BMS": processed_dataset_path("BMS"),

    "NASA_RANDOM_CHARGE": processed_dataset_path("NASA_RANDOM_CHARGE"),

    "NASA_RANDOM_DISCHARGE": processed_dataset_path("NASA_RANDOM_DISCHARGE"),

    "CALCE": processed_dataset_path("CALCE"),

    "CALCE2": processed_dataset_path("CALCE"),

}


DEFAULT_SERIES = {

    "MSL": "MSL",

}


DEFAULT_TRANAD_PREFIX = {

    "MSL": "C-1",

    "SMD": "machine-1-1",

}


def parse_args():

    parser = argparse.ArgumentParser(description="Export project processed data to external baseline formats.")

    subparsers = parser.add_subparsers(dest="command")


    list_parser = subparsers.add_parser("list", help="List available processed series for a dataset.")

    list_parser.add_argument("--source-dataset", required=True, choices=sorted(DATASET_ROOTS.keys()))


    export_parser = subparsers.add_parser("export", help="Export one processed series to a baseline-specific format.")

    export_parser.add_argument("--source-dataset", required=True, choices=sorted(DATASET_ROOTS.keys()))

    export_parser.add_argument("--series-name", default="", help="Series stem without _train.pkl. Optional for MSL.")

    export_parser.add_argument(

        "--target",

        required=True,

        choices=["anomaly_transformer", "tranad", "omnianomaly", "gdn", "lstm_ae"],

    )

    export_parser.add_argument("--output-dir", required=True, help="Target directory for exported files.")

    export_parser.add_argument(

        "--target-dataset-name",

        default="",

        help="Target dataset name used by the external baseline. Defaults to source dataset name.",

    )

    export_parser.add_argument(

        "--target-series-prefix",

        default="",

        help="Series prefix used by TranAD exports, such as P-1 or C-1.",

    )

    export_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")


    args = parser.parse_args()

    if not getattr(args, "command", None):

        parser.error("A sub-command is required. Use 'list' or 'export'.")

    return args


def ensure_dir(path: Path):

    path.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):

    with open(path, "rb") as f:

        return pickle.load(f)


def _concat_sequence_segments(data, value_name: str):

    if isinstance(data, list):

        raise ValueError(

            f"{value_name} is a segmented series. External baseline export no longer supports segmented datasets such as NASA_RANDOM."

        )

    return np.asarray(data)


def list_series_names(source_dataset: str):

    root = DATASET_ROOTS[source_dataset]

    if not root.exists():

        raise FileNotFoundError(f"Processed directory not found: {root}")


    series_names = []

    for file_path in root.glob("*_train.pkl"):

        series_names.append(file_path.name.replace("_train.pkl", ""))

    return sorted(series_names)


def resolve_series_name(source_dataset: str, series_name: str):

    if series_name:

        return series_name

    default_name = DEFAULT_SERIES.get(source_dataset)

    if default_name:

        return default_name

    raise ValueError(f"Dataset {source_dataset} requires --series-name")


def get_feature_names(source_dataset: str, train_array: np.ndarray):

    if source_dataset == "BMS" and train_array.shape[1] == 35:

        return [

            "BMSnVol_T", "BMSnVol_B", "BMSnI", "BMSnRSOC", "BMSnSOH",

            "BMSnICMax", "BMSnIDMax", "BMSnVmax", "BMSnVmin", "BMSnVmean",

            "BMSnTmax", "BMSnTmin", "BMSnTmean", "BMSnETmax", "BMSnETmean",

            "cell_v_std", "cell_v_range", "cell_v_max_dev_from_mean", "cell_v_min_dev_from_mean",

            "cell_t_std", "cell_t_range", "SYS_Vol", "SYS_I", "SYS_SOH", "SYS_Vmax",

            "SYS_Vmin", "SYS_Tmax", "SYS_Tmin", "hier_vmax_sys_gap", "hier_vmin_sys_gap",

            "hier_tmax_sys_gap", "hier_tmin_sys_gap", "hier_soh_sys_gap",

            "hier_cell_v_range_ratio", "hier_cell_t_range_ratio",

        ]

    return [f"f{i}" for i in range(train_array.shape[1])]


def load_processed_series(source_dataset: str, series_name: str):

    root = DATASET_ROOTS[source_dataset]

    train_path = root / f"{series_name}_train.pkl"

    test_path = root / f"{series_name}_test.pkl"

    label_path = root / f"{series_name}_test_label.pkl"


    if not train_path.exists():

        raise FileNotFoundError(f"Train file not found: {train_path}")

    if not test_path.exists():

        raise FileNotFoundError(f"Test file not found: {test_path}")

    if not label_path.exists():

        raise FileNotFoundError(f"Label file not found: {label_path}")


    train = _concat_sequence_segments(load_pickle(train_path), "train").astype(np.float32, copy=False)

    test = _concat_sequence_segments(load_pickle(test_path), "test").astype(np.float32, copy=False)

    labels = _concat_sequence_segments(load_pickle(label_path), "labels")


    if train.ndim != 2 or test.ndim != 2:

        raise ValueError("Processed train/test data must be 2-D arrays.")

    if labels.ndim != 1:

        labels = labels.reshape(-1)


    feature_names = get_feature_names(source_dataset, train)

    return train, test, labels.astype(np.int64), feature_names


def write_numpy(path: Path, array: np.ndarray, overwrite: bool):

    if path.exists() and not overwrite:

        raise FileExistsError(f"File exists: {path}")

    np.save(path, array)


def write_pickle(path: Path, obj, overwrite: bool):

    if path.exists() and not overwrite:

        raise FileExistsError(f"File exists: {path}")

    with open(path, "wb") as f:

        pickle.dump(obj, f)


def export_anomaly_transformer(output_dir: Path, target_dataset_name: str, train, test, labels, overwrite: bool):

    ensure_dir(output_dir)

    write_numpy(output_dir / f"{target_dataset_name}_train.npy", train, overwrite)

    write_numpy(output_dir / f"{target_dataset_name}_test.npy", test, overwrite)

    write_numpy(output_dir / f"{target_dataset_name}_test_label.npy", labels, overwrite)


def export_tranad(output_dir: Path, target_dataset_name: str, series_prefix: str, train, test, labels, overwrite: bool):

    dataset_dir = output_dir / target_dataset_name

    ensure_dir(dataset_dir)

    if str(target_dataset_name).startswith(("BMS_", "NASA_", "NASA_RANDOM_CHARGE_", "NASA_RANDOM_DISCHARGE_")):

        labels = np.asarray(labels)

        if labels.ndim == 1:

            labels = np.repeat(labels.reshape(-1, 1), train.shape[1], axis=1)

        elif labels.ndim == 2 and labels.shape[1] == 1:

            labels = np.repeat(labels, train.shape[1], axis=1)

    write_numpy(dataset_dir / f"{series_prefix}_train.npy", train, overwrite)

    write_numpy(dataset_dir / f"{series_prefix}_test.npy", test, overwrite)

    write_numpy(dataset_dir / f"{series_prefix}_labels.npy", labels, overwrite)


def export_omnianomaly(output_dir: Path, target_dataset_name: str, train, test, labels, overwrite: bool):

    ensure_dir(output_dir)

    write_pickle(output_dir / f"{target_dataset_name}_train.pkl", train, overwrite)

    write_pickle(output_dir / f"{target_dataset_name}_test.pkl", test, overwrite)

    write_pickle(output_dir / f"{target_dataset_name}_test_label.pkl", labels, overwrite)


def export_gdn(output_dir: Path, train, test, labels, feature_names, overwrite: bool):

    ensure_dir(output_dir)

    train_path = output_dir / "train.csv"

    test_path = output_dir / "test.csv"

    list_path = output_dir / "list.txt"


    for path in [train_path, test_path, list_path]:

        if path.exists() and not overwrite:

            raise FileExistsError(f"File exists: {path}")


    train_df = pd.DataFrame(train, columns=feature_names)

    test_df = pd.DataFrame(test, columns=feature_names)

    test_df["attack"] = labels.astype(int)


    train_df.to_csv(train_path)

    test_df.to_csv(test_path)

    with open(list_path, "w", encoding="utf-8") as f:

        for name in feature_names:

            f.write(f"{name}\n")


def export_generic_arrays(output_dir: Path, train, test, labels, source_dataset: str, series_name: str, overwrite: bool):

    ensure_dir(output_dir)

    write_numpy(output_dir / "train.npy", train, overwrite)

    write_numpy(output_dir / "test.npy", test, overwrite)

    write_numpy(output_dir / "test_label.npy", labels, overwrite)


    meta_path = output_dir / "meta.json"

    if meta_path.exists() and not overwrite:

        raise FileExistsError(f"File exists: {meta_path}")

    with open(meta_path, "w", encoding="utf-8") as f:

        json.dump(

            {

                "source_dataset": source_dataset,

                "series_name": series_name,

                "train_shape": list(train.shape),

                "test_shape": list(test.shape),

                "label_shape": list(labels.shape),

            },

            f,

            indent=2,

            ensure_ascii=False,

        )


def main():

    args = parse_args()


    if args.command == "list":

        for series_name in list_series_names(args.source_dataset):

            print(series_name)

        return


    series_name = resolve_series_name(args.source_dataset, args.series_name)

    train, test, labels, feature_names = load_processed_series(args.source_dataset, series_name)

    output_dir = Path(args.output_dir).resolve()

    target_dataset_name = args.target_dataset_name or args.source_dataset


    if args.target == "anomaly_transformer":

        export_anomaly_transformer(output_dir, target_dataset_name, train, test, labels, args.overwrite)

    elif args.target == "tranad":

        series_prefix = args.target_series_prefix or DEFAULT_TRANAD_PREFIX.get(target_dataset_name, series_name)

        export_tranad(output_dir, target_dataset_name, series_prefix, train, test, labels, args.overwrite)

    elif args.target == "omnianomaly":

        export_omnianomaly(output_dir, target_dataset_name, train, test, labels, args.overwrite)

    elif args.target == "gdn":

        export_gdn(output_dir, train, test, labels, feature_names, args.overwrite)

    elif args.target == "lstm_ae":

        export_generic_arrays(output_dir, train, test, labels, args.source_dataset, series_name, args.overwrite)

    else:

        raise ValueError(f"Unsupported target: {args.target}")


    print(f"Exported {args.source_dataset}:{series_name} -> {args.target} at {output_dir}")

    print(f"train={train.shape}, test={test.shape}, labels={labels.shape}")


if __name__ == "__main__":

    main()

