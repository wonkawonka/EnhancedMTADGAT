import json
import os
import numpy as np
import pandas as pd
from pathlib import Path


def save_standardized_output(output_dir, metrics=None, thresholds=None,
                             train_scores=None, test_scores=None,
                             train_labels=None, test_labels=None,
                             train_preds=None, test_preds=None,
                             config=None,
                             train_losses=None, val_losses=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # summary_metrics.json
    if metrics:
        metrics_path = output_dir / "summary_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                existing = json.load(f)
            existing.update(metrics)
            metrics = existing
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    # thresholds.json
    if thresholds:
        with open(output_dir / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2, ensure_ascii=False)

    # test_output.pkl — DataFrame matching internal model format
    if test_scores is not None:
        df = pd.DataFrame({"A_Score_Global": test_scores})
        if test_labels is not None:
            df["A_True_Global"] = test_labels
        if test_preds is not None:
            df["A_Pred_Global"] = test_preds
        if thresholds and "global_threshold" in thresholds:
            df["Thresh_Global"] = thresholds["global_threshold"]
        df.to_pickle(output_dir / "test_output.pkl")

    # train_output.pkl
    if train_scores is not None:
        df = pd.DataFrame({"A_Score_Global": train_scores})
        if train_labels is not None:
            df["A_True_Global"] = train_labels
        if train_preds is not None:
            df["A_Pred_Global"] = train_preds
        df.to_pickle(output_dir / "train_output.pkl")

    # train_losses.png / validation_losses.png
    if train_losses is not None and len(train_losses) > 0:
        _save_loss_plot(train_losses, val_losses, output_dir)

    # summary.txt
    with open(output_dir / "summary.txt", "w") as f:
        f.write("===== External Baseline Results =====\n\n")
        if config:
            f.write("Config:\n")
            for k, v in sorted(config.items()):
                f.write(f"  {k}: {v}\n")
            f.write("\n")
        if metrics:
            f.write("Metrics:\n")
            for k, v in sorted(metrics.items()):
                f.write(f"  {k}: {v}\n")
        if thresholds:
            f.write("\nThresholds:\n")
            for k, v in sorted(thresholds.items()):
                f.write(f"  {k}: {v}\n")

    # config.txt
    if config:
        with open(output_dir / "config.txt", "w") as f:
            for k, v in sorted(config.items()):
                f.write(f"{k}={v}\n")


def _save_loss_plot(train_losses, val_losses, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label='Train Loss')
        if val_losses is not None and len(val_losses) > 0:
            ax.plot(val_losses, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        fig.savefig(output_dir / "train_losses.png", dpi=100, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[OUTPUT] Warning: could not save loss plot: {e}")
