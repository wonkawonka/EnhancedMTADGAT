import argparse
import os
import pickle
import math
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.GANF import GANF
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common_output import save_standardized_output


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_processed_path(dataset_name):
    ds = str(dataset_name)
    if ds in {"SMAP", "MSL"}:
        return PROJECT_ROOT / "datasets" / "data" / "processed", ds
    if ds.startswith("BMS_"):
        return PROJECT_ROOT / "datasets" / "BMS" / "processed", ds
    if ds.startswith("NASA_RANDOM_DISCHARGE_"):
        return PROJECT_ROOT / "datasets" / "NASA_RANDOM_DISCHARGE" / "processed", ds
    if ds.startswith("NASA_"):
        return PROJECT_ROOT / "datasets" / "NASA" / "processed", ds
    raise ValueError(f"Unknown dataset: {ds}")


def load_project_data(dataset_name):
    proc_dir, stem = resolve_processed_path(dataset_name)
    train_path = proc_dir / f"{stem}_train.pkl"
    test_path = proc_dir / f"{stem}_test.pkl"
    label_path = proc_dir / f"{stem}_test_label.pkl"
    with open(train_path, "rb") as f:
        train = np.asarray(pickle.load(f), dtype=np.float64)
    with open(test_path, "rb") as f:
        test = np.asarray(pickle.load(f), dtype=np.float64)
    with open(label_path, "rb") as f:
        labels = np.asarray(pickle.load(f), dtype=np.float64).squeeze()
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test, labels


def compute_dag_adjacency(train_data):
    K = train_data.shape[1]
    corr = np.corrcoef(train_data.T)
    adj = (np.abs(corr) > 0.3).astype(np.float32)
    adj = adj + np.eye(K)
    adj = (adj > 0).astype(np.float32)
    return torch.FloatTensor(adj)


class SlidingWindowDataset(Dataset):
    def __init__(self, data, labels=None, window_size=60, stride=10):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.n_samples = (len(data) - window_size) // stride + 1
        self.labels = labels

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        start = index * self.stride
        end = start + self.window_size
        window = self.data[start:end]
        x = torch.FloatTensor(window).unsqueeze(-1).transpose(0, 1)
        if self.labels is not None:
            lbl = self.labels[start:end].max()
            return x, lbl
        return x


def adjust_learning_rate(optimizer, epoch, lr):
    lr_adjust = {epoch: lr * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_adjust[epoch]


def train_epoch(model, loader, A, optimizer, device):
    model.train()
    total_loss = 0
    for x, *_ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        loss = -model(x, A.to(device))
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test(model, loader, A, device):
    model.eval()
    scores_list = []
    labels_list = []
    for batch in loader:
        if len(batch) == 2:
            x, lbl = batch
            labels_list.append(lbl.numpy())
        else:
            x = batch
        x = x.to(device)
        log_prob = model.test(x, A.to(device))
        scores_list.append(-log_prob.cpu().numpy())
    scores = np.concatenate(scores_list)
    labels = np.concatenate(labels_list) if labels_list else None
    return scores, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--n_hidden", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="MAF", choices=["MAF", "RealNVP"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    if args.output_dir:
        args.save_path = str(Path(args.output_dir) / "checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data, test_data, test_labels = load_project_data(args.dataset)
    n_sensor = train_data.shape[1]

    A = compute_dag_adjacency(train_data)

    train_dataset = SlidingWindowDataset(train_data, window_size=args.window_size, stride=args.stride)
    test_dataset = SlidingWindowDataset(test_data, test_labels, window_size=args.window_size, stride=args.stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = GANF(
        n_blocks=args.n_blocks,
        input_size=1,
        hidden_size=args.hidden_size,
        n_hidden=args.n_hidden,
        dropout=0.1,
        model=args.model_type,
        batch_norm=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, A, optimizer, device)
        scores, _ = test(model, test_loader, A, device)

        if test_labels is not None and len(test_labels) > 0:
            min_len = min(len(scores), len(test_labels))
            auc = roc_auc_score(test_labels[:min_len], scores[:min_len])
            print(f"Epoch {epoch:2d} | loss: {loss:.4f} | AUROC: {auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                if args.save_path:
                    os.makedirs(args.save_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(args.save_path, "model.pth"))
        else:
            print(f"Epoch {epoch:2d} | loss: {loss:.4f}")

    scores, _ = test(model, test_loader, A, device)
    if test_labels is not None and len(test_labels) > 0:
        min_len = min(len(scores), len(test_labels))
        y_true = test_labels[:min_len]
        y_score = scores[:min_len]
        auc = roc_auc_score(y_true, y_score)
        thresh = np.percentile(y_score, 95)
        y_pred = (y_score > thresh).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        print(f"Threshold : {thresh:.4f}")
        print(f"Accuracy : {accuracy:.4f}, Precision : {precision:.4f}, Recall : {recall:.4f}, F-score : {f_score:.4f}")
        print(f"AUROC: {auc:.4f}")
    else:
        print("No labels available for evaluation. Scores computed.")
        print(f"Scores - min: {scores.min():.4f}, max: {scores.max():.4f}, mean: {scores.mean():.4f}")

    # Save standardized output
    if args.output_dir:
        metrics = {}
        if test_labels is not None and len(test_labels) > 0:
            metrics = {
                "metric_f1": float(f_score),
                "metric_precision": float(precision),
                "metric_recall": float(recall),
                "metric_auroc": float(auc),
                "metric_threshold": float(thresh),
                "metric_accuracy": float(accuracy),
            }
        save_standardized_output(
            output_dir=args.output_dir,
            metrics=metrics,
            thresholds={
                "global_threshold": float(np.percentile(scores, 95)) if len(scores) > 0 else 0,
            },
            test_scores=scores,
            test_labels=test_labels[:min_len] if test_labels is not None and len(test_labels) > 0 else None,
            config=vars(args),
        )


if __name__ == "__main__":
    main()
