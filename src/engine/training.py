"""Training loop for the frozen MTAD-GAT baseline, C3 and C4 routes."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data.utils import (
    DataLoader,
    SlidingWindowDataset,
    create_data_loaders,
    resolve_dataloader_options,
)


def _build_grad_scaler(use_cuda):
    enabled = use_cuda and torch.cuda.is_available()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(device):
    enabled = device == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _unwrap_model(model):
    return getattr(model, "_orig_mod", model)


class Trainer:
    """Train the baseline loss plus the one frozen C3/C4 auxiliary loss."""

    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        forecast_criterion=nn.MSELoss(),
        recon_criterion=nn.MSELoss(),
        use_cuda=True,
        dload="",
        log_dir="runs/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        window_stride=1,
        regime_aux_lambda=0.0,
        early_stopping_patience=0,
        early_stopping_min_delta=1e-4,
    ):
        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard
        self.loader_num_workers = max(int(num_workers), 0)
        self.loader_persistent_workers = bool(persistent_workers)
        self.loader_prefetch_factor = max(int(prefetch_factor), 1)
        self.window_stride = max(int(window_stride), 1)
        self.regime_aux_lambda = max(0.0, float(regime_aux_lambda))
        self.early_stopping_patience = max(0, int(early_stopping_patience))
        self.early_stopping_min_delta = max(0.0, float(early_stopping_min_delta))
        self.early_stopping_bad_epochs = 0
        self.scaler = _build_grad_scaler(use_cuda)
        self.losses = self._empty_losses()
        self.epoch_times = []
        self.start_epoch = 0
        self.best_val_loss = None
        self.checkpoint_name = "last_checkpoint.pt"
        if self.device == "cuda":
            self.model.cuda()
        if self.log_tensorboard:
            self.writer = SummaryWriter(log_dir)
            self.writer.add_text("args_summary", args_summary)

    @staticmethod
    def _empty_losses():
        return {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "train_regime_aux": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
            "val_regime_aux": [],
        }

    def _batch_losses(self, x, y):
        preds, recons = self.model(x)
        if self.target_dims is not None:
            x_target = x[:, :, self.target_dims]
            y_target = y[:, :, self.target_dims].squeeze(-1)
        else:
            x_target = x
            y_target = y
        if preds.ndim == 3:
            preds = preds.squeeze(1)
        if y_target.ndim == 3:
            y_target = y_target.squeeze(1)
        forecast = torch.sqrt(self.forecast_criterion(y_target, preds))
        reconstruction = torch.sqrt(self.recon_criterion(x_target, recons))
        base_model = _unwrap_model(self.model)
        if self.regime_aux_lambda > 0.0:
            auxiliary = base_model.regime_auxiliary_loss(x)
        else:
            auxiliary = x.new_tensor(0.0)
        total = forecast + reconstruction + self.regime_aux_lambda * auxiliary
        return forecast, reconstruction, auxiliary, total

    @staticmethod
    def _epoch_value(values):
        values = np.asarray(values, dtype=float)
        return float(np.sqrt(np.mean(values ** 2))) if len(values) else 0.0

    def _run_loader(self, data_loader, training):
        self.model.train(training)
        collected = [[], [], []]
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                if training:
                    self.optimizer.zero_grad()
                with _autocast_context(self.device):
                    forecast, reconstruction, auxiliary, total = self._batch_losses(x, y)
                if training:
                    self.scaler.scale(total).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                for bucket, value in zip(collected, (forecast, reconstruction, auxiliary)):
                    bucket.append(value.item())
        forecast, reconstruction, auxiliary = map(self._epoch_value, collected)
        total = forecast + reconstruction + self.regime_aux_lambda * auxiliary
        return forecast, reconstruction, total, 0.0, 0.0, auxiliary

    def fit(self, train_loader, val_loader=None):
        """Train with optional validation, checkpoint resume and early stopping."""
        initial = self.evaluate(train_loader)
        print(f"Init total train loss: {initial[2]:.5f}")
        if val_loader is not None:
            print(f"Init total val loss: {self.evaluate(val_loader)[2]:.5f}")
        train_start = time.time()
        for epoch in range(self.start_epoch, self.n_epochs):
            epoch_start = time.time()
            train_values = self._run_loader(train_loader, training=True)
            self._record("train", train_values)
            val_values = None
            should_stop = False
            if val_loader is not None:
                val_values = self.evaluate(val_loader)
                self._record("val", val_values)
                improved = self.best_val_loss is None or (
                    val_values[2] < self.best_val_loss - self.early_stopping_min_delta
                )
                if improved:
                    self.best_val_loss = val_values[2]
                    self.early_stopping_bad_epochs = 0
                    self.save("model.pt")
                else:
                    self.early_stopping_bad_epochs += 1
                    should_stop = (
                        self.early_stopping_patience > 0
                        and self.early_stopping_bad_epochs >= self.early_stopping_patience
                    )
            self.start_epoch = epoch + 1
            elapsed = time.time() - epoch_start
            self.epoch_times.append(elapsed)
            self.save_checkpoint()
            if self.log_tensorboard:
                self.write_loss(epoch)
            if epoch % self.print_every == 0:
                message = self._format_losses(epoch, train_values, val_values, elapsed)
                print(message)
            if should_stop:
                print(f"Early stopping at epoch {epoch + 1} after {self.early_stopping_bad_epochs} non-improving epochs.")
                break
        if val_loader is None:
            self.save("model.pt")
        duration = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(duration))
        print(f"-- Training done in {duration}s.")

    def _record(self, prefix, values):
        forecast, reconstruction, total, _, _, auxiliary = values
        self.losses[f"{prefix}_forecast"].append(forecast)
        self.losses[f"{prefix}_recon"].append(reconstruction)
        self.losses[f"{prefix}_regime_aux"].append(auxiliary)
        self.losses[f"{prefix}_total"].append(total)

    @staticmethod
    def _format_losses(epoch, train_values, val_values, elapsed):
        f, r, total, _, _, aux = train_values
        message = (
            f"[Epoch {epoch + 1}] forecast_loss = {f:.5f}, recon_loss = {r:.5f}, "
            f"frozen_aux = {aux:.5f}, total_loss = {total:.5f}"
        )
        if val_values is not None:
            vf, vr, vt, _, _, va = val_values
            message += (
                f" ---- val_forecast_loss = {vf:.5f}, val_recon_loss = {vr:.5f}, "
                f"val_frozen_aux = {va:.5f}, val_total_loss = {vt:.5f}"
            )
        return f"{message} [{elapsed:.1f}s]"

    def evaluate(self, data_loader, epoch=None):
        del epoch
        return self._run_loader(data_loader, training=False)

    def save(self, file_name):
        os.makedirs(self.dload, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.dload, file_name))

    def save_checkpoint(self, file_name=None):
        os.makedirs(self.dload, exist_ok=True)
        path = os.path.join(self.dload, file_name or self.checkpoint_name)
        torch.save(
            {
                "epoch": int(self.start_epoch),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "losses": self.losses,
                "epoch_times": self.epoch_times,
                "best_val_loss": self.best_val_loss,
                "early_stopping_bad_epochs": self.early_stopping_bad_epochs,
                "n_epochs": self.n_epochs,
            },
            path,
        )

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def resume_from_checkpoint(self, file_name=None):
        path = os.path.join(self.dload, file_name or self.checkpoint_name)
        if not os.path.isfile(path):
            return False
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)
        if checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        old_losses = checkpoint.get("losses", {})
        self.losses = self._empty_losses()
        for key in self.losses:
            self.losses[key] = old_losses.get(key, [])
        self.epoch_times = checkpoint.get("epoch_times", [])
        self.best_val_loss = checkpoint.get("best_val_loss")
        self.early_stopping_bad_epochs = int(checkpoint.get("early_stopping_bad_epochs", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Loaded checkpoint from {path} (epoch={self.start_epoch})")
        return True

    def write_loss(self, epoch):
        for key, values in self.losses.items():
            if values:
                self.writer.add_scalar(key, values[-1], epoch)

    def fit_round_robin(self, train_entity_data, window_size, target_dims, val_split, shuffle_dataset):
        """Retain the generic entity round-robin schedule with frozen losses."""
        options = dict(
            num_workers=self.loader_num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.loader_persistent_workers,
            prefetch_factor=self.loader_prefetch_factor,
        )
        loaders = []
        for entity_name, entity_data in train_entity_data:
            dataset = SlidingWindowDataset(entity_data, window_size, target_dims, stride=self.window_stride)
            train_loader, val_loader, _ = create_data_loaders(
                dataset, self.batch_size, val_split, shuffle_dataset, test_dataset=None, **options
            )
            loaders.append((entity_name, train_loader, val_loader))
        train_start = time.time()
        for epoch in range(self.start_epoch, self.n_epochs):
            epoch_start = time.time()
            entity_name, train_loader, val_loader = loaders[epoch % len(loaders)]
            train_values = self._run_loader(train_loader, training=True)
            self._record("train", train_values)
            val_values = self.evaluate(val_loader) if val_loader is not None else None
            if val_values is not None:
                self._record("val", val_values)
                if self.best_val_loss is None or val_values[2] <= self.best_val_loss:
                    self.best_val_loss = val_values[2]
                    self.save("model.pt")
            self.start_epoch = epoch + 1
            elapsed = time.time() - epoch_start
            self.epoch_times.append(elapsed)
            self.save_checkpoint()
            print(f"Entity: {entity_name} | {self._format_losses(epoch, train_values, val_values, elapsed)}")
        if all(val_loader is None for _, _, val_loader in loaders):
            self.save("model.pt")
        print(f"-- Round-robin training done in {int(time.time() - train_start)}s.")
