"""提供训练循环、验证和检查点保存工具。"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data.utils import *
from src.models.physical_response import compute_torch_physical_response_errors


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
    """MTAD-GAT 模型的训练器类。
    :param model: 待训练的 MTAD-GAT 模型
    :param optimizer: 用于最小化损失函数的优化器
    :param window_size: 输入历史窗口长度
    :param n_features: 输入特征数量
    :param target_dims: 需要预测和重构的目标通道
    :param n_epochs: 训练轮数
    :param batch_size: 每个批次的窗口数量
    :param init_lr: 初始学习率
    :param forecast_criterion: 预测分支使用的损失函数
    :param recon_criterion: 重构分支使用的损失函数
    :param boolean use_cuda: 是否使用 GPU
    :param dload: 模型和检查点保存目录
    :param log_dir: 训练日志保存目录
    :param print_every: 每隔多少轮打印一次损失
    :param log_tensorboard: 是否写入 TensorBoard
    :param args_summary: 需要随日志保存的实验参数摘要
    """

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
        use_physical_regularization=False,
        physical_reg_config=None,
        physical_reg_warmup_ratio=0.2,
        physical_alg_lambda=0.1,
        physical_smooth_lambda=0.01,
        physical_transition_threshold=0.05,
        physical_transition_relax=0.1,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        window_stride=1,
        regime_aux_lambda=0.0,
        regime_group_dro_lambda=0.0,
        regime_group_dro_temperature=0.05,
        sparse_graph_lambda=0.0,
        normal_tail_lambda=0.0,
        normal_tail_fraction=0.1,
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
        self.use_physical_regularization = use_physical_regularization and physical_reg_config is not None
        self.physical_reg_config = dict(physical_reg_config or {}) if physical_reg_config is not None else None
        self.physical_reg_warmup_ratio = max(0.0, float(physical_reg_warmup_ratio))
        self.physical_alg_lambda = float(physical_alg_lambda)
        self.physical_smooth_lambda = float(physical_smooth_lambda)
        self.physical_transition_threshold = float(physical_transition_threshold)
        self.physical_transition_relax = float(physical_transition_relax)
        self.last_physical_response_terms = {}
        self.loader_num_workers = max(int(num_workers), 0)
        self.loader_persistent_workers = bool(persistent_workers)
        self.loader_prefetch_factor = max(int(prefetch_factor), 1)
        self.window_stride = max(int(window_stride), 1)
        self.regime_aux_lambda = max(0.0, float(regime_aux_lambda))
        self.regime_group_dro_lambda = max(0.0, float(regime_group_dro_lambda))
        self.regime_group_dro_temperature = max(1e-4, float(regime_group_dro_temperature))
        self.sparse_graph_lambda = max(0.0, float(sparse_graph_lambda))
        self.normal_tail_lambda = max(0.0, float(normal_tail_lambda))
        self.normal_tail_fraction = min(max(float(normal_tail_fraction), 1e-3), 1.0)
        self.early_stopping_patience = max(0, int(early_stopping_patience))
        self.early_stopping_min_delta = max(0.0, float(early_stopping_min_delta))
        self.early_stopping_bad_epochs = 0
        # 计算批次损失
        self.scaler = _build_grad_scaler(use_cuda)
        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "train_phys_alg": [],
            "train_phys_smooth": [],
            "train_regime_aux": [],
            "train_regime_group_dro": [],
            "train_sparse_graph": [],
            "train_normal_tail": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
            "val_phys_alg": [],
            "val_phys_smooth": [],
            "val_regime_aux": [],
        }
        self.epoch_times = []
        self.start_epoch = 0
        self.best_val_loss = None
        self.checkpoint_name = "last_checkpoint.pt"
        if self.device == "cuda":
            self.model.cuda()
        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)

    def _compute_regime_group_dro(self, x, y_target, preds, x_target, recons):
        """均衡正常样本在电流/SOC四个相对工况组上的响应风险。

        工况组只由当前批次内的平均电流和平均SOC秩确定，不读取电压、温度、
        故障标签或测试分布。该项只改变训练目标，不增加推理分支，因而与C4的
        控制量到响应量物理预测头保持独立。
        """
        if self.regime_group_dro_lambda <= 0.0 or self.physical_reg_config is None:
            return x.new_tensor(0.0)
        current_index = self.physical_reg_config.get("current_index")
        soc_index = self.physical_reg_config.get("soc_index")
        response_dims = self.physical_reg_config.get("consistency_response_dims")
        if current_index is None or soc_index is None or not response_dims or x.size(0) < 4:
            return x.new_tensor(0.0)

        current_level = x[:, :, current_index].mean(dim=1)
        soc_level = x[:, :, soc_index].mean(dim=1)
        current_high = current_level > current_level.detach().median()
        soc_high = soc_level > soc_level.detach().median()
        group_ids = current_high.long() + 2 * soc_high.long()

        # 完整七通道训练时只在五个响应量上形成鲁棒风险；若模型已经只输出
        # 响应量，则本地输出维度全部属于响应。
        if self.target_dims is None:
            local_response_dims = [dim for dim in response_dims if dim < preds.size(-1)]
        else:
            local_response_dims = list(range(preds.size(-1)))
        if not local_response_dims:
            return x.new_tensor(0.0)
        pred_error = (preds[:, local_response_dims] - y_target[:, local_response_dims]).square().mean(dim=1)
        recon_error = (
            recons[:, :, local_response_dims] - x_target[:, :, local_response_dims]
        ).square().mean(dim=(1, 2))
        sample_risk = pred_error.sqrt() + recon_error.sqrt()

        group_risks = []
        for group_id in range(4):
            mask = group_ids == group_id
            if mask.any():
                group_risks.append(sample_risk[mask].mean())
        if len(group_risks) < 2:
            return x.new_tensor(0.0)
        group_risks = torch.stack(group_risks)
        temperature = self.regime_group_dro_temperature
        smooth_worst = temperature * (
            torch.logsumexp(group_risks / temperature, dim=0)
            - np.log(float(group_risks.numel()))
        )
        return torch.clamp_min(smooth_worst - group_risks.mean(), 0.0)

    def _compute_normal_tail_excess(self, y_target, preds, x_target, recons):
        """Penalize dispersion of the highest normal residuals without using fault labels."""
        if self.normal_tail_lambda <= 0.0 or preds.size(0) < 2:
            return preds.new_tensor(0.0)
        forecast_risk = (preds - y_target).square().mean(dim=1).sqrt()
        recon_risk = (recons - x_target).square().mean(dim=(1, 2)).sqrt()
        sample_risk = forecast_risk + recon_risk
        count = max(1, int(np.ceil(sample_risk.numel() * self.normal_tail_fraction)))
        tail_risk = torch.topk(sample_risk, count, largest=True).values.mean()
        # The ordinary objective already minimizes the mean. Penalizing only the excess
        # targets the normal tail that later becomes false alarms.
        return torch.clamp_min(tail_risk - sample_risk.mean(), 0.0)

    def _get_reg_scale(self, epoch):
        if not self.use_physical_regularization:
            return 0.0
        warmup_epochs = max(int(round(self.n_epochs * self.physical_reg_warmup_ratio)), 0)
        if warmup_epochs <= 0:
            return 1.0
        return min(float(epoch + 1) / float(warmup_epochs), 1.0)

    @staticmethod
    def _safe_channel(x, index):
        if index is None or index < 0 or index >= x.size(2):
            return None
        return x[:, :, index:index + 1]

    def _compute_phase_signal(self, x, current):
        step_type = self._safe_channel(x, self.physical_reg_config.get("step_type_index"))
        if step_type is not None:
            return torch.clamp(step_type, -1.0, 1.0)
        if current is None:
            return x.new_zeros(x.size(0), x.size(1), 1)
        current_scale = current.abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
        normalized_current = current / current_scale
        threshold = self.physical_transition_threshold
        phase = torch.zeros_like(normalized_current)
        phase = torch.where(normalized_current > threshold, torch.ones_like(phase), phase)
        phase = torch.where(normalized_current < -threshold, -torch.ones_like(phase), phase)
        return phase

    def _compute_physical_regularization(self, x, recons):
        if not self.use_physical_regularization or self.physical_reg_config is None:
            zero = x.new_tensor(0.0)
            return zero, zero
        voltage = self._safe_channel(x, self.physical_reg_config.get("voltage_index"))
        voltage_hat = self._safe_channel(recons, self.physical_reg_config.get("voltage_index"))
        current = self._safe_channel(x, self.physical_reg_config.get("current_index"))
        temperature_hat = self._safe_channel(recons, self.physical_reg_config.get("temperature_index"))
        response_errors = compute_torch_physical_response_errors(x, recons, self.physical_reg_config)
        self.last_physical_response_terms = {
            name: float(value.detach().cpu()) for name, value in response_errors.items()
        }
        if response_errors:
            alg_loss = torch.stack(list(response_errors.values())).mean()
        else:
            alg_loss = x.new_tensor(0.0)
        smooth_terms = []
        phase = self._compute_phase_signal(x, current)
        if x.size(1) >= 3:
            boundary = (phase[:, 1:, :] - phase[:, :-1, :]).abs() > 0
            transition_mask = torch.logical_or(boundary[:, 1:, :], boundary[:, :-1, :]).float()
            weights = torch.ones_like(transition_mask) * self.physical_transition_relax
            weights = torch.where(transition_mask > 0, weights, torch.ones_like(weights))
            smooth_targets = []
            if self.physical_reg_config.get("smooth_voltage", False) and voltage_hat is not None:
                smooth_targets.append(voltage_hat)
            if self.physical_reg_config.get("smooth_temperature", True) and temperature_hat is not None:
                smooth_targets.append(temperature_hat)
            for series in smooth_targets:
                second_diff = series[:, 2:, :] - 2.0 * series[:, 1:-1, :] + series[:, :-2, :]
                smooth_terms.append((weights * second_diff.abs()).mean())
        if smooth_terms:
            smooth_loss = torch.stack(smooth_terms).mean()
        else:
            smooth_loss = x.new_tensor(0.0)
        return alg_loss, smooth_loss

    def fit(self, train_loader, val_loader=None):
        """训练模型 self.n_epochs 轮。
        每轮训练损失和可选验证损失都保存到 ``self.losses``。
        :param train_loader: 训练数据加载器
        :param val_loader: 可选的验证数据加载器
        """
        if self.start_epoch == 0:
            init_train_loss = self.evaluate(train_loader)
            print(f"Init total train loss: {init_train_loss[2]:5f}")
            if val_loader is not None:
                init_val_loss = self.evaluate(val_loader)
                print(f"Init total val loss: {init_val_loss[2]:.5f}")
        else:
            print(f"Resuming training from epoch {self.start_epoch + 1}/{self.n_epochs}")
        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()
        for epoch in range(self.start_epoch, self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_b_losses = []
            recon_b_losses = []
            phys_alg_b_losses = []
            phys_smooth_b_losses = []
            regime_aux_b_losses = []
            regime_group_dro_b_losses = []
            sparse_graph_b_losses = []
            normal_tail_b_losses = []
            reg_scale = self._get_reg_scale(epoch)
            should_stop = False
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                with _autocast_context(self.device):
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
                    forecast_loss = torch.sqrt(self.forecast_criterion(y_target, preds))
                    recon_loss = torch.sqrt(self.recon_criterion(x_target, recons))
                    phys_alg_loss, phys_smooth_loss = self._compute_physical_regularization(x_target, recons)
                    base_model = _unwrap_model(self.model)
                    if self.regime_aux_lambda > 0.0 and hasattr(base_model, "regime_auxiliary_loss"):
                        regime_aux_loss = base_model.regime_auxiliary_loss(x)
                    else:
                        regime_aux_loss = x.new_tensor(0.0)
                    regime_group_dro_loss = self._compute_regime_group_dro(
                        x, y_target, preds, x_target, recons
                    )
                    normal_tail_loss = self._compute_normal_tail_excess(
                        y_target, preds, x_target, recons
                    )
                    if self.sparse_graph_lambda > 0.0 and hasattr(base_model.feature_gat, "sparse_graph_regularization"):
                        sparse_graph_loss = base_model.feature_gat.sparse_graph_regularization()
                    else:
                        sparse_graph_loss = x.new_tensor(0.0)
                    loss = (
                        forecast_loss
                        + recon_loss
                        + reg_scale * self.physical_alg_lambda * phys_alg_loss
                        + reg_scale * self.physical_smooth_lambda * phys_smooth_loss
                        + self.regime_aux_lambda * regime_aux_loss
                        + self.regime_group_dro_lambda * regime_group_dro_loss
                        + self.sparse_graph_lambda * sparse_graph_loss
                        + self.normal_tail_lambda * normal_tail_loss
                    )
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                forecast_b_losses.append(forecast_loss.item())
                recon_b_losses.append(recon_loss.item())
                phys_alg_b_losses.append(phys_alg_loss.item())
                phys_smooth_b_losses.append(phys_smooth_loss.item())
                regime_aux_b_losses.append(regime_aux_loss.item())
                regime_group_dro_b_losses.append(regime_group_dro_loss.item())
                sparse_graph_b_losses.append(sparse_graph_loss.item())
                normal_tail_b_losses.append(normal_tail_loss.item())
            forecast_b_losses = np.array(forecast_b_losses)
            recon_b_losses = np.array(recon_b_losses)
            phys_alg_b_losses = np.array(phys_alg_b_losses)
            phys_smooth_b_losses = np.array(phys_smooth_b_losses)
            regime_aux_b_losses = np.array(regime_aux_b_losses)
            regime_group_dro_b_losses = np.array(regime_group_dro_b_losses)
            sparse_graph_b_losses = np.array(sparse_graph_b_losses)
            normal_tail_b_losses = np.array(normal_tail_b_losses)
            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())
            phys_alg_epoch_loss = np.sqrt((phys_alg_b_losses ** 2).mean()) if len(phys_alg_b_losses) else 0.0
            phys_smooth_epoch_loss = np.sqrt((phys_smooth_b_losses ** 2).mean()) if len(phys_smooth_b_losses) else 0.0
            regime_aux_epoch_loss = np.sqrt((regime_aux_b_losses ** 2).mean()) if len(regime_aux_b_losses) else 0.0
            regime_group_dro_epoch_loss = (
                np.sqrt((regime_group_dro_b_losses ** 2).mean())
                if len(regime_group_dro_b_losses) else 0.0
            )
            sparse_graph_epoch_loss = (
                np.sqrt((sparse_graph_b_losses ** 2).mean())
                if len(sparse_graph_b_losses) else 0.0
            )
            normal_tail_epoch_loss = (
                np.sqrt((normal_tail_b_losses ** 2).mean())
                if len(normal_tail_b_losses) else 0.0
            )
            total_epoch_loss = (
                forecast_epoch_loss
                + recon_epoch_loss
                + reg_scale * self.physical_alg_lambda * phys_alg_epoch_loss
                + reg_scale * self.physical_smooth_lambda * phys_smooth_epoch_loss
                + self.regime_aux_lambda * regime_aux_epoch_loss
                + self.regime_group_dro_lambda * regime_group_dro_epoch_loss
                + self.sparse_graph_lambda * sparse_graph_epoch_loss
                + self.normal_tail_lambda * normal_tail_epoch_loss
            )
            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_phys_alg"].append(phys_alg_epoch_loss)
            self.losses["train_phys_smooth"].append(phys_smooth_epoch_loss)
            self.losses["train_regime_aux"].append(regime_aux_epoch_loss)
            self.losses["train_regime_group_dro"].append(regime_group_dro_epoch_loss)
            self.losses["train_sparse_graph"].append(sparse_graph_epoch_loss)
            self.losses["train_normal_tail"].append(normal_tail_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)
            # 保存最佳模型
            forecast_val_loss, recon_val_loss, total_val_loss = "NA", "NA", "NA"
            phys_alg_val_loss, phys_smooth_val_loss, regime_aux_val_loss = "NA", "NA", "NA"
            if val_loader is not None:
                forecast_val_loss, recon_val_loss, total_val_loss, phys_alg_val_loss, phys_smooth_val_loss, regime_aux_val_loss = self.evaluate(
                    val_loader, epoch=epoch
                )
                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_recon"].append(recon_val_loss)
                self.losses["val_phys_alg"].append(phys_alg_val_loss)
                self.losses["val_phys_smooth"].append(phys_smooth_val_loss)
                self.losses["val_regime_aux"].append(regime_aux_val_loss)
                self.losses["val_total"].append(total_val_loss)
                improved = (
                    self.best_val_loss is None
                    or total_val_loss < self.best_val_loss - self.early_stopping_min_delta
                )
                if improved:
                    self.best_val_loss = total_val_loss
                    self.early_stopping_bad_epochs = 0
                    self.save(f"model.pt")
                else:
                    self.early_stopping_bad_epochs += 1
                    should_stop = (
                        self.early_stopping_patience > 0
                        and self.early_stopping_bad_epochs >= self.early_stopping_patience
                    )
            if self.log_tensorboard:
                self.write_loss(epoch)
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            self.start_epoch = epoch + 1
            self.save_checkpoint()
            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"phys_alg = {phys_alg_epoch_loss:.5f}, "
                    f"phys_smooth = {phys_smooth_epoch_loss:.5f}, "
                    f"regime_aux = {regime_aux_epoch_loss:.5f}, "
                    f"regime_group_dro = {regime_group_dro_epoch_loss:.5f}, "
                    f"sparse_graph = {sparse_graph_epoch_loss:.5f}, "
                    f"normal_tail = {normal_tail_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )
                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_recon_loss = {recon_val_loss:.5f}, "
                        f"val_phys_alg = {phys_alg_val_loss:.5f}, "
                        f"val_phys_smooth = {phys_smooth_val_loss:.5f}, "
                        f"val_regime_aux = {regime_aux_val_loss:.5f}, "
                        f"val_total_loss = {total_val_loss:.5f}"
                    )
                s += f" [{epoch_time:.1f}s]"
                print(s)
            if should_stop:
                print(
                    f"Early stopping at epoch {epoch + 1}: validation loss did not improve by "
                    f"{self.early_stopping_min_delta:g} for {self.early_stopping_patience} epochs."
                )
                break
        if val_loader is None:
            self.save(f"model.pt")
        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader, epoch=None):
        """评估模型。
        :param data_loader: data loader of input data
        :return forecasting loss, reconstruction loss, total loss
        """
        self.model.eval()
        forecast_losses = []
        recon_losses = []
        phys_alg_losses = []
        phys_smooth_losses = []
        regime_aux_losses = []
        normal_tail_losses = []
        reg_scale = self._get_reg_scale(self.start_epoch - 1 if epoch is None else epoch)
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                with _autocast_context(self.device):
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
                    forecast_loss = torch.sqrt(self.forecast_criterion(y_target, preds))
                    recon_loss = torch.sqrt(self.recon_criterion(x_target, recons))
                    phys_alg_loss, phys_smooth_loss = self._compute_physical_regularization(x_target, recons)
                    base_model = _unwrap_model(self.model)
                    if self.regime_aux_lambda > 0.0 and hasattr(base_model, "regime_auxiliary_loss"):
                        regime_aux_loss = base_model.regime_auxiliary_loss(x)
                    else:
                        regime_aux_loss = x.new_tensor(0.0)
                    normal_tail_loss = self._compute_normal_tail_excess(
                        y_target, preds, x_target, recons
                    )
                forecast_losses.append(forecast_loss.item())
                recon_losses.append(recon_loss.item())
                phys_alg_losses.append(phys_alg_loss.item())
                phys_smooth_losses.append(phys_smooth_loss.item())
                regime_aux_losses.append(regime_aux_loss.item())
                normal_tail_losses.append(normal_tail_loss.item())
        forecast_losses = np.array(forecast_losses)
        recon_losses = np.array(recon_losses)
        phys_alg_losses = np.array(phys_alg_losses)
        phys_smooth_losses = np.array(phys_smooth_losses)
        regime_aux_losses = np.array(regime_aux_losses)
        normal_tail_losses = np.array(normal_tail_losses)
        forecast_loss = np.sqrt((forecast_losses ** 2).mean())
        recon_loss = np.sqrt((recon_losses ** 2).mean())
        phys_alg_loss = np.sqrt((phys_alg_losses ** 2).mean()) if len(phys_alg_losses) else 0.0
        phys_smooth_loss = np.sqrt((phys_smooth_losses ** 2).mean()) if len(phys_smooth_losses) else 0.0
        regime_aux_loss = np.sqrt((regime_aux_losses ** 2).mean()) if len(regime_aux_losses) else 0.0
        normal_tail_loss = np.sqrt((normal_tail_losses ** 2).mean()) if len(normal_tail_losses) else 0.0
        total_loss = (
            forecast_loss
            + recon_loss
            + reg_scale * self.physical_alg_lambda * phys_alg_loss
            + reg_scale * self.physical_smooth_lambda * phys_smooth_loss
            + self.regime_aux_lambda * regime_aux_loss
            + self.normal_tail_lambda * normal_tail_loss
        )
        return forecast_loss, recon_loss, total_loss, phys_alg_loss, phys_smooth_loss, regime_aux_loss

    def save(self, file_name):
        """
        将模型参数序列化保存，供后续读取。
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        # 预测阶段
        if not os.path.exists(self.dload):
            os.makedirs(self.dload)
        PATH = os.path.join(self.dload, file_name)
        torch.save(self.model.state_dict(), PATH)

    def save_checkpoint(self, file_name=None):
        if file_name is None:
            file_name = self.checkpoint_name
        if not os.path.exists(self.dload):
            os.makedirs(self.dload)
        path = os.path.join(self.dload, file_name)
        checkpoint = {
            "epoch": int(self.start_epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "losses": self.losses,
            "epoch_times": self.epoch_times,
            "best_val_loss": self.best_val_loss,
            "early_stopping_bad_epochs": self.early_stopping_bad_epochs,
            "n_epochs": self.n_epochs,
        }
        torch.save(checkpoint, path)

    def load(self, PATH):
        """
        从指定路径加载模型参数。
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def resume_from_checkpoint(self, file_name=None):
        if file_name is None:
            file_name = self.checkpoint_name
        path = os.path.join(self.dload, file_name)
        if not os.path.isfile(path):
            return False
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)
        scaler_state_dict = checkpoint.get("scaler_state_dict")
        if scaler_state_dict is not None:
            self.scaler.load_state_dict(scaler_state_dict)
        self.losses = checkpoint.get("losses", self.losses)
        self.epoch_times = checkpoint.get("epoch_times", [])
        self.best_val_loss = checkpoint.get("best_val_loss")
        self.early_stopping_bad_epochs = int(checkpoint.get("early_stopping_bad_epochs", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Loaded checkpoint from {path} (epoch={self.start_epoch})")
        return True

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)

    def fit_round_robin(self, train_entity_data, window_size, target_dims, val_split, shuffle_dataset):
        """
        实现轮流训练：每个 epoch 使用不同实体的数据。
        :param train_entity_data: List of tuples (entity_name, entity_tensor_data)
        :param window_size: Length of the input sequence
        :param target_dims: dimension of input features to forecast and reconstruct
        :param val_split: Validation split ratio
        :param shuffle_dataset: Whether to shuffle the dataset
        """
        print(f"Starting round-robin training with {len(train_entity_data)} entities")
        # 数据准备
        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "train_phys_alg": [],
            "train_phys_smooth": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
            "val_phys_alg": [],
            "val_phys_smooth": [],
        }
        # 训练前准备
        init_train_losses = []
        loader_options = resolve_dataloader_options(
            num_workers=self.loader_num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.loader_persistent_workers,
            prefetch_factor=self.loader_prefetch_factor,
        )
        for entity_name, entity_data in train_entity_data:
            entity_dataset = SlidingWindowDataset(
                entity_data,
                window_size,
                target_dims,
                stride=self.window_stride,
            )
            entity_loader = DataLoader(
                entity_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle_dataset,
                **loader_options,
            )
            init_loss = self.evaluate(entity_loader)
            init_train_losses.append(init_loss)
            print(f"Init total train loss for {entity_name}: {init_loss[2]:.5f}")
        print(f"Training model for {self.n_epochs} epochs using round-robin approach..")
        train_start = time.time()
        # 预先创建数据加载器以提升效率
        print("Pre-creating DataLoaders for all entities...")
        entity_loaders = []
        for entity_name, entity_data in train_entity_data:
            entity_dataset = SlidingWindowDataset(
                entity_data,
                window_size,
                target_dims,
                stride=self.window_stride,
            )
            train_loader, val_loader, _ = create_data_loaders(
                entity_dataset,
                self.batch_size,
                val_split,
                shuffle_dataset,
                test_dataset=None,
                num_workers=self.loader_num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=self.loader_persistent_workers,
                prefetch_factor=self.loader_prefetch_factor,
            )
            entity_loaders.append((entity_name, train_loader, val_loader))
        # 训练循环
        if self.start_epoch > 0:
            print(f"Resuming round-robin training from epoch {self.start_epoch + 1}/{self.n_epochs}")
        for epoch in range(self.start_epoch, self.n_epochs):
            epoch_start = time.time()
            print(f"[Epoch {epoch + 1}] Starting round-robin training")
            # 每个训练轮次选择一个实体进行训练
            entity_idx = epoch % len(train_entity_data)
            entity_name, train_loader, val_loader = entity_loaders[entity_idx]
            print(f"[Epoch {epoch + 1}] Training on entity: {entity_name}")
            # 训练一个轮次
            self.model.train()
            forecast_b_losses = []
            recon_b_losses = []
            phys_alg_b_losses = []
            phys_smooth_b_losses = []
            regime_aux_b_losses = []
            reg_scale = self._get_reg_scale(epoch)
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                with _autocast_context(self.device):
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
                    forecast_loss = torch.sqrt(self.forecast_criterion(y_target, preds))
                    recon_loss = torch.sqrt(self.recon_criterion(x_target, recons))
                    phys_alg_loss, phys_smooth_loss = self._compute_physical_regularization(x_target, recons)
                    base_model = _unwrap_model(self.model)
                    if self.regime_aux_lambda > 0.0 and hasattr(base_model, "regime_auxiliary_loss"):
                        regime_aux_loss = base_model.regime_auxiliary_loss(x)
                    else:
                        regime_aux_loss = x.new_tensor(0.0)
                    loss = (
                        forecast_loss
                        + recon_loss
                        + reg_scale * self.physical_alg_lambda * phys_alg_loss
                        + reg_scale * self.physical_smooth_lambda * phys_smooth_loss
                        + self.regime_aux_lambda * regime_aux_loss
                    )
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                forecast_b_losses.append(forecast_loss.item())
                recon_b_losses.append(recon_loss.item())
                phys_alg_b_losses.append(phys_alg_loss.item())
                phys_smooth_b_losses.append(phys_smooth_loss.item())
                regime_aux_b_losses.append(regime_aux_loss.item())
            forecast_b_losses = np.array(forecast_b_losses)
            recon_b_losses = np.array(recon_b_losses)
            phys_alg_b_losses = np.array(phys_alg_b_losses)
            phys_smooth_b_losses = np.array(phys_smooth_b_losses)
            regime_aux_b_losses = np.array(regime_aux_b_losses)
            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())
            phys_alg_epoch_loss = np.sqrt((phys_alg_b_losses ** 2).mean()) if len(phys_alg_b_losses) else 0.0
            phys_smooth_epoch_loss = np.sqrt((phys_smooth_b_losses ** 2).mean()) if len(phys_smooth_b_losses) else 0.0
            regime_aux_epoch_loss = np.sqrt((regime_aux_b_losses ** 2).mean()) if len(regime_aux_b_losses) else 0.0
            total_epoch_loss = (
                forecast_epoch_loss
                + recon_epoch_loss
                + reg_scale * self.physical_alg_lambda * phys_alg_epoch_loss
                + reg_scale * self.physical_smooth_lambda * phys_smooth_epoch_loss
                + self.regime_aux_lambda * regime_aux_epoch_loss
            )
            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_phys_alg"].append(phys_alg_epoch_loss)
            self.losses["train_phys_smooth"].append(phys_smooth_epoch_loss)
            self.losses["train_regime_aux"].append(regime_aux_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)
            # 预测阶段
            forecast_val_loss, recon_val_loss, total_val_loss = "NA", "NA", "NA"
            phys_alg_val_loss, phys_smooth_val_loss, regime_aux_val_loss = "NA", "NA", "NA"
            if val_loader is not None:
                forecast_val_loss, recon_val_loss, total_val_loss, phys_alg_val_loss, phys_smooth_val_loss, regime_aux_val_loss = self.evaluate(
                    val_loader, epoch=epoch
                )
                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_recon"].append(recon_val_loss)
                self.losses["val_phys_alg"].append(phys_alg_val_loss)
                self.losses["val_phys_smooth"].append(phys_smooth_val_loss)
                self.losses["val_regime_aux"].append(regime_aux_val_loss)
                self.losses["val_total"].append(total_val_loss)
                # 保存训练配置
                if self.best_val_loss is None or total_val_loss <= self.best_val_loss:
                    self.best_val_loss = total_val_loss
                    self.save(f"model.pt")
                    print(f"[Epoch {epoch + 1}] New best model saved with val loss: {total_val_loss}")
            if self.log_tensorboard:
                self.write_loss(epoch)
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            self.start_epoch = epoch + 1
            self.save_checkpoint()
            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] Entity: {entity_name} | "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"phys_alg = {phys_alg_epoch_loss:.5f}, "
                    f"phys_smooth = {phys_smooth_epoch_loss:.5f}, "
                    f"regime_aux = {regime_aux_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )
                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_recon_loss = {recon_val_loss:.5f}, "
                        f"val_phys_alg = {phys_alg_val_loss:.5f}, "
                        f"val_phys_smooth = {phys_smooth_val_loss:.5f}, "
                        f"val_regime_aux = {regime_aux_val_loss:.5f}, "
                        f"val_total_loss = {total_val_loss:.5f}"
                    )
                s += f" [{epoch_time:.1f}s]"
                print(s)
        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Round-robin training done in {train_time}s.")
