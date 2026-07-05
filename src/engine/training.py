"""提供训练循环、验证和检查点保存工具。"""


import os

import time

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


from src.data.utils import *


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


    :param model: MTAD-GAT model

    :param optimizer: Optimizer used to minimize the loss function

    :param window_size: Length of the input sequence

    :param n_features: Number of input features

    :param target_dims: dimension of input features to forecast and reconstruct

    :param n_epochs: Number of iterations/epochs

    :param batch_size: Number of windows in a single batch

    :param init_lr: Initial learning rate of the module

    :param forecast_criterion: Loss to be used for forecasting.

    :param recon_criterion: Loss to be used for reconstruction.

    :param boolean use_cuda: To be run on GPU or not

    :param dload: Download directory where models are to be dumped

    :param log_dir: Directory where SummaryWriter logs are written to

    :param print_every: At what epoch interval to print losses

    :param log_tensorboard: Whether to log loss++ to tensorboard

    :param args_summary: Summary of args that will also be written to tensorboard if log_tensorboard

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

        self.loader_num_workers = max(int(num_workers), 0)

        self.loader_persistent_workers = bool(persistent_workers)

        self.loader_prefetch_factor = max(int(prefetch_factor), 1)

        self.window_stride = max(int(window_stride), 1)


        # 计算批次损失
        self.scaler = _build_grad_scaler(use_cuda)


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

        self.epoch_times = []

        self.start_epoch = 0

        self.best_val_loss = None

        self.checkpoint_name = "last_checkpoint.pt"


        if self.device == "cuda":

            self.model.cuda()


        if self.log_tensorboard:

            self.writer = SummaryWriter(f"{log_dir}")

            self.writer.add_text("args_summary", args_summary)


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


    @staticmethod

    def _normalized_cumsum(value, eps=1e-6):

        cumulative = torch.cumsum(value, dim=1)

        scale = cumulative.abs().amax(dim=1, keepdim=True).clamp_min(eps)

        return torch.clamp(cumulative / scale, -1.0, 1.0)


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

        current_hat = self._safe_channel(recons, self.physical_reg_config.get("current_index"))

        temperature = self._safe_channel(x, self.physical_reg_config.get("temperature_index"))

        temperature_hat = self._safe_channel(recons, self.physical_reg_config.get("temperature_index"))


        alg_terms = []

        if voltage is not None and voltage_hat is not None and x.size(1) >= 2:

            alg_terms.append(F.l1_loss(torch.diff(voltage_hat, dim=1), torch.diff(voltage, dim=1)))

        if temperature is not None and temperature_hat is not None and x.size(1) >= 2:

            alg_terms.append(F.l1_loss(torch.diff(temperature_hat, dim=1), torch.diff(temperature, dim=1)))

        if current is not None and current_hat is not None:

            alg_terms.append(

                F.l1_loss(

                    self._normalized_cumsum(current_hat),

                    self._normalized_cumsum(current),

                )

            )


        if alg_terms:

            alg_loss = torch.stack(alg_terms).mean()

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

        Train and validation (if validation loader given) losses stored in self.losses


        :param train_loader: train loader of input data

        :param val_loader: validation loader of input data

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

                    loss = (

                        forecast_loss

                        + recon_loss

                        + reg_scale * self.physical_alg_lambda * phys_alg_loss

                        + reg_scale * self.physical_smooth_lambda * phys_smooth_loss

                    )


                self.scaler.scale(loss).backward()

                self.scaler.step(self.optimizer)

                self.scaler.update()


                forecast_b_losses.append(forecast_loss.item())

                recon_b_losses.append(recon_loss.item())

                phys_alg_b_losses.append(phys_alg_loss.item())

                phys_smooth_b_losses.append(phys_smooth_loss.item())


            forecast_b_losses = np.array(forecast_b_losses)

            recon_b_losses = np.array(recon_b_losses)

            phys_alg_b_losses = np.array(phys_alg_b_losses)

            phys_smooth_b_losses = np.array(phys_smooth_b_losses)


            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())

            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())

            phys_alg_epoch_loss = np.sqrt((phys_alg_b_losses ** 2).mean()) if len(phys_alg_b_losses) else 0.0

            phys_smooth_epoch_loss = np.sqrt((phys_smooth_b_losses ** 2).mean()) if len(phys_smooth_b_losses) else 0.0

            total_epoch_loss = (

                forecast_epoch_loss

                + recon_epoch_loss

                + reg_scale * self.physical_alg_lambda * phys_alg_epoch_loss

                + reg_scale * self.physical_smooth_lambda * phys_smooth_epoch_loss

            )


            self.losses["train_forecast"].append(forecast_epoch_loss)

            self.losses["train_recon"].append(recon_epoch_loss)

            self.losses["train_phys_alg"].append(phys_alg_epoch_loss)

            self.losses["train_phys_smooth"].append(phys_smooth_epoch_loss)

            self.losses["train_total"].append(total_epoch_loss)


            # 保存最佳模型

            forecast_val_loss, recon_val_loss, total_val_loss = "NA", "NA", "NA"

            phys_alg_val_loss, phys_smooth_val_loss = "NA", "NA"

            if val_loader is not None:

                forecast_val_loss, recon_val_loss, total_val_loss, phys_alg_val_loss, phys_smooth_val_loss = self.evaluate(

                    val_loader, epoch=epoch

                )

                self.losses["val_forecast"].append(forecast_val_loss)

                self.losses["val_recon"].append(recon_val_loss)

                self.losses["val_phys_alg"].append(phys_alg_val_loss)

                self.losses["val_phys_smooth"].append(phys_smooth_val_loss)

                self.losses["val_total"].append(total_val_loss)


                if self.best_val_loss is None or total_val_loss <= self.best_val_loss:

                    self.best_val_loss = total_val_loss

                    self.save(f"model.pt")


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

                    f"total_loss = {total_epoch_loss:.5f}"

                )


                if val_loader is not None:

                    s += (

                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "

                        f"val_recon_loss = {recon_val_loss:.5f}, "

                        f"val_phys_alg = {phys_alg_val_loss:.5f}, "

                        f"val_phys_smooth = {phys_smooth_val_loss:.5f}, "

                        f"val_total_loss = {total_val_loss:.5f}"

                    )


                s += f" [{epoch_time:.1f}s]"

                print(s)


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


                forecast_losses.append(forecast_loss.item())

                recon_losses.append(recon_loss.item())

                phys_alg_losses.append(phys_alg_loss.item())

                phys_smooth_losses.append(phys_smooth_loss.item())


        forecast_losses = np.array(forecast_losses)

        recon_losses = np.array(recon_losses)

        phys_alg_losses = np.array(phys_alg_losses)

        phys_smooth_losses = np.array(phys_smooth_losses)


        forecast_loss = np.sqrt((forecast_losses ** 2).mean())

        recon_loss = np.sqrt((recon_losses ** 2).mean())

        phys_alg_loss = np.sqrt((phys_alg_losses ** 2).mean()) if len(phys_alg_losses) else 0.0

        phys_smooth_loss = np.sqrt((phys_smooth_losses ** 2).mean()) if len(phys_smooth_losses) else 0.0

        total_loss = (

            forecast_loss

            + recon_loss

            + reg_scale * self.physical_alg_lambda * phys_alg_loss

            + reg_scale * self.physical_smooth_lambda * phys_smooth_loss

        )


        return forecast_loss, recon_loss, total_loss, phys_alg_loss, phys_smooth_loss


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

                    loss = (

                        forecast_loss

                        + recon_loss

                        + reg_scale * self.physical_alg_lambda * phys_alg_loss

                        + reg_scale * self.physical_smooth_lambda * phys_smooth_loss

                    )


                self.scaler.scale(loss).backward()

                self.scaler.step(self.optimizer)

                self.scaler.update()


                forecast_b_losses.append(forecast_loss.item())

                recon_b_losses.append(recon_loss.item())

                phys_alg_b_losses.append(phys_alg_loss.item())

                phys_smooth_b_losses.append(phys_smooth_loss.item())


            forecast_b_losses = np.array(forecast_b_losses)

            recon_b_losses = np.array(recon_b_losses)

            phys_alg_b_losses = np.array(phys_alg_b_losses)

            phys_smooth_b_losses = np.array(phys_smooth_b_losses)


            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())

            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())

            phys_alg_epoch_loss = np.sqrt((phys_alg_b_losses ** 2).mean()) if len(phys_alg_b_losses) else 0.0

            phys_smooth_epoch_loss = np.sqrt((phys_smooth_b_losses ** 2).mean()) if len(phys_smooth_b_losses) else 0.0

            total_epoch_loss = (

                forecast_epoch_loss

                + recon_epoch_loss

                + reg_scale * self.physical_alg_lambda * phys_alg_epoch_loss

                + reg_scale * self.physical_smooth_lambda * phys_smooth_epoch_loss

            )


            self.losses["train_forecast"].append(forecast_epoch_loss)

            self.losses["train_recon"].append(recon_epoch_loss)

            self.losses["train_phys_alg"].append(phys_alg_epoch_loss)

            self.losses["train_phys_smooth"].append(phys_smooth_epoch_loss)

            self.losses["train_total"].append(total_epoch_loss)


            # 预测阶段

            forecast_val_loss, recon_val_loss, total_val_loss = "NA", "NA", "NA"

            phys_alg_val_loss, phys_smooth_val_loss = "NA", "NA"

            if val_loader is not None:

                forecast_val_loss, recon_val_loss, total_val_loss, phys_alg_val_loss, phys_smooth_val_loss = self.evaluate(

                    val_loader, epoch=epoch

                )

                self.losses["val_forecast"].append(forecast_val_loss)

                self.losses["val_recon"].append(recon_val_loss)

                self.losses["val_phys_alg"].append(phys_alg_val_loss)

                self.losses["val_phys_smooth"].append(phys_smooth_val_loss)

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

                    f"total_loss = {total_epoch_loss:.5f}"

                )


                if val_loader is not None:

                    s += (

                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "

                        f"val_recon_loss = {recon_val_loss:.5f}, "

                        f"val_phys_alg = {phys_alg_val_loss:.5f}, "

                        f"val_phys_smooth = {phys_smooth_val_loss:.5f}, "

                        f"val_total_loss = {total_val_loss:.5f}"

                    )


                s += f" [{epoch_time:.1f}s]"

                print(s)


        train_time = int(time.time() - train_start)

        if self.log_tensorboard:

            self.writer.add_text("total_train_time", str(train_time))

        print(f"-- Round-robin training done in {train_time}s.")


