import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils import *


class Trainer:
    """Trainer class for MTAD-GAT model.

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
        log_dir="output/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
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

        # 混合精度训练设置
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))

        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader, val_loader=None):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        :param train_loader: train loader of input data
        :param val_loader: validation loader of input data
        """

        init_train_loss = self.evaluate(train_loader)
        print(f"Init total train loss: {init_train_loss[2]:5f}")

        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader)
            print(f"Init total val loss: {init_val_loss[2]:.5f}")

        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_b_losses = []
            recon_b_losses = []

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
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
                    loss = forecast_loss + recon_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                forecast_b_losses.append(forecast_loss.item())
                recon_b_losses.append(recon_loss.item())

            forecast_b_losses = np.array(forecast_b_losses)
            recon_b_losses = np.array(recon_b_losses)

            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())

            total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)

            # Evaluate on validation set
            forecast_val_loss, recon_val_loss, total_val_loss = "NA", "NA", "NA"
            if val_loader is not None:
                forecast_val_loss, recon_val_loss, total_val_loss = self.evaluate(val_loader)
                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_recon"].append(recon_val_loss)
                self.losses["val_total"].append(total_val_loss)

                if total_val_loss <= self.losses["val_total"][-1]:
                    self.save(f"model.pt")

            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_recon_loss = {recon_val_loss:.5f}, "
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

    def evaluate(self, data_loader):
        """Evaluate model

        :param data_loader: data loader of input data
        :return forecasting loss, reconstruction loss, total loss
        """

        self.model.eval()

        forecast_losses = []
        recon_losses = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
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

                forecast_losses.append(forecast_loss.item())
                recon_losses.append(recon_loss.item())

        forecast_losses = np.array(forecast_losses)
        recon_losses = np.array(recon_losses)

        forecast_loss = np.sqrt((forecast_losses ** 2).mean())
        recon_loss = np.sqrt((recon_losses ** 2).mean())

        total_loss = forecast_loss + recon_loss

        return forecast_loss, recon_loss, total_loss

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        # 确保目录存在
        if not os.path.exists(self.dload):
            os.makedirs(self.dload)
        PATH = os.path.join(self.dload, file_name)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)

    def fit_round_robin(self, train_entity_data, window_size, target_dims, val_split, shuffle_dataset):
        """
        实现轮流训练，每个epoch使用不同实体的数据进行训练
        
        :param train_entity_data: List of tuples (entity_name, entity_tensor_data)
        :param window_size: Length of the input sequence
        :param target_dims: dimension of input features to forecast and reconstruct
        :param val_split: 分割验证集的比例
        :param shuffle_dataset: 是否打乱数据集
        """
        print(f"Starting round-robin training with {len(train_entity_data)} entities")
        
        # 初始化损失记录
        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
        }

        # 计算所有实体的初始训练损失
        init_train_losses = []
        num_workers = 2 if os.name == 'nt' else 4
        pin_memory = torch.cuda.is_available()
        
        for entity_name, entity_data in train_entity_data:
            entity_dataset = SlidingWindowDataset(entity_data, window_size, target_dims)
            entity_loader = DataLoader(
                entity_dataset, batch_size=self.batch_size, shuffle=shuffle_dataset,
                num_workers=num_workers, pin_memory=pin_memory
            )
            init_loss = self.evaluate(entity_loader)
            init_train_losses.append(init_loss)
            print(f"Init total train loss for {entity_name}: {init_loss[2]:.5f}")

        print(f"Training model for {self.n_epochs} epochs using round-robin approach..")
        train_start = time.time()
        
        # 预先创建所有实体的 DataLoader 以提高效率
        print("Pre-creating DataLoaders for all entities...")
        entity_loaders = []
        for entity_name, entity_data in train_entity_data:
            entity_dataset = SlidingWindowDataset(entity_data, window_size, target_dims)
            train_loader, val_loader, _ = create_data_loaders(
                entity_dataset, self.batch_size, val_split, shuffle_dataset, test_dataset=None
            )
            entity_loaders.append((entity_name, train_loader, val_loader))
        
        # 轮流训练
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            print(f"[Epoch {epoch + 1}] Starting round-robin training")
            
            # 在每个epoch选择一个实体进行训练
            entity_idx = epoch % len(train_entity_data)
            entity_name, train_loader, val_loader = entity_loaders[entity_idx]
            print(f"[Epoch {epoch + 1}] Training on entity: {entity_name}")
            
            # 训练阶段
            self.model.train()
            forecast_b_losses = []
            recon_b_losses = []
            
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
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
                    loss = forecast_loss + recon_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                forecast_b_losses.append(forecast_loss.item())
                recon_b_losses.append(recon_loss.item())

            forecast_b_losses = np.array(forecast_b_losses)
            recon_b_losses = np.array(recon_b_losses)

            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())

            total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)

            # 验证阶段
            forecast_val_loss, recon_val_loss, total_val_loss = "NA", "NA", "NA"
            if val_loader is not None:
                forecast_val_loss, recon_val_loss, total_val_loss = self.evaluate(val_loader)
                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_recon"].append(recon_val_loss)
                self.losses["val_total"].append(total_val_loss)

                # 如果是最好的模型则保存
                if len(self.losses["val_total"]) == 1 or total_val_loss <= min(self.losses["val_total"]):
                    self.save(f"model.pt")
                    print(f"[Epoch {epoch + 1}] New best model saved with val loss: {total_val_loss}")

            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] Entity: {entity_name} | "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_recon_loss = {recon_val_loss:.5f}, "
                        f"val_total_loss = {total_val_loss:.5f}"
                    )

                s += f" [{epoch_time:.1f}s]"
                print(s)

        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Round-robin training done in {train_time}s.")
