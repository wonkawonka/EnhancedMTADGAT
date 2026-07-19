"""Modern PyTorch compatibility implementations for the NC battery baselines.

Architectures and public hyperparameters follow the code released with Zhang
et al., "Realistic fault detection of Li-ion battery via dynamical deep
learning".  The implementations live in this project so experiments do not
depend on cloning an additional repository or on its Python 3.6-era APIs.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class NCDynamicVAE(nn.Module):
    """DyAD DynamicVAE with the public brand-specific task definitions."""

    BRAND_CONFIG = {
        1: {
            "hidden_size": 128,
            "latent_size": 8,
            "num_layers": 2,
            "decoder_dimension": 2,
            "output_dimension": 5,
            "encoder_indices": (2, 1, 6, 3, 5, 4, 0),
            "noise_scale": 1.0,
        },
        2: {
            "hidden_size": 1024,
            "latent_size": 24,
            "num_layers": 1,
            "decoder_dimension": 4,
            "output_dimension": 3,
            "encoder_indices": (2, 1, 6, 3, 4, 0, 5),
            "noise_scale": 0.01,
        },
        3: {
            "hidden_size": 256,
            "latent_size": 16,
            "num_layers": 1,
            "decoder_dimension": 2,
            "output_dimension": 5,
            "encoder_indices": (2, 1, 6, 3, 5, 4, 0),
            "noise_scale": 0.01,
        },
    }

    def __init__(self, brand: int):
        super().__init__()
        if brand not in self.BRAND_CONFIG:
            raise ValueError("brand must be 1, 2, or 3")
        config = self.BRAND_CONFIG[brand]
        self.brand = int(brand)
        self.hidden_size = int(config["hidden_size"])
        self.latent_size = int(config["latent_size"])
        self.num_layers = int(config["num_layers"])
        self.decoder_dimension = int(config["decoder_dimension"])
        self.output_dimension = int(config["output_dimension"])
        self.encoder_indices = tuple(config["encoder_indices"])
        self.noise_scale = float(config["noise_scale"])
        self.hidden_factor = 2 * self.num_layers

        self.encoder_rnn = nn.GRU(
            7,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.decoder_rnn = nn.GRU(
            self.decoder_dimension,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )
        hidden_dimension = self.hidden_size * self.hidden_factor
        self.hidden2mean = nn.Linear(hidden_dimension, self.latent_size)
        self.hidden2log_v = nn.Linear(hidden_dimension, self.latent_size)
        self.latent2hidden = nn.Linear(self.latent_size, hidden_dimension)
        self.outputs2embedding = nn.Linear(self.hidden_size * 2, self.output_dimension)
        self.mean2mileage = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
        )

    def _ordered(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, :, self.encoder_indices]

    def target(self, values: torch.Tensor) -> torch.Tensor:
        return self._ordered(values)[:, :, self.decoder_dimension :]

    def forward(self, values: torch.Tensor):
        ordered = self._ordered(values)
        _, hidden = self.encoder_rnn(ordered)
        batch_size = values.shape[0]
        hidden_flat = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        mean = self.hidden2mean(hidden_flat)
        log_v = self.hidden2log_v(hidden_flat)
        std = torch.exp(0.5 * log_v)
        latent = mean
        if self.training:
            latent = mean + torch.randn_like(std) * std * self.noise_scale

        decoder_hidden = self.latent2hidden(latent)
        decoder_hidden = decoder_hidden.view(batch_size, self.hidden_factor, self.hidden_size)
        decoder_hidden = decoder_hidden.transpose(0, 1).contiguous()
        decoder_input = ordered[:, :, : self.decoder_dimension]
        decoded, _ = self.decoder_rnn(decoder_input, decoder_hidden)
        response = self.outputs2embedding(decoded)
        mileage = self.mean2mileage(mean).squeeze(-1)
        return response, mean, log_v, mileage


class NCGDN(nn.Module):
    """Battery GDN using the public learned top-k graph-attention structure."""

    def __init__(
        self,
        node_count: int = 6,
        window_size: int = 32,
        hidden_size: int = 64,
        topk: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.node_count = int(node_count)
        self.topk = min(int(topk), self.node_count)
        self.hidden_size = int(hidden_size)
        self.embedding = nn.Embedding(self.node_count, self.hidden_size)
        self.input_projection = nn.Linear(int(window_size), self.hidden_size, bias=False)
        self.att_data_i = nn.Parameter(torch.empty(self.hidden_size))
        self.att_data_j = nn.Parameter(torch.empty(self.hidden_size))
        self.att_embedding_i = nn.Parameter(torch.zeros(self.hidden_size))
        self.att_embedding_j = nn.Parameter(torch.zeros(self.hidden_size))
        self.message_norm = nn.BatchNorm1d(self.hidden_size)
        self.output_norm = nn.BatchNorm1d(self.hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(self.hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.att_data_i.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_data_j.unsqueeze(0))
        nn.init.zeros_(self.att_embedding_i)
        nn.init.zeros_(self.att_embedding_j)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # Input is [batch, time, node], matching this project's window helper.
        node_windows = windows.transpose(1, 2)
        projected = self.input_projection(node_windows)
        embeddings = self.embedding.weight

        normalized = F.normalize(embeddings, p=2, dim=-1)
        similarity = normalized @ normalized.transpose(0, 1)
        neighbours = torch.topk(similarity, self.topk, dim=-1).indices

        source_data = projected[:, neighbours, :]
        target_data = projected[:, :, None, :].expand_as(source_data)
        source_embedding = embeddings[neighbours][None, :, :, :]
        target_embedding = embeddings[None, :, None, :].expand_as(source_embedding)

        logits = (target_data * self.att_data_i).sum(-1)
        logits = logits + (source_data * self.att_data_j).sum(-1)
        logits = logits + (target_embedding * self.att_embedding_i).sum(-1)
        logits = logits + (source_embedding * self.att_embedding_j).sum(-1)
        attention = torch.softmax(F.leaky_relu(logits, negative_slope=0.2), dim=-1)
        aggregated = torch.sum(source_data * attention.unsqueeze(-1), dim=2)

        batch_size = aggregated.shape[0]
        aggregated = self.message_norm(aggregated.reshape(-1, self.hidden_size))
        aggregated = F.relu(aggregated).view(batch_size, self.node_count, self.hidden_size)
        output = aggregated * embeddings.unsqueeze(0)
        output = self.output_norm(output.reshape(-1, self.hidden_size))
        output = F.relu(output).view(batch_size, self.node_count, self.hidden_size)
        return self.output(self.dropout(output)).squeeze(-1)


class NCLSTMAutoEncoder(nn.Module):
    """Public LSTM-AD recurrent encoder and autoregressive decoder."""

    def __init__(self, n_features: int = 6, latent_size: int = 32):
        super().__init__()
        self.n_features = int(n_features)
        self.encoder = nn.LSTM(self.n_features, int(latent_size), batch_first=True)
        self.decoder_cell = nn.LSTMCell(self.n_features, int(latent_size))
        self.output = nn.Linear(int(latent_size), self.n_features)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(values)
        hidden = hidden.squeeze(0)
        cell = cell.squeeze(0)
        current = self.output(hidden)
        decoded = []
        for _ in range(values.shape[1]):
            hidden, cell = self.decoder_cell(current, (hidden, cell))
            current = self.output(hidden)
            decoded.append(current)
        return torch.flip(torch.stack(decoded, dim=1), dims=(1,))
