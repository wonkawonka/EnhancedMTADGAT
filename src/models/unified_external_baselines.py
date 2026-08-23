"""Model-side compatibility adapters for the unified external protocol."""

from __future__ import annotations

import math
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

from src.models.nc_official_baselines import NCGDN, NCLSTMAutoEncoder
from src.runners.run_public_baselines import TwoAutoEncoder
from src.project_paths import PROJECT_ROOT


class FlatAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        hidden_dim = min(max(16, hidden_dim), input_dim)
        latent_dim = min(max(8, latent_dim), hidden_dim)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def forward(self, values):
        flat = values.flatten(1)
        return self.decoder(self.encoder(flat)).view_as(values)


class DeepSVDDEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        hidden_dim = min(max(16, hidden_dim), input_dim)
        latent_dim = min(max(8, latent_dim), hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim, bias=False),
        )

    def forward(self, values):
        return self.network(values.flatten(1))


class DeepSVDDAutoEncoder(nn.Module):
    """Autoencoder pretraining stage used by the public Deep-SVDD protocol."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = DeepSVDDEncoder(input_dim, hidden_dim, latent_dim)
        hidden_dim = min(max(16, hidden_dim), input_dim)
        latent_dim = min(max(8, latent_dim), hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim, bias=False),
        )

    def forward(self, values):
        return self.decoder(self.encoder(values)).view_as(values)


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension))
        encoding = torch.zeros(max_len, 1, dimension)
        encoding[:, 0, 0::2] = torch.sin(positions * div)
        encoding[:, 0, 1::2] = torch.cos(positions * div)
        self.register_buffer("encoding", encoding)

    def forward(self, values):
        return self.dropout(values + self.encoding[: values.size(0)])


class TranADCompat(nn.Module):
    """Dataset-agnostic port of the public TranAD self-conditioning architecture."""

    def __init__(self, features: int, window: int):
        super().__init__()
        dimension = 2 * features
        self.features = features
        self.positional = PositionalEncoding(dimension, 0.1, window)
        encoder_layer = TransformerEncoderLayer(d_model=dimension, nhead=features, dim_feedforward=16, dropout=0.1)
        self.encoder = TransformerEncoder(encoder_layer, 1)
        decoder1 = TransformerDecoderLayer(d_model=dimension, nhead=features, dim_feedforward=16, dropout=0.1)
        decoder2 = TransformerDecoderLayer(d_model=dimension, nhead=features, dim_feedforward=16, dropout=0.1)
        self.decoder1 = TransformerDecoder(decoder1, 1)
        self.decoder2 = TransformerDecoder(decoder2, 1)
        self.output = nn.Sequential(nn.Linear(dimension, features), nn.Sigmoid())

    def _encode(self, source, condition, target):
        source = self.positional(torch.cat((source, condition), dim=2) * math.sqrt(self.features))
        memory = self.encoder(source)
        return target.repeat(1, 1, 2), memory

    def forward(self, values):
        source = values.transpose(0, 1)
        target = source[-1:].clone()
        condition = torch.zeros_like(source)
        first = self.output(self.decoder1(*self._encode(source, condition, target)))
        condition = (first - source) ** 2
        second = self.output(self.decoder2(*self._encode(source, condition, target)))
        return first.transpose(0, 1), second.transpose(0, 1)


def build_reconstruction_model(method: str, features: int, window: int, hidden_dim: int, latent_dim: int):
    input_dim = int(features * window)
    if method == "auto_encoder":
        return FlatAutoEncoder(input_dim, hidden_dim, latent_dim)
    if method == "lstm_ad":
        return NCLSTMAutoEncoder(n_features=features, latent_size=latent_dim)
    if method == "gdn":
        return NCGDN(node_count=features, window_size=window, hidden_size=hidden_dim, topk=min(5, features))
    if method == "usad":
        return nn.ModuleList((TwoAutoEncoder(input_dim, hidden_dim, latent_dim), TwoAutoEncoder(input_dim, hidden_dim, latent_dim)))
    if method == "tranad":
        return TranADCompat(features, window)
    raise ValueError(f"No reconstruction model registered for {method}")


def _load_relative_module(package_name: str, directory: Path, module_name: str):
    package = sys.modules.get(package_name)
    if package is None:
        init_path = directory / "__init__.py"
        if init_path.exists():
            spec = importlib.util.spec_from_file_location(
                package_name, init_path, submodule_search_locations=[str(directory)]
            )
            package = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = package
            assert spec.loader is not None
            spec.loader.exec_module(package)
        else:
            package = types.ModuleType(package_name)
            package.__path__ = [str(directory)]
            sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.{module_name}")


class GANFCompat(nn.Module):
    def __init__(self, model, features: int):
        super().__init__()
        self.model = model
        self.adjacency_logits = nn.Parameter(torch.zeros(features, features))
        self.register_buffer("identity", torch.eye(features))

    @property
    def adjacency(self):
        return torch.sigmoid(self.adjacency_logits) * (1.0 - self.identity)

    def forward(self, values):
        # Public GANF expects [batch, sensors, time, scalar feature].
        return self.model(values.transpose(1, 2).unsqueeze(-1), self.adjacency)

    def score(self, values):
        return -self.model.test(values.transpose(1, 2).unsqueeze(-1), self.adjacency)


def build_reference_model(method: str, features: int, window: int):
    if method == "anomaly_transformer":
        directory = PROJECT_ROOT / "external_baselines" / "Anomaly-Transformer" / "model"
        module = _load_relative_module("_unified_anomaly_transformer", directory, "AnomalyTransformer")
        return module.AnomalyTransformer(
            win_size=window, enc_in=features, c_out=features, d_model=512,
            n_heads=8, e_layers=3, d_ff=512, output_attention=True,
        )
    if method == "dcdetector":
        directory = PROJECT_ROOT / "external_baselines" / "DCdetector" / "model"
        module = _load_relative_module("_unified_dcdetector", directory, "DCdetector")
        patch_sizes = [size for size in (5, 10, 20) if window % size == 0]
        return module.DCdetector(
            win_size=window, enc_in=features, c_out=features, channel=features,
            d_model=256, n_heads=1, e_layers=3, patch_size=patch_sizes,
        )
    if method == "ganf":
        root = PROJECT_ROOT / "external_baselines" / "GANF"
        sys.path.insert(0, str(root))
        try:
            module = importlib.import_module("models.GANF")
            model = module.GANF(1, 1, 32, 1, dropout=0.1, batch_norm=False)
        finally:
            if sys.path[0] == str(root):
                sys.path.pop(0)
        return GANFCompat(model, features)
    raise ValueError(f"No public reference model registered for {method}")
