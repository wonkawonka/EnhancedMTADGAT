"""Shared electrical/thermal response descriptors for training and scoring."""

from __future__ import annotations

import numpy as np
import torch


DEFAULT_PHYSICAL_RESPONSE_TERMS = (
    "voltage_rate",
    "temperature_rate",
    "charge_flow",
    "voltage_spread",
    "temperature_spread",
    "soc_current_coupling",
)


def resolve_physical_response_terms(config) -> tuple[str, ...]:
    """Return validated response terms while preserving the requested order."""
    if not config:
        return DEFAULT_PHYSICAL_RESPONSE_TERMS
    configured = config.get("response_terms", DEFAULT_PHYSICAL_RESPONSE_TERMS)
    if isinstance(configured, str):
        configured = configured.split(",")
    requested = [str(term).strip() for term in configured if str(term).strip()]
    unknown = sorted(set(requested) - set(DEFAULT_PHYSICAL_RESPONSE_TERMS))
    if unknown:
        raise ValueError(f"Unsupported physical response terms: {unknown}")
    return tuple(dict.fromkeys(requested))


def _torch_channel(values, index):
    if index is None or index < 0 or index >= values.size(2):
        return None
    return values[:, :, index:index + 1]


def _torch_normalize(values, eps=1e-6):
    scale = values.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    return values / scale


def _torch_normalized_cumsum(values, eps=1e-6):
    cumulative = torch.cumsum(values, dim=1)
    scale = cumulative.abs().amax(dim=1, keepdim=True).clamp_min(eps)
    return torch.clamp(cumulative / scale, -1.0, 1.0)


def compute_torch_physical_response_errors(actual, reconstructed, config):
    """Compute differentiable per-term response errors for [batch, time, feature]."""
    enabled = set(resolve_physical_response_terms(config))
    channels = {
        key: (_torch_channel(actual, config.get(key)), _torch_channel(reconstructed, config.get(key)))
        for key in (
            "voltage_index",
            "temperature_index",
            "current_index",
            "soc_index",
            "voltage_max_index",
            "voltage_min_index",
            "temperature_max_index",
            "temperature_min_index",
        )
    }
    errors = {}

    voltage, voltage_hat = channels["voltage_index"]
    if "voltage_rate" in enabled and voltage is not None and voltage_hat is not None and actual.size(1) >= 2:
        errors["voltage_rate"] = torch.mean(
            torch.abs(torch.diff(voltage_hat, dim=1) - torch.diff(voltage, dim=1))
        )

    temperature, temperature_hat = channels["temperature_index"]
    if (
        "temperature_rate" in enabled
        and temperature is not None
        and temperature_hat is not None
        and actual.size(1) >= 2
    ):
        errors["temperature_rate"] = torch.mean(
            torch.abs(torch.diff(temperature_hat, dim=1) - torch.diff(temperature, dim=1))
        )

    current, current_hat = channels["current_index"]
    if "charge_flow" in enabled and current is not None and current_hat is not None:
        errors["charge_flow"] = torch.mean(
            torch.abs(_torch_normalized_cumsum(current_hat) - _torch_normalized_cumsum(current))
        )

    voltage_max, voltage_max_hat = channels["voltage_max_index"]
    voltage_min, voltage_min_hat = channels["voltage_min_index"]
    if (
        "voltage_spread" in enabled
        and all(value is not None for value in (voltage_max, voltage_min, voltage_max_hat, voltage_min_hat))
    ):
        errors["voltage_spread"] = torch.mean(
            torch.abs((voltage_max_hat - voltage_min_hat) - (voltage_max - voltage_min))
        )

    temperature_max, temperature_max_hat = channels["temperature_max_index"]
    temperature_min, temperature_min_hat = channels["temperature_min_index"]
    if (
        "temperature_spread" in enabled
        and all(
            value is not None
            for value in (temperature_max, temperature_min, temperature_max_hat, temperature_min_hat)
        )
    ):
        errors["temperature_spread"] = torch.mean(
            torch.abs((temperature_max_hat - temperature_min_hat) - (temperature_max - temperature_min))
        )

    soc, soc_hat = channels["soc_index"]
    if (
        "soc_current_coupling" in enabled
        and all(value is not None for value in (soc, soc_hat, current, current_hat))
        and actual.size(1) >= 2
    ):
        response = _torch_normalize(torch.diff(soc, dim=1)) - _torch_normalize(current[:, 1:])
        response_hat = _torch_normalize(torch.diff(soc_hat, dim=1)) - _torch_normalize(current_hat[:, 1:])
        errors["soc_current_coupling"] = torch.mean(torch.abs(response_hat - response))

    return errors


def _numpy_channel(values, index):
    if index is None or index < 0 or index >= values.shape[1]:
        return None
    return values[:, index]


def _numpy_normalize(values, eps=1e-6):
    return values / (np.mean(np.abs(values)) + eps)


def _numpy_normalized_cumsum(values, eps=1e-6):
    cumulative = np.cumsum(values)
    return np.clip(cumulative / (np.max(np.abs(cumulative)) + eps), -1.0, 1.0)


def _pad_first(values, length):
    if length == 0:
        return np.empty(0, dtype=np.float32)
    return np.pad(values, (1, 0), mode="constant")[:length]


def compute_numpy_physical_response_errors(actual, reconstructed, config):
    """Compute point-aligned response errors within one continuous sequence."""
    actual = np.asarray(actual)
    reconstructed = np.asarray(reconstructed)
    if actual.ndim != 2 or reconstructed.shape != actual.shape:
        raise ValueError("Physical response arrays must have matching [time, feature] shapes")

    enabled = set(resolve_physical_response_terms(config))
    channels = {
        key: (_numpy_channel(actual, config.get(key)), _numpy_channel(reconstructed, config.get(key)))
        for key in (
            "voltage_index",
            "temperature_index",
            "current_index",
            "soc_index",
            "voltage_max_index",
            "voltage_min_index",
            "temperature_max_index",
            "temperature_min_index",
        )
    }
    errors = {}
    length = len(actual)

    voltage, voltage_hat = channels["voltage_index"]
    if "voltage_rate" in enabled and voltage is not None and voltage_hat is not None and length >= 2:
        errors["voltage_rate"] = _pad_first(np.abs(np.diff(voltage_hat) - np.diff(voltage)), length)

    temperature, temperature_hat = channels["temperature_index"]
    if "temperature_rate" in enabled and temperature is not None and temperature_hat is not None and length >= 2:
        errors["temperature_rate"] = _pad_first(
            np.abs(np.diff(temperature_hat) - np.diff(temperature)), length
        )

    current, current_hat = channels["current_index"]
    if "charge_flow" in enabled and current is not None and current_hat is not None:
        errors["charge_flow"] = np.abs(
            _numpy_normalized_cumsum(current_hat) - _numpy_normalized_cumsum(current)
        )

    voltage_max, voltage_max_hat = channels["voltage_max_index"]
    voltage_min, voltage_min_hat = channels["voltage_min_index"]
    if (
        "voltage_spread" in enabled
        and all(value is not None for value in (voltage_max, voltage_min, voltage_max_hat, voltage_min_hat))
    ):
        errors["voltage_spread"] = np.abs(
            (voltage_max_hat - voltage_min_hat) - (voltage_max - voltage_min)
        )

    temperature_max, temperature_max_hat = channels["temperature_max_index"]
    temperature_min, temperature_min_hat = channels["temperature_min_index"]
    if (
        "temperature_spread" in enabled
        and all(
            value is not None
            for value in (temperature_max, temperature_min, temperature_max_hat, temperature_min_hat)
        )
    ):
        errors["temperature_spread"] = np.abs(
            (temperature_max_hat - temperature_min_hat) - (temperature_max - temperature_min)
        )

    soc, soc_hat = channels["soc_index"]
    if (
        "soc_current_coupling" in enabled
        and all(value is not None for value in (soc, soc_hat, current, current_hat))
        and length >= 2
    ):
        response = _numpy_normalize(np.diff(soc)) - _numpy_normalize(current[1:])
        response_hat = _numpy_normalize(np.diff(soc_hat)) - _numpy_normalize(current_hat[1:])
        errors["soc_current_coupling"] = _pad_first(np.abs(response_hat - response), length)

    return {name: np.asarray(value, dtype=np.float32) for name, value in errors.items()}
