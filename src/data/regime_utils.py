"""Operating-regime labels used for representation analysis, not fault labels."""

from __future__ import annotations

import numpy as np


NASA_REGIME_NAMES = {
    0: "rest",
    1: "charge",
    2: "discharge",
}

BMS_REGIME_NAMES = {
    0: "idle",
    1: "frequency_regulation",
}


def derive_bms_regime_labels(
    data: np.ndarray,
    current_index: int,
    window: int = 60,
    active_threshold: float = 0.05,
    switch_rate_threshold: float = 0.02,
) -> np.ndarray:
    """Infer idle/frequency-regulation intervals from current activity.

    The thresholds operate on current normalized by its robust 95th percentile,
    so they remain usable for raw and globally scaled BMS current channels.
    """
    current = np.asarray(data, dtype=np.float32)[:, current_index]
    scale = float(np.quantile(np.abs(current), 0.95))
    normalized = current / max(scale, 1e-6)
    active = (np.abs(normalized) >= active_threshold).astype(np.float32)

    signs = np.sign(normalized)
    switches = np.zeros_like(active)
    switches[1:] = ((signs[1:] * signs[:-1]) < 0).astype(np.float32)

    kernel = np.ones(max(1, int(window)), dtype=np.float32)
    active_rate = np.convolve(active, kernel, mode="same") / kernel.size
    switch_rate = np.convolve(switches, kernel, mode="same") / kernel.size
    return ((active_rate >= 0.10) | (switch_rate >= switch_rate_threshold)).astype(np.int64)


def derive_fine_current_state(
    data: np.ndarray,
    current_index: int,
    zero_threshold: float = 0.05,
) -> np.ndarray:
    """Return 0=rest, 1=charge and 2=discharge from current direction."""
    current = np.asarray(data, dtype=np.float32)[:, current_index]
    scale = float(np.quantile(np.abs(current), 0.95))
    normalized = current / max(scale, 1e-6)
    labels = np.zeros(current.shape[0], dtype=np.int64)
    labels[normalized > zero_threshold] = 1
    labels[normalized < -zero_threshold] = 2
    return labels
