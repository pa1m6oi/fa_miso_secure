"""Secrecy-rate objectives and compact experiment summaries."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from code.common.beamforming import solve_grq
from code.common.channel import (
    effective_channels,
    normalized_to_rx_positions,
    normalized_to_tx_positions,
)
from code.common.config import SystemConfig
from code.common.types import ChannelRealization


def secrecy_rate(
    h_b: np.ndarray,
    h_e: np.ndarray,
    beamformer: np.ndarray,
    config: SystemConfig,
) -> float:
    """Evaluate the nonnegative colluding-Eve secrecy rate."""
    bob = np.asarray(h_b, dtype=complex).reshape(1, -1)
    eve = np.asarray(h_e, dtype=complex)
    if eve.ndim == 1:
        eve = eve.reshape(1, -1)
    w = np.asarray(beamformer, dtype=complex).reshape(-1, 1)
    if bob.shape[1] != w.shape[0] or eve.ndim != 2 or eve.shape[1] != w.shape[0]:
        raise ValueError("channel and beamformer dimensions do not agree")
    bob_power = float(np.sum(np.abs(bob @ w) ** 2).item())
    eve_power = float(np.sum(np.abs(eve @ w) ** 2).item())
    value = math.log2(1.0 + bob_power / config.noise_power) - math.log2(
        1.0 + eve_power / config.noise_power
    )
    return float(max(value, 0.0))


def objective_from_positions(
    x: np.ndarray,
    realization: ChannelRealization,
    config: SystemConfig,
) -> float:
    """Evaluate normalized continuous TX/RX position variables."""
    values = np.asarray(x, dtype=float).reshape(-1)
    expected = 2 * config.m_t + 2
    if values.size != expected:
        raise ValueError(f"continuous initialization must contain {expected} values")
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1.0 + 1e-12):
        raise ValueError("continuous positions must be finite and lie in [-1, 1]")
    tx_normalized = values[: 2 * config.m_t].reshape(config.m_t, 2)
    rx_normalized = values[2 * config.m_t :].reshape(1, 2)
    h_b, h_e = effective_channels(
        normalized_to_tx_positions(tx_normalized, config),
        normalized_to_rx_positions(rx_normalized, config),
        realization,
        config,
    )
    _, rate = solve_grq(h_b, h_e, config)
    return rate


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    """Return count, arithmetic mean, and population standard deviation."""
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        raise ValueError("values must be nonempty")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }
