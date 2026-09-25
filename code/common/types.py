"""Shared immutable result containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelRealization:
    bob_gains: np.ndarray
    eve_gains: np.ndarray
    bob_directions: np.ndarray
    eve_directions: np.ndarray


@dataclass(frozen=True)
class MethodResult:
    method: str
    secrecy_rate: float
    tx_indices: tuple[int, ...] | None
    rx_index: int | None
    beamformer: np.ndarray
    elapsed_seconds: float
    iterations: int
    evaluations: int
    continuous_positions: np.ndarray | None = None
    history: tuple[object, ...] = ()
