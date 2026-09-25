"""Bessel-correlated multipath channel from manuscript equations (19)–(22)."""

from __future__ import annotations

import numpy as np
from scipy.special import j0

from code.common.config import SystemConfig
from code.common.types import ChannelRealization


def bessel_correlations(
    rx_positions: np.ndarray,
    wavelength: float,
    reference_position: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate manuscript Eq. (20) from two-dimensional physical distances."""
    positions = np.asarray(rx_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 1:
        raise ValueError("rx_positions must have shape (num_positions, 2)")
    if not np.isfinite(positions).all():
        raise ValueError("rx_positions must be finite")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if reference_position is None:
        reference = positions[0]
    else:
        reference = np.asarray(reference_position, dtype=float).reshape(2)
        if not np.isfinite(reference).all():
            raise ValueError("reference_position must be finite")
    distances = np.linalg.norm(positions - reference, axis=1)
    return np.asarray(j0(2.0 * np.pi * distances / wavelength), dtype=float)


def _sample_directions(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    angles = rng.uniform(-np.pi / 2.0, np.pi / 2.0, shape)
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1)


def sample_channel(
    rng: np.random.Generator,
    config: SystemConfig,
) -> ChannelRealization:
    """Draw one Bessel-correlated Bob/multiple-Eve channel realization."""
    config.validate()
    reference_loss = 10.0 ** (config.path_loss_db / 10.0)
    bob_variance = reference_loss * config.d_bob ** (-config.path_loss_exponent)
    eve_variance = reference_loss * config.d_eve ** (-config.path_loss_exponent)
    num_positions = config.grid_size * config.grid_size

    bob_shape = (num_positions, config.num_bob_paths)
    eve_shape = (config.num_eavesdroppers, config.num_eve_paths)
    bob_gains = np.sqrt(bob_variance / 2.0) * (
        rng.standard_normal(bob_shape) + 1j * rng.standard_normal(bob_shape)
    )
    eve_gains = np.sqrt(eve_variance / 2.0) * (
        rng.standard_normal(eve_shape) + 1j * rng.standard_normal(eve_shape)
    )
    return ChannelRealization(
        bob_gains=bob_gains,
        eve_gains=eve_gains,
        bob_directions=_sample_directions(rng, bob_shape),
        eve_directions=_sample_directions(rng, eve_shape),
    )


def normalized_to_tx_positions(points: np.ndarray, config: SystemConfig) -> np.ndarray:
    """Map normalized offsets to Alice's physical movable region."""
    normalized = np.asarray(points, dtype=float).reshape(-1, 2)
    return normalized * config.area_half_width


def normalized_to_rx_positions(points: np.ndarray, config: SystemConfig) -> np.ndarray:
    """Map normalized offsets to Bob's physical movable region."""
    normalized = np.asarray(points, dtype=float).reshape(-1, 2)
    return normalized * config.area_half_width + np.array([config.d_bob, 0.0])


def _field_response(
    positions: np.ndarray,
    gains: np.ndarray,
    directions: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    phase = (2.0 * np.pi / wavelength) * np.einsum(
        "md,...ld->m...l",
        np.asarray(positions, dtype=float),
        np.asarray(directions, dtype=float),
    )
    return np.einsum("...l,m...l->...m", gains, np.exp(1j * phase))


def _correlate_bob_ports(
    raw_channels: np.ndarray,
    rx_positions: np.ndarray,
    config: SystemConfig,
) -> np.ndarray:
    correlations = np.clip(
        bessel_correlations(rx_positions, config.wavelength), -1.0, 1.0
    )
    if raw_channels.shape[0] != correlations.size:
        raise ValueError("raw Bob channels and receive positions must align")
    independent_scale = np.sqrt(np.maximum(1.0 - correlations**2, 0.0))
    reference = raw_channels[0]
    real = independent_scale[:, None] * raw_channels.real + correlations[:, None] * reference.real
    imag = independent_scale[:, None] * raw_channels.imag + correlations[:, None] * reference.imag
    return real + 1j * imag


def _normalized_rx_position(rx_position: np.ndarray, config: SystemConfig) -> np.ndarray:
    points = np.asarray(rx_position, dtype=float).reshape(-1, 2)
    if points.shape[0] != 1 or not np.isfinite(points).all():
        raise ValueError("exactly one finite receive position is required")
    point = points[0]
    center = np.array([config.d_bob, 0.0])
    normalized = (point - center) / config.area_half_width
    if np.any(np.abs(normalized) > 1.0 + 1e-10):
        raise ValueError("receive position lies outside the movable region")
    return np.clip(normalized, -1.0, 1.0)


def _interpolate_bob_raw(
    raw_channels: np.ndarray,
    rx_position: np.ndarray,
    config: SystemConfig,
) -> np.ndarray:
    """Bilinearly extend the sampled port field for continuous PGD positions."""
    expected = config.grid_size * config.grid_size
    if raw_channels.shape[0] != expected:
        raise ValueError("raw Bob channels do not cover the configured receive grid")
    normalized = _normalized_rx_position(rx_position, config)
    coordinates = (normalized + 1.0) * (config.grid_size - 1) / 2.0
    lower = np.floor(coordinates).astype(int)
    upper = np.minimum(lower + 1, config.grid_size - 1)
    weight = coordinates - lower
    grid = raw_channels.reshape(config.grid_size, config.grid_size, -1)
    low_low = grid[lower[0], lower[1]]
    high_low = grid[upper[0], lower[1]]
    low_high = grid[lower[0], upper[1]]
    high_high = grid[upper[0], upper[1]]
    along_x_low = (1.0 - weight[0]) * low_low + weight[0] * high_low
    along_x_high = (1.0 - weight[0]) * low_high + weight[0] * high_high
    return (1.0 - weight[1]) * along_x_low + weight[1] * along_x_high


def _continuous_correlated_bob(
    raw_channels: np.ndarray,
    rx_position: np.ndarray,
    config: SystemConfig,
) -> np.ndarray:
    point = np.asarray(rx_position, dtype=float).reshape(-1, 2)
    raw_value = _interpolate_bob_raw(raw_channels, point, config)
    reference_position = normalized_to_rx_positions(
        np.array([[-1.0, -1.0]]), config
    )[0]
    correlation = float(
        bessel_correlations(point, config.wavelength, reference_position)[0]
    )
    correlation = float(np.clip(correlation, -1.0, 1.0))
    independent_scale = float(np.sqrt(max(1.0 - correlation**2, 0.0)))
    reference = raw_channels[0]
    return independent_scale * raw_value + correlation * reference


def effective_channels(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    realization: ChannelRealization,
    config: SystemConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Bob and colluding-Eve channels for physical positions."""
    tx_offsets = np.asarray(tx_positions, dtype=float).reshape(-1, 2)
    if tx_offsets.shape[0] < 1:
        raise ValueError("at least one transmit position is required")

    bob_raw = _field_response(
        tx_offsets,
        realization.bob_gains,
        realization.bob_directions,
        config.wavelength,
    )
    h_b = _continuous_correlated_bob(
        bob_raw, np.asarray(rx_positions, dtype=float), config
    ).reshape(1, -1)
    h_e = _field_response(
        tx_offsets,
        realization.eve_gains,
        realization.eve_directions,
        config.wavelength,
    )
    return h_b, h_e


def discrete_channel_matrices(
    realization: ChannelRealization,
    config: SystemConfig,
    grid_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the complete discrete Bob matrix and Eve matrix once."""
    normalized = np.asarray(grid_points, dtype=float)
    expected = config.grid_size * config.grid_size
    if normalized.shape != (expected, 2):
        raise ValueError("grid_points must cover the configured square grid")
    tx_positions = normalized_to_tx_positions(normalized, config)
    rx_positions = normalized_to_rx_positions(normalized, config)
    bob_raw = _field_response(
        tx_positions,
        realization.bob_gains,
        realization.bob_directions,
        config.wavelength,
    )
    h_b_full = _correlate_bob_ports(bob_raw, rx_positions, config)
    h_e_full = _field_response(
        tx_positions,
        realization.eve_gains,
        realization.eve_directions,
        config.wavelength,
    )
    return h_b_full, h_e_full
