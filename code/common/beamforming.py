"""Closed-form generalized Rayleigh-quotient beamforming."""

from __future__ import annotations

import math

import numpy as np

from code.common.config import SystemConfig


def _validated_channels(h_b: np.ndarray, h_e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bob = np.asarray(h_b, dtype=complex)
    eve = np.asarray(h_e, dtype=complex)
    if bob.size == 0:
        raise ValueError("h_b must be nonempty")
    bob = bob.reshape(1, -1)
    if eve.ndim == 1:
        eve = eve.reshape(1, -1)
    if eve.ndim != 2 or eve.shape[1] != bob.shape[1]:
        raise ValueError("h_e must have the same number of columns as h_b")
    return bob, eve


def solve_grq(
    h_b: np.ndarray,
    h_e: np.ndarray,
    config: SystemConfig,
) -> tuple[np.ndarray, float]:
    """Return the power-normalized GRQ beamformer and secrecy rate."""
    bob, eve = _validated_channels(h_b, h_e)
    num_transmit = bob.shape[1]
    a_matrix = bob.conj().T @ bob
    b_matrix = eve.conj().T @ eve
    regularizer = config.noise_power / config.p_max
    identity = np.eye(num_transmit, dtype=complex)

    numerator = a_matrix + regularizer * identity
    denominator = b_matrix + regularizer * identity
    cholesky = np.linalg.cholesky(denominator)
    inverse_cholesky = np.linalg.solve(cholesky, identity)
    transformed = inverse_cholesky @ numerator @ inverse_cholesky.conj().T
    transformed = 0.5 * (transformed + transformed.conj().T)

    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    dominant = int(np.argmax(np.real(eigenvalues)))
    direction = inverse_cholesky.conj().T @ eigenvectors[:, dominant].reshape(-1, 1)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        direction = np.ones((num_transmit, 1), dtype=complex) / np.sqrt(num_transmit)
    else:
        direction /= norm
    beamformer = np.sqrt(config.p_max) * direction

    bob_power = float(np.real((beamformer.conj().T @ a_matrix @ beamformer).item()))
    eve_power = float(np.real((beamformer.conj().T @ b_matrix @ beamformer).item()))
    bob_term = max(1.0 + bob_power / config.noise_power, np.finfo(float).tiny)
    eve_term = max(1.0 + eve_power / config.noise_power, np.finfo(float).tiny)
    rate = max(math.log2(bob_term) - math.log2(eve_term), 0.0)
    return beamformer, float(rate)
