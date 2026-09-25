"""Shared immutable inputs and counted objective evaluation for discrete solvers."""

from __future__ import annotations

import numpy as np

from code.common.beamforming import solve_grq
from code.common.config import SystemConfig
from code.common.geometry import is_feasible_indices, physical_grid


class DiscreteProblem:
    def __init__(
        self,
        h_b_full: np.ndarray,
        h_e_full: np.ndarray,
        config: SystemConfig,
    ) -> None:
        config.validate()
        expected = config.grid_size * config.grid_size
        bob = np.array(h_b_full, dtype=complex, copy=True)
        eve = np.array(h_e_full, dtype=complex, copy=True)
        if bob.shape != (expected, expected):
            raise ValueError("h_b_full must have shape (num_positions, num_positions)")
        if eve.shape != (config.num_eavesdroppers, expected):
            raise ValueError("h_e_full must have shape (num_eavesdroppers, num_positions)")
        if not np.isfinite(bob).all() or not np.isfinite(eve).all():
            raise ValueError("channel matrices must be finite")
        bob.setflags(write=False)
        eve.setflags(write=False)
        self.h_b_full = bob
        self.h_e_full = eve
        self.config = config
        self.points = physical_grid(config)
        self._evaluations = 0

    @property
    def num_positions(self) -> int:
        return self.h_b_full.shape[0]

    def evaluate(
        self,
        tx_indices: tuple[int, ...],
        rx_index: int,
    ) -> tuple[np.ndarray, float]:
        selection = tuple(int(index) for index in tx_indices)
        if not self.is_feasible_partial(selection):
            raise ValueError("transmit-port selection is infeasible")
        if not 0 <= int(rx_index) < self.num_positions:
            raise ValueError("rx_index is outside the port grid")
        self._evaluations += 1
        h_b = self.h_b_full[np.ix_([int(rx_index)], selection)]
        h_e = self.h_e_full[:, selection]
        return solve_grq(h_b, h_e, self.config)

    def is_feasible(self, tx_indices: tuple[int, ...] | None) -> bool:
        if tx_indices is None or len(tx_indices) != self.config.m_t:
            return False
        return is_feasible_indices(tx_indices, self.points, self.config.d_min)

    def is_feasible_partial(self, tx_indices: tuple[int, ...] | None) -> bool:
        if tx_indices is None or not 1 <= len(tx_indices) <= self.config.m_t:
            return False
        return is_feasible_indices(tx_indices, self.points, self.config.d_min)

    @property
    def evaluations(self) -> int:
        return self._evaluations
