"""Exhaustive discrete antenna-port selection baseline."""

from __future__ import annotations

from itertools import combinations
from time import perf_counter

import numpy as np

from code.common.types import MethodResult
from code.solvers.context import DiscreteProblem


def solve_exhaustive(problem: DiscreteProblem) -> MethodResult:
    start = perf_counter()
    evaluations_before = problem.evaluations
    best_rate = -np.inf
    best_tx: tuple[int, ...] | None = None
    best_rx: int | None = None
    best_beamformer = np.empty((0, 1), dtype=complex)
    combinations_checked = 0

    for tx_indices in combinations(range(problem.num_positions), problem.config.m_t):
        if not problem.is_feasible(tx_indices):
            continue
        for rx_index in range(problem.num_positions):
            combinations_checked += 1
            beamformer, rate = problem.evaluate(tx_indices, rx_index)
            if rate > best_rate:
                best_rate = rate
                best_tx = tx_indices
                best_rx = rx_index
                best_beamformer = beamformer

    if best_tx is None or best_rx is None:
        raise RuntimeError("exhaustive search found no feasible candidate")
    return MethodResult(
        method="Exhaustive",
        secrecy_rate=float(best_rate),
        tx_indices=best_tx,
        rx_index=best_rx,
        beamformer=best_beamformer,
        elapsed_seconds=perf_counter() - start,
        iterations=combinations_checked,
        evaluations=problem.evaluations - evaluations_before,
    )
