"""Seeded feasible random-search baseline."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from code.common.types import MethodResult
from code.solvers.context import DiscreteProblem


def solve_random(
    problem: DiscreteProblem,
    rng: np.random.Generator,
    trials: int | None = None,
) -> MethodResult:
    start = perf_counter()
    total_trials = trials if trials is not None else problem.config.m_t * problem.num_positions
    evaluations_before = problem.evaluations
    best_rate = -np.inf
    best_tx: tuple[int, ...] | None = None
    best_rx: int | None = None
    best_beamformer = np.empty((0, 1), dtype=complex)

    def feasible_candidates(chosen: list[int]) -> list[int]:
        return [
            candidate
            for candidate in range(problem.num_positions)
            if candidate not in chosen
            and all(
                np.linalg.norm(problem.points[candidate] - problem.points[previous])
                + 1e-12
                >= problem.config.d_min
                for previous in chosen
            )
        ]

    for _ in range(max(int(total_trials), 0)):
        chosen: list[int] = []
        for transmit_index in range(problem.config.m_t):
            if transmit_index == 0:
                chosen.append(int(rng.integers(problem.num_positions)))
                continue
            feasible = feasible_candidates(chosen)
            if not feasible:
                chosen = []
                break
            chosen.append(int(rng.choice(feasible)))
        if len(chosen) != problem.config.m_t:
            continue
        tx_indices = tuple(chosen)
        rx_index = int(rng.integers(problem.num_positions))
        beamformer, rate = problem.evaluate(tx_indices, rx_index)
        if rate > best_rate:
            best_rate = rate
            best_tx = tx_indices
            best_rx = rx_index
            best_beamformer = beamformer

    if best_tx is None or best_rx is None:
        raise RuntimeError("random search found no feasible candidate")
    return MethodResult(
        method="Random",
        secrecy_rate=float(best_rate),
        tx_indices=best_tx,
        rx_index=best_rx,
        beamformer=best_beamformer,
        elapsed_seconds=perf_counter() - start,
        iterations=max(int(total_trials), 0),
        evaluations=problem.evaluations - evaluations_before,
    )
