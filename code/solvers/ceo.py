"""Cross-entropy optimization for joint transmit/receive port selection."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from code.common.types import MethodResult
from code.solvers.context import DiscreteProblem


def _feasible_mask(problem: DiscreteProblem, chosen: list[int]) -> np.ndarray:
    mask = np.ones(problem.num_positions, dtype=bool)
    for candidate in range(problem.num_positions):
        if candidate in chosen:
            mask[candidate] = False
            continue
        mask[candidate] = all(
            np.linalg.norm(problem.points[candidate] - problem.points[previous])
            + 1e-12
            >= problem.config.d_min
            for previous in chosen
        )
    return mask


def solve_ceo(
    problem: DiscreteProblem,
    rng: np.random.Generator,
) -> MethodResult:
    """Run manuscript Algorithm 1 with explicit, isolated randomness."""
    start = perf_counter()
    evaluations_before = problem.evaluations
    algorithm = problem.config.ceo
    sample_size = algorithm.sample_factor * problem.num_positions
    elite_count = max(1, int(algorithm.elite_ratio * sample_size))
    p_tx = np.full(
        (problem.config.m_t, problem.num_positions),
        1.0 / problem.num_positions,
    )
    p_rx = np.full(problem.num_positions, 1.0 / problem.num_positions)

    legacy_seed = int(rng.integers(0, 2**32 - 1))
    sampling_rng = np.random.RandomState(legacy_seed)
    best_rate = -np.inf
    best_tx_order: tuple[int, ...] | None = None
    best_rx: int | None = None
    best_beamformer = np.empty((0, 1), dtype=complex)
    history: list[dict[str, float | int]] = []

    for iteration in range(algorithm.max_iterations):
        candidates: list[tuple[tuple[int, ...], int, np.ndarray]] = []
        objectives: list[float] = []
        for _ in range(sample_size):
            chosen: list[int] = []
            for antenna_index in range(problem.config.m_t):
                probabilities = p_tx[antenna_index].copy()
                probabilities[~_feasible_mask(problem, chosen)] = 0.0
                mass = float(np.sum(probabilities))
                if mass <= 0.0:
                    raise RuntimeError("CEO sampling found no feasible candidate")
                probabilities /= mass
                chosen.append(
                    int(sampling_rng.choice(problem.num_positions, p=probabilities))
                )
            rx_index = int(sampling_rng.choice(problem.num_positions, p=p_rx))
            tx_indices = tuple(chosen)
            beamformer, rate = problem.evaluate(tx_indices, rx_index)
            candidates.append((tx_indices, rx_index, beamformer))
            objectives.append(rate)

        order = np.argsort(objectives)[::-1]
        elite_indices = order[:elite_count]
        iteration_best = int(order[0])
        if objectives[iteration_best] > best_rate:
            best_rate = float(objectives[iteration_best])
            best_tx_order, best_rx, best_beamformer = candidates[iteration_best]

        next_tx = np.zeros_like(p_tx)
        next_rx = np.zeros_like(p_rx)
        for elite_index in elite_indices:
            tx_indices, rx_index, _ = candidates[int(elite_index)]
            for antenna_index, port_index in enumerate(tx_indices):
                next_tx[antenna_index, port_index] += 1.0
            next_rx[rx_index] += 1.0
        next_tx /= elite_count
        next_rx /= elite_count
        p_tx = algorithm.smoothing * next_tx + (1.0 - algorithm.smoothing) * p_tx
        p_rx = algorithm.smoothing * next_rx + (1.0 - algorithm.smoothing) * p_rx
        p_tx /= np.sum(p_tx, axis=1, keepdims=True)
        p_rx /= np.sum(p_rx)

        elite_values = np.asarray([objectives[int(index)] for index in elite_indices])
        history.append(
            {
                "iteration": iteration + 1,
                "iteration_best": float(objectives[iteration_best]),
                "global_best": float(best_rate),
                "sample_mean": float(np.mean(objectives)),
                "sample_std": float(np.std(objectives)),
                "elite_mean": float(np.mean(elite_values)),
                "elite_std": float(np.std(elite_values)),
            }
        )
        if elite_values.size > 1 and float(np.std(elite_values)) < algorithm.tolerance:
            break

    if best_tx_order is None or best_rx is None:
        raise RuntimeError("CEO failed to produce a feasible result")
    sorted_tx = tuple(sorted(best_tx_order))
    permutation = [best_tx_order.index(port) for port in sorted_tx]
    sorted_beamformer = best_beamformer[permutation]
    return MethodResult(
        method="Proposed CEO",
        secrecy_rate=best_rate,
        tx_indices=sorted_tx,
        rx_index=best_rx,
        beamformer=sorted_beamformer,
        elapsed_seconds=perf_counter() - start,
        iterations=len(history),
        evaluations=problem.evaluations - evaluations_before,
        history=tuple(history),
    )
