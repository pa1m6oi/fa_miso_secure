"""Block projected-gradient optimization and named initializations."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from code.common.beamforming import solve_grq
from code.common.channel import (
    effective_channels,
    normalized_to_rx_positions,
    normalized_to_tx_positions,
)
from code.common.config import SystemConfig
from code.common.geometry import index_to_normalized, is_feasible_continuous
from code.common.metrics import objective_from_positions
from code.common.types import ChannelRealization, MethodResult
from code.solvers.ceo import solve_ceo
from code.solvers.context import DiscreteProblem
from code.solvers.exhaustive import solve_exhaustive
from code.solvers.fpa import solve_fpa
from code.solvers.greedy import solve_greedy
from code.solvers.random_search import solve_random


def _sample_feasible_continuous(
    rng: np.random.Generator,
    config: SystemConfig,
    max_attempts: int = 50_000,
) -> np.ndarray:
    size = 2 * config.m_t + 2
    for _ in range(max_attempts):
        candidate = rng.uniform(-1.0, 1.0, size=size)
        if is_feasible_continuous(candidate, config.m_t, config):
            return candidate
    raise RuntimeError("failed to sample a feasible PGD initialization")


def _numerical_gradient(
    x: np.ndarray,
    block: tuple[int, int],
    realization: ChannelRealization,
    config: SystemConfig,
) -> tuple[np.ndarray, int]:
    gradient = np.zeros(2, dtype=float)
    evaluations = 0
    for offset, index in enumerate(block):
        positive = x.copy()
        negative = x.copy()
        positive[index] = np.clip(
            positive[index] + config.pgd.gradient_epsilon,
            -1.0,
            1.0,
        )
        negative[index] = np.clip(
            negative[index] - config.pgd.gradient_epsilon,
            -1.0,
            1.0,
        )
        upper = objective_from_positions(positive, realization, config)
        lower = objective_from_positions(negative, realization, config)
        evaluations += 2
        denominator = positive[index] - negative[index]
        gradient[offset] = 0.0 if denominator == 0 else (upper - lower) / denominator
    return gradient, evaluations


def _beamformer_at(
    x: np.ndarray,
    realization: ChannelRealization,
    config: SystemConfig,
) -> np.ndarray:
    tx = normalized_to_tx_positions(x[: 2 * config.m_t].reshape(config.m_t, 2), config)
    rx = normalized_to_rx_positions(x[2 * config.m_t :].reshape(1, 2), config)
    h_b, h_e = effective_channels(tx, rx, realization, config)
    beamformer, _ = solve_grq(h_b, h_e, config)
    return beamformer


def solve_pgd(
    rng: np.random.Generator,
    realization: ChannelRealization,
    config: SystemConfig,
    x0: np.ndarray | None = None,
    init_name: str = "Random",
) -> MethodResult:
    """Run paper-style feasible block PGD from one initialization."""
    start = perf_counter()
    if x0 is None:
        x = _sample_feasible_continuous(rng, config)
    else:
        candidate = np.asarray(x0, dtype=float).reshape(-1)
        if not is_feasible_continuous(candidate, config.m_t, config):
            raise ValueError("PGD initialization is invalid or infeasible")
        x = candidate.copy()

    current = objective_from_positions(x, realization, config)
    evaluations = 1
    history: list[object] = [("init", init_name, float(current))]
    blocks = [(2 * index, 2 * index + 1) for index in range(config.m_t + 1)]

    for iteration in range(config.pgd.iterations):
        improved = False
        for block in blocks:
            gradient, gradient_evaluations = _numerical_gradient(
                x,
                block,
                realization,
                config,
            )
            evaluations += gradient_evaluations
            norm = float(np.linalg.norm(gradient))
            if not np.isfinite(norm) or norm < 1e-12:
                continue
            step = config.pgd.initial_step
            while step >= config.pgd.minimum_step:
                trial = x.copy()
                trial[list(block)] = np.clip(
                    trial[list(block)] + step * gradient,
                    -1.0,
                    1.0,
                )
                if is_feasible_continuous(trial, config.m_t, config):
                    trial_value = objective_from_positions(trial, realization, config)
                    evaluations += 1
                    if trial_value > current + 1e-12:
                        x = trial
                        current = trial_value
                        improved = True
                        history.append(("update", iteration, block, float(current)))
                        break
                step *= 0.5
        if not improved:
            break

    return MethodResult(
        method="PGD",
        secrecy_rate=float(current),
        tx_indices=None,
        rx_index=None,
        beamformer=_beamformer_at(x, realization, config),
        elapsed_seconds=perf_counter() - start,
        iterations=len(history),
        evaluations=evaluations,
        continuous_positions=x.copy(),
        history=tuple(history),
    )


def _continuous_from_indices(
    tx_indices: tuple[int, ...],
    rx_index: int | None,
    grid_size: int,
) -> np.ndarray:
    if rx_index is None:
        raise ValueError("rx_index is required for a PGD initialization")
    tx = np.concatenate(
        [index_to_normalized(index, grid_size) for index in tx_indices]
    )
    return np.concatenate((tx, index_to_normalized(rx_index, grid_size)))


def _solve_from_discrete(
    rng: np.random.Generator,
    problem: DiscreteProblem,
    realization: ChannelRealization,
    config: SystemConfig,
    initial: MethodResult,
    method: str,
) -> MethodResult:
    if initial.tx_indices is None:
        raise RuntimeError("discrete initializer returned no transmit selection")
    x0 = _continuous_from_indices(
        initial.tx_indices,
        initial.rx_index,
        problem.config.grid_size,
    )
    result = solve_pgd(rng, realization, config, x0, initial.method)
    return replace(result, method=method)


def solve_pgd_from_fpa(
    rng: np.random.Generator,
    problem: DiscreteProblem,
    realization: ChannelRealization,
    config: SystemConfig,
) -> MethodResult:
    return _solve_from_discrete(
        rng,
        problem,
        realization,
        config,
        solve_fpa(problem),
        "PGD",
    )


def solve_pgd_multiple_random(
    rng: np.random.Generator,
    problem: DiscreteProblem,
    realization: ChannelRealization,
    config: SystemConfig,
    starts: int,
) -> MethodResult:
    if starts < 1:
        raise ValueError("starts must be positive")
    started = perf_counter()
    results = [
        solve_pgd(rng, realization, config, None, "Random") for _ in range(starts)
    ]
    best = max(results, key=lambda result: result.secrecy_rate)
    return replace(
        best,
        method="PGD(Multiple)",
        elapsed_seconds=perf_counter() - started,
        iterations=sum(result.iterations for result in results),
        evaluations=sum(result.evaluations for result in results),
    )


def solve_pgd_from_random(
    rng: np.random.Generator,
    problem: DiscreteProblem,
    realization: ChannelRealization,
    config: SystemConfig,
) -> MethodResult:
    initial = solve_random(
        problem,
        rng,
        trials=problem.config.m_t * problem.num_positions,
    )
    return _solve_from_discrete(
        rng,
        problem,
        realization,
        config,
        initial,
        "PGD(Random init)",
    )


def solve_pgd_from_ceo(
    rng: np.random.Generator,
    problem: DiscreteProblem,
    realization: ChannelRealization,
    config: SystemConfig,
) -> MethodResult:
    return _solve_from_discrete(
        rng,
        problem,
        realization,
        config,
        solve_ceo(problem, rng),
        "PGD(CEO init)",
    )


def solve_pgd_from_greedy(
    rng: np.random.Generator,
    problem: DiscreteProblem,
    realization: ChannelRealization,
    config: SystemConfig,
) -> MethodResult:
    return _solve_from_discrete(
        rng,
        problem,
        realization,
        config,
        solve_greedy(problem),
        "PGD(Greedy init)",
    )


def solve_pgd_from_exhaustive(
    rng: np.random.Generator,
    problem: DiscreteProblem,
    realization: ChannelRealization,
    config: SystemConfig,
) -> MethodResult:
    return _solve_from_discrete(
        rng,
        problem,
        realization,
        config,
        solve_exhaustive(problem),
        "UB",
    )
