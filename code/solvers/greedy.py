"""Low-complexity greedy joint port-selection algorithm."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np

from code.common.types import MethodResult
from code.solvers.context import DiscreteProblem


RateFunction = Callable[[tuple[int, ...], int], float]


def _can_add(problem: DiscreteProblem, chosen: tuple[int, ...], candidate: int) -> bool:
    return candidate not in chosen and all(
        np.linalg.norm(problem.points[candidate] - problem.points[previous]) + 1e-12
        >= problem.config.d_min
        for previous in chosen
    )


def _best_receiver(
    problem: DiscreteProblem,
    tx_indices: tuple[int, ...],
    rate_for: RateFunction,
) -> tuple[int, float]:
    best_rx = 0
    best_rate = -np.inf
    for rx_index in range(problem.num_positions):
        rate = rate_for(tx_indices, rx_index)
        if rate > best_rate:
            best_rx = rx_index
            best_rate = rate
    return best_rx, float(best_rate)


def _initial_pair(
    problem: DiscreteProblem,
    rate_for: RateFunction,
) -> tuple[tuple[int, ...], int, float, list[tuple[object, ...]]]:
    best_tx = 0
    best_rx = 0
    best_rate = -np.inf
    for tx_index in range(problem.num_positions):
        for rx_index in range(problem.num_positions):
            rate = rate_for((tx_index,), rx_index)
            if rate > best_rate:
                best_tx, best_rx, best_rate = tx_index, rx_index, rate
    history = [("both", best_tx, best_rx, float(best_rate))]
    return (best_tx,), best_rx, float(best_rate), history


def _fill_transmit_ports(
    problem: DiscreteProblem,
    tx_indices: tuple[int, ...],
    rx_index: int,
    current_rate: float,
    history: list[tuple[object, ...]],
    rate_for: RateFunction,
) -> tuple[tuple[int, ...], int, float]:
    while len(tx_indices) < problem.config.m_t:
        best_candidate: int | None = None
        best_rate = current_rate
        for candidate in range(problem.num_positions):
            if not _can_add(problem, tx_indices, candidate):
                continue
            trial = (*tx_indices, candidate)
            _, rate = _best_receiver(problem, trial, rate_for)
            if rate > best_rate:
                best_candidate, best_rate = candidate, rate

        if best_candidate is None:
            fallback_rate = -np.inf
            for candidate in range(problem.num_positions):
                if not _can_add(problem, tx_indices, candidate):
                    continue
                trial = (*tx_indices, candidate)
                _, rate = _best_receiver(problem, trial, rate_for)
                if rate > fallback_rate:
                    best_candidate, fallback_rate = candidate, rate
            if best_candidate is None:
                raise RuntimeError("greedy search cannot complete a feasible selection")

        tx_indices = (*tx_indices, best_candidate)
        rx_index, current_rate = _best_receiver(problem, tx_indices, rate_for)
        history.append(("tx", best_candidate, None, current_rate))
    return tx_indices, rx_index, current_rate


def _refine_ports(
    problem: DiscreteProblem,
    tx_indices: tuple[int, ...],
    rx_index: int,
    current_rate: float,
    history: list[tuple[object, ...]],
    rate_for: RateFunction,
) -> tuple[tuple[int, ...], int, float]:
    for _ in range(problem.config.greedy.max_iterations):
        improved = False
        candidate_rx, candidate_rate = _best_receiver(problem, tx_indices, rate_for)
        if candidate_rate > current_rate + problem.config.greedy.tolerance:
            rx_index, current_rate = candidate_rx, candidate_rate
            history.append(("rx_reopt", None, rx_index, current_rate))
            improved = True

        best_swap: tuple[int, int] | None = None
        best_swap_rate = current_rate
        for outgoing in tx_indices:
            remainder = tuple(port for port in tx_indices if port != outgoing)
            for incoming in range(problem.num_positions):
                if incoming in tx_indices or not _can_add(problem, remainder, incoming):
                    continue
                trial = (*remainder, incoming)
                rate = rate_for(trial, rx_index)
                if rate > best_swap_rate + problem.config.greedy.tolerance:
                    best_swap = outgoing, incoming
                    best_swap_rate = rate
        if best_swap is not None:
            outgoing, incoming = best_swap
            tx_indices = tuple(incoming if port == outgoing else port for port in tx_indices)
            current_rate = best_swap_rate
            history.append(("tx_swap_fixed_rx", outgoing, incoming, current_rate))
            improved = True
        if not improved:
            break
    return tx_indices, rx_index, current_rate


def solve_greedy(problem: DiscreteProblem) -> MethodResult:
    """Run manuscript Algorithm 2 without embedding beamforming logic."""
    start = perf_counter()
    evaluations_before = problem.evaluations
    cache: dict[tuple[frozenset[int], int], float] = {}

    def rate_for(tx_indices: tuple[int, ...], rx_index: int) -> float:
        key = frozenset(tx_indices), rx_index
        if key not in cache:
            _, cache[key] = problem.evaluate(tx_indices, rx_index)
        return cache[key]

    tx_indices, rx_index, rate, history = _initial_pair(problem, rate_for)
    tx_indices, rx_index, rate = _fill_transmit_ports(
        problem,
        tx_indices,
        rx_index,
        rate,
        history,
        rate_for,
    )
    tx_indices, rx_index, rate = _refine_ports(
        problem,
        tx_indices,
        rx_index,
        rate,
        history,
        rate_for,
    )
    beamformer, final_rate = problem.evaluate(tx_indices, rx_index)
    sorted_tx = tuple(sorted(tx_indices))
    permutation = [tx_indices.index(port) for port in sorted_tx]
    sorted_beamformer = beamformer[permutation]
    return MethodResult(
        method="Proposed Greedy",
        secrecy_rate=final_rate,
        tx_indices=sorted_tx,
        rx_index=rx_index,
        beamformer=sorted_beamformer,
        elapsed_seconds=perf_counter() - start,
        iterations=len(history),
        evaluations=problem.evaluations - evaluations_before,
        history=tuple(history),
    )
