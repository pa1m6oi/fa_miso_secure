"""Fixed-position antenna baseline."""

from __future__ import annotations

from time import perf_counter

from code.common.geometry import fixed_port_indices
from code.common.types import MethodResult
from code.solvers.context import DiscreteProblem


def solve_fpa(problem: DiscreteProblem) -> MethodResult:
    start = perf_counter()
    evaluations_before = problem.evaluations
    tx_indices, rx_index = fixed_port_indices(
        problem.config.m_t,
        problem.config.grid_size,
    )
    if not problem.is_feasible(tx_indices):
        raise RuntimeError("fixed-position selection is infeasible")
    beamformer, rate = problem.evaluate(tx_indices, rx_index)
    return MethodResult(
        method="FPA",
        secrecy_rate=rate,
        tx_indices=tx_indices,
        rx_index=rx_index,
        beamformer=beamformer,
        elapsed_seconds=perf_counter() - start,
        iterations=1,
        evaluations=problem.evaluations - evaluations_before,
    )
