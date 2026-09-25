"""Deterministic shared-channel execution for all manuscript methods."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from code.common.channel import discrete_channel_matrices, sample_channel
from code.common.config import SystemConfig
from code.common.geometry import generate_grid
from code.common.metrics import summarize
from code.common.types import MethodResult
from code.solvers.ceo import solve_ceo
from code.solvers.context import DiscreteProblem
from code.solvers.exhaustive import solve_exhaustive
from code.solvers.fpa import solve_fpa
from code.solvers.greedy import solve_greedy
from code.solvers.pgd import (
    solve_pgd_from_ceo,
    solve_pgd_from_exhaustive,
    solve_pgd_from_fpa,
    solve_pgd_from_greedy,
    solve_pgd_from_random,
    solve_pgd_multiple_random,
)
from code.solvers.random_search import solve_random


SUPPORTED_METHODS = (
    "Proposed CEO",
    "Proposed Greedy",
    "Exhaustive",
    "Random",
    "FPA",
    "PGD",
    "PGD(Multiple)",
    "PGD(Random init)",
    "PGD(CEO init)",
    "PGD(Greedy init)",
    "UB",
)


def _method_rng(seed: int, method: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, SUPPORTED_METHODS.index(method)]))


def _dispatch(
    method: str,
    rng: np.random.Generator,
    problem: DiscreteProblem,
    realization,
    config: SystemConfig,
) -> MethodResult:
    if method == "Proposed CEO":
        return solve_ceo(problem, rng)
    if method == "Proposed Greedy":
        return solve_greedy(problem)
    if method == "Exhaustive":
        return solve_exhaustive(problem)
    if method == "Random":
        return solve_random(problem, rng)
    if method == "FPA":
        return solve_fpa(problem)
    if method == "PGD":
        return solve_pgd_from_fpa(rng, problem, realization, config)
    if method == "PGD(Multiple)":
        return solve_pgd_multiple_random(
            rng,
            problem,
            realization,
            config,
            starts=config.m_t * problem.num_positions,
        )
    if method == "PGD(Random init)":
        return solve_pgd_from_random(rng, problem, realization, config)
    if method == "PGD(CEO init)":
        return solve_pgd_from_ceo(rng, problem, realization, config)
    if method == "PGD(Greedy init)":
        return solve_pgd_from_greedy(rng, problem, realization, config)
    if method == "UB":
        return solve_pgd_from_exhaustive(rng, problem, realization, config)
    supported = ", ".join(SUPPORTED_METHODS)
    raise ValueError(f"unknown method {method!r}; supported methods: {supported}")


def run_methods(
    config: SystemConfig,
    methods: Iterable[str],
    seed: int,
) -> dict[str, MethodResult]:
    """Run selected methods on one shared Bessel channel realization."""
    config.validate()
    selected = tuple(methods)
    unknown = tuple(method for method in selected if method not in SUPPORTED_METHODS)
    if unknown:
        supported = ", ".join(SUPPORTED_METHODS)
        raise ValueError(f"unknown methods {unknown}; supported methods: {supported}")
    realization = sample_channel(np.random.default_rng(seed), config)
    h_b_full, h_e_full = discrete_channel_matrices(
        realization,
        config,
        generate_grid(config.grid_size),
    )
    results: dict[str, MethodResult] = {}
    for method in selected:
        problem = DiscreteProblem(h_b_full, h_e_full, config)
        results[method] = _dispatch(
            method,
            _method_rng(seed, method),
            problem,
            realization,
            config,
        )
    return results


def run_method_suite(
    config: SystemConfig,
    methods: Iterable[str],
    num_samples: int,
    seed_base: int,
) -> dict:
    """Aggregate deterministic rates and separate nondeterministic timings."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    selected = tuple(methods)
    sample_seeds = [seed_base + index for index in range(num_samples)]
    rates = {method: [] for method in selected}
    times = {method: [] for method in selected}
    for seed in sample_seeds:
        sample_results = run_methods(config, selected, seed)
        for method in selected:
            rates[method].append(sample_results[method].secrecy_rate)
            times[method].append(sample_results[method].elapsed_seconds)

    series = {}
    timing_seconds = {}
    for method in selected:
        rate_stats = summarize(rates[method])
        time_stats = summarize(times[method])
        series[method] = {
            "values": [float(value) for value in rates[method]],
            "mean": rate_stats["mean"],
            "std": rate_stats["std"],
        }
        timing_seconds[method] = {
            "values": [float(value) for value in times[method]],
            "mean": time_stats["mean"],
            "std": time_stats["std"],
        }
    return {
        "config": {
            "grid_size": config.grid_size,
            "m_t": config.m_t,
            "num_samples": num_samples,
            "seed_base": seed_base,
        },
        "sample_seeds": sample_seeds,
        "series": series,
        "timing_seconds": timing_seconds,
    }
