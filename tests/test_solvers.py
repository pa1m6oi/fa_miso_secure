import unittest

import numpy as np

from code.common.config import SystemConfig
from code.common.channel import sample_channel
from code.common.geometry import is_feasible_continuous
from code.common.metrics import secrecy_rate
from code.common.metrics import objective_from_positions
from code.common.types import MethodResult
from code.solvers.ceo import solve_ceo
from code.solvers.context import DiscreteProblem
from code.solvers.exhaustive import solve_exhaustive
from code.solvers.fpa import solve_fpa
from code.solvers.greedy import solve_greedy
from code.solvers.pgd import _numerical_gradient, solve_pgd
from code.solvers.random_search import solve_random


def fixed_problem(config=None):
    cfg = config or SystemConfig(grid_size=2, m_t=2, d_min=0.03)
    h_b = np.array(
        [
            [1.0, 0.8, 0.3, 0.1],
            [0.9, 1.1, 0.2, 0.4],
            [0.4, 0.2, 1.0, 0.7],
            [0.2, 0.3, 0.8, 1.2],
        ],
        dtype=complex,
    )
    h_e = np.full((cfg.num_eavesdroppers, 4), 0.05 + 0.02j)
    return DiscreteProblem(h_b, h_e, cfg)


class BaselineSolverTest(unittest.TestCase):
    def test_exhaustive_is_not_worse_than_random_or_fpa(self):
        problem = fixed_problem()
        exhaustive = solve_exhaustive(problem)
        random_result = solve_random(problem, np.random.default_rng(7), trials=8)
        fpa = solve_fpa(problem)
        self.assertGreaterEqual(
            exhaustive.secrecy_rate + 1e-12,
            random_result.secrecy_rate,
        )
        self.assertGreaterEqual(exhaustive.secrecy_rate + 1e-12, fpa.secrecy_rate)

    def test_all_returned_indices_are_feasible(self):
        problem = fixed_problem()
        for result in (solve_exhaustive(problem), solve_fpa(problem)):
            self.assertIsInstance(result, MethodResult)
            self.assertIsInstance(result.tx_indices, tuple)
            self.assertTrue(problem.is_feasible(result.tx_indices))
            self.assertGreater(result.evaluations, 0)

    def test_infeasible_problem_is_rejected_before_search(self):
        with self.assertRaisesRegex(ValueError, "minimum spacing"):
            fixed_problem(SystemConfig(grid_size=2, m_t=4, d_min=1.0))

    def test_random_search_reports_failure_instead_of_substituting_fpa(self):
        with self.assertRaisesRegex(RuntimeError, "random search found no feasible candidate"):
            solve_random(fixed_problem(), np.random.default_rng(7), trials=0)


class ProposedSolverTest(unittest.TestCase):
    def test_ceo_is_seeded_and_returns_feasible_selection(self):
        left = solve_ceo(fixed_problem(), np.random.default_rng(11))
        right = solve_ceo(fixed_problem(), np.random.default_rng(11))
        self.assertEqual(left.tx_indices, right.tx_indices)
        self.assertEqual(left.rx_index, right.rx_index)
        self.assertAlmostEqual(left.secrecy_rate, right.secrecy_rate, places=12)
        self.assertTrue(fixed_problem().is_feasible(left.tx_indices))
        self.assertGreater(len(left.history), 0)

    def test_greedy_returns_full_feasible_cardinality(self):
        problem = fixed_problem()
        result = solve_greedy(problem)
        self.assertEqual(len(result.tx_indices), problem.config.m_t)
        self.assertTrue(problem.is_feasible(result.tx_indices))
        self.assertGreater(result.evaluations, 0)

    def test_greedy_beamformer_matches_returned_port_order(self):
        problem = fixed_problem()
        result = solve_greedy(problem)
        h_b = problem.h_b_full[np.ix_([result.rx_index], result.tx_indices)]
        h_e = problem.h_e_full[:, result.tx_indices]
        returned_rate = secrecy_rate(h_b, h_e, result.beamformer, problem.config)
        self.assertAlmostEqual(returned_rate, result.secrecy_rate, places=12)


class PgdSolverTest(unittest.TestCase):
    def test_pgd_receive_block_has_a_continuous_gradient(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        realization = sample_channel(np.random.default_rng(0), cfg)
        x = np.array([-1.0, -1.0, 1.0, 1.0, 0.15, -0.2])
        gradient, evaluations = _numerical_gradient(
            x,
            (2 * cfg.m_t, 2 * cfg.m_t + 1),
            realization,
            cfg,
        )
        self.assertEqual(evaluations, 4)
        self.assertGreater(float(np.linalg.norm(gradient)), 1e-8)

    def test_pgd_preserves_feasibility_and_does_not_reduce_objective(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        realization = sample_channel(np.random.default_rng(4), cfg)
        x0 = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
        initial = objective_from_positions(x0, realization, cfg)
        result = solve_pgd(np.random.default_rng(5), realization, cfg, x0, "FPA")
        self.assertTrue(
            is_feasible_continuous(result.continuous_positions, cfg.m_t, cfg)
        )
        self.assertGreaterEqual(result.secrecy_rate + 1e-12, initial)

    def test_invalid_pgd_initialization_raises(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        realization = sample_channel(np.random.default_rng(4), cfg)
        with self.assertRaisesRegex(ValueError, "initialization"):
            solve_pgd(
                np.random.default_rng(5),
                realization,
                cfg,
                np.zeros(2),
                "invalid",
            )

    def test_pgd_can_differentiate_at_the_spacing_boundary(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        realization = sample_channel(np.random.default_rng(4), cfg)
        x0 = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        result = solve_pgd(np.random.default_rng(5), realization, cfg, x0, "boundary")
        self.assertTrue(
            is_feasible_continuous(result.continuous_positions, cfg.m_t, cfg)
        )


if __name__ == "__main__":
    unittest.main()
