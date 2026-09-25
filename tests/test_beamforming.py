import math
import unittest

import numpy as np

from code.common.beamforming import solve_grq
from code.common.channel import sample_channel
from code.common.config import SystemConfig
from code.common.metrics import objective_from_positions, secrecy_rate, summarize


class BeamformingTest(unittest.TestCase):
    def test_grq_respects_power_and_returns_finite_rate(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        h_b = np.array([[1 + 0j, 0.5 + 0.2j]])
        h_e = np.array([[0.1 + 0j, 0.2 + 0.1j], [0.05j, 0.1 + 0j]])
        w, rate = solve_grq(h_b, h_e, cfg)
        self.assertEqual(w.shape, (2, 1))
        self.assertLessEqual(float(np.vdot(w, w).real), cfg.p_max + 1e-12)
        self.assertTrue(np.isfinite(rate))
        self.assertGreaterEqual(rate, 0.0)

    def test_zero_channels_do_not_produce_nan(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        w, rate = solve_grq(
            np.zeros((1, 2), complex),
            np.zeros((3, 2), complex),
            cfg,
        )
        self.assertTrue(np.isfinite(w).all())
        self.assertEqual(rate, 0.0)

    def test_nearly_singular_channels_remain_finite(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        h_b = np.array([[1.0 + 0j, 1.0 + 1e-14j]])
        h_e = np.vstack([h_b, h_b * (1.0 + 1e-14), h_b * (1.0 - 1e-14)])
        w, rate = solve_grq(h_b, h_e, cfg)
        self.assertTrue(np.isfinite(w).all())
        self.assertTrue(np.isfinite(rate))
        self.assertLessEqual(float(np.vdot(w, w).real), cfg.p_max + 1e-12)

    def test_secrecy_rate_matches_scalar_definition(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        h_b = np.array([[1.0 + 0j, 0.0 + 0j]])
        h_e = np.zeros((3, 2), dtype=complex)
        w = np.array([[math.sqrt(cfg.p_max)], [0.0]], dtype=complex)
        expected = math.log2(1.0 + cfg.p_max / cfg.noise_power)
        self.assertAlmostEqual(secrecy_rate(h_b, h_e, w, cfg), expected, places=12)

    def test_continuous_objective_and_summary_are_finite(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        realization = sample_channel(np.random.default_rng(9), cfg)
        x = np.array([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0])
        value = objective_from_positions(x, realization, cfg)
        stats = summarize([value, value + 2.0])
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(stats["mean"], value + 1.0)
        self.assertAlmostEqual(stats["std"], 1.0)
        self.assertEqual(stats["count"], 2)


if __name__ == "__main__":
    unittest.main()
