import unittest

import numpy as np
from scipy.special import j0

from code.common.channel import (
    bessel_correlations,
    discrete_channel_matrices,
    effective_channels,
    normalized_to_rx_positions,
    sample_channel,
)
from code.common.config import SystemConfig
from code.common.geometry import generate_grid


class ChannelTest(unittest.TestCase):
    def test_equation_20_correlations_use_first_port_reference(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        positions = normalized_to_rx_positions(generate_grid(3), cfg)
        actual = bessel_correlations(positions, cfg.wavelength)
        distances = np.linalg.norm(positions - positions[0], axis=1)
        expected = j0(2.0 * np.pi * distances / cfg.wavelength)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
        self.assertEqual(actual[0], 1.0)
        self.assertAlmostEqual(actual[1], actual[3])

    def test_fixed_seed_sampling_is_deterministic(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        left = sample_channel(np.random.default_rng(42), cfg)
        right = sample_channel(np.random.default_rng(42), cfg)
        np.testing.assert_array_equal(left.bob_gains, right.bob_gains)
        np.testing.assert_array_equal(left.eve_gains, right.eve_gains)
        np.testing.assert_array_equal(left.bob_directions, right.bob_directions)
        np.testing.assert_array_equal(left.eve_directions, right.eve_directions)
        np.testing.assert_allclose(
            np.linalg.norm(left.bob_directions, axis=-1),
            1.0,
            rtol=0.0,
            atol=1e-14,
        )
        self.assertTrue(np.all(left.bob_directions[..., 0] >= 0.0))

    def test_complex_gain_power_matches_cn_variance(self):
        cfg = SystemConfig(
            grid_size=2,
            m_t=2,
            num_bob_paths=20_000,
            num_eve_paths=20_000,
        )
        realization = sample_channel(np.random.default_rng(123), cfg)
        reference_loss = 10.0 ** (cfg.path_loss_db / 10.0)
        bob_variance = reference_loss * cfg.d_bob ** (-cfg.path_loss_exponent)
        eve_variance = reference_loss * cfg.d_eve ** (-cfg.path_loss_exponent)
        self.assertAlmostEqual(
            float(np.mean(np.abs(realization.bob_gains) ** 2)) / bob_variance,
            1.0,
            delta=0.03,
        )
        self.assertAlmostEqual(
            float(np.mean(np.abs(realization.eve_gains) ** 2)) / eve_variance,
            1.0,
            delta=0.03,
        )

    def test_effective_channel_shapes_are_valid(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        realization = sample_channel(np.random.default_rng(1), cfg)
        h_b, h_e = effective_channels(
            np.array([[-0.06, -0.06], [0.06, 0.06]]),
            np.array([[cfg.d_bob - 0.06, -0.06]]),
            realization,
            cfg,
        )
        self.assertEqual(h_b.shape, (1, 2))
        self.assertEqual(h_e.shape, (3, 2))
        self.assertTrue(np.isfinite(h_b).all())
        self.assertTrue(np.isfinite(h_e).all())

    def test_continuous_receive_position_changes_the_channel(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        realization = sample_channel(np.random.default_rng(7), cfg)
        tx = np.array([[-0.04, -0.03], [0.05, 0.02]])
        center = np.array([[cfg.d_bob, 0.0]])
        left, _ = effective_channels(tx, center, realization, cfg)
        right, _ = effective_channels(
            tx,
            center + np.array([[1e-5, 0.0]]),
            realization,
            cfg,
        )
        self.assertGreater(float(np.linalg.norm(left - right)), 1e-12)

    def test_discrete_channel_matrices_cover_every_port(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        realization = sample_channel(np.random.default_rng(8), cfg)
        h_b, h_e = discrete_channel_matrices(realization, cfg, generate_grid(3))
        self.assertEqual(h_b.shape, (9, 9))
        self.assertEqual(h_e.shape, (3, 9))
        self.assertTrue(np.isfinite(h_b).all())
        self.assertTrue(np.isfinite(h_e).all())


if __name__ == "__main__":
    unittest.main()
