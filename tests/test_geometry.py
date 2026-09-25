import unittest

import numpy as np

from code.common.config import CeoConfig, GreedyConfig, PgdConfig, SystemConfig
from code.common.geometry import generate_grid, index_to_normalized, is_feasible_indices


class GeometryTest(unittest.TestCase):
    def test_grid_and_index_mapping_are_consistent(self):
        grid = generate_grid(3)
        np.testing.assert_allclose(grid[0], [-1.0, -1.0])
        np.testing.assert_allclose(grid[1], [-1.0, 0.0])
        np.testing.assert_allclose(grid[-1], [1.0, 1.0])
        np.testing.assert_allclose(index_to_normalized(4, 3), [0.0, 0.0])

    def test_impossible_spacing_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "minimum spacing"):
            SystemConfig(grid_size=2, m_t=4, d_min=1.0).validate_geometry()

    def test_duplicate_indices_are_not_feasible(self):
        points = generate_grid(3) * 0.06
        self.assertFalse(is_feasible_indices((0, 0), points, 0.03))

    def test_invalid_physical_and_optimizer_parameters_are_rejected(self):
        invalid = (
            SystemConfig(d_bob=0.0),
            SystemConfig(d_eve=-1.0),
            SystemConfig(ceo=CeoConfig(tolerance=-1.0)),
            SystemConfig(greedy=GreedyConfig(tolerance=-1.0)),
            SystemConfig(pgd=PgdConfig(initial_step=0.0)),
            SystemConfig(pgd=PgdConfig(minimum_step=-1.0)),
            SystemConfig(pgd=PgdConfig(gradient_epsilon=0.0)),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()


if __name__ == "__main__":
    unittest.main()
