import unittest
from unittest.mock import patch

from code.common.channel import sample_channel
from code.common.config import PgdConfig, SystemConfig
from code.experiments.runner import run_method_suite


class RunnerTest(unittest.TestCase):
    def test_selected_methods_share_one_channel_realization(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        with patch(
            "code.experiments.runner.sample_channel",
            wraps=sample_channel,
        ) as sampler:
            report = run_method_suite(
                config=cfg,
                methods=("FPA", "Proposed Greedy"),
                num_samples=1,
                seed_base=42,
            )
        self.assertEqual(report["config"]["seed_base"], 42)
        self.assertEqual(set(report["series"]), {"FPA", "Proposed Greedy"})
        self.assertEqual(report["sample_seeds"], [42])
        self.assertEqual(sampler.call_count, 1)

    def test_method_order_does_not_change_seeded_results(self):
        cfg = SystemConfig(grid_size=3, m_t=2)
        left = run_method_suite(cfg, ("FPA", "Proposed Greedy"), 1, 42)
        right = run_method_suite(cfg, ("Proposed Greedy", "FPA"), 1, 42)
        self.assertEqual(left["series"], right["series"])

    def test_manuscript_pgd_multiple_name_is_publicly_supported(self):
        cfg = SystemConfig(
            grid_size=3,
            m_t=2,
            pgd=PgdConfig(iterations=1),
        )
        report = run_method_suite(cfg, ("PGD(Multiple)",), 1, 42)
        self.assertEqual(tuple(report["series"]), ("PGD(Multiple)",))


if __name__ == "__main__":
    unittest.main()
