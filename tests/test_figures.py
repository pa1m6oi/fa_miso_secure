import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from code.common.config import SystemConfig
from code.experiments import figure2, figure3, figure4, figure5, figure6


RELEASE_ROOT = Path(__file__).resolve().parents[1]


class Figure2Test(unittest.TestCase):
    def test_existing_data_loads_and_plots_without_recomputation(self):
        with patch("code.experiments.figure2.run_experiment") as run:
            paths = figure2.plot_existing(
                RELEASE_ROOT / "results",
                RELEASE_ROOT / "outputs" / "fig2_test",
            )
        run.assert_not_called()
        self.assertTrue(all(path.exists() for path in paths))

    def test_missing_json_names_the_path(self):
        missing = Path("results/not-present.json")
        with self.assertRaisesRegex(FileNotFoundError, "not-present.json"):
            figure2.load_data(missing)

    def test_malformed_json_lists_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"wrong": 1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "meta.*sample_size"):
                figure2.load_data(path)


class Figure34Test(unittest.TestCase):
    def test_fig3_uses_discrete_series_only(self):
        data = figure3.load_data(RELEASE_ROOT / "results" / "fig3_fig4.json")
        self.assertEqual(
            set(figure3.series_for_plot(data)),
            {"CEO", "Greedy", "Exhaustive", "Random", "FPA"},
        )

    def test_fig4_uses_pgd_initialization_series_only(self):
        data = figure4.load_data(RELEASE_ROOT / "results" / "fig3_fig4.json")
        self.assertIn("PGD(Random)", figure4.series_for_plot(data))
        self.assertIn("UB", figure4.series_for_plot(data))
        self.assertNotIn("FPA", figure4.series_for_plot(data))

    def test_fig4_matches_manuscript_solid_line_styles(self):
        self.assertTrue(
            all(style["linestyle"] == "-" for style in figure4.STYLES.values())
        )

    def test_plot_only_does_not_call_runner(self):
        data_path = RELEASE_ROOT / "results" / "fig3_fig4.json"
        with patch("code.experiments.figure3.run_experiment") as fig3_run, patch(
            "code.experiments.figure4.run_experiment"
        ) as fig4_run:
            fig3_paths = figure3.plot_existing(
                data_path,
                RELEASE_ROOT / "outputs" / "fig3_test",
            )
            fig4_paths = figure4.plot_existing(
                data_path,
                RELEASE_ROOT / "outputs" / "fig4_test",
            )
        fig3_run.assert_not_called()
        fig4_run.assert_not_called()
        self.assertTrue(all(path.exists() for path in (*fig3_paths, *fig4_paths)))

    def test_single_point_quick_plots_do_not_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                fig3_paths = figure3.plot(
                    {method: {"2": 1.0} for method in figure3.METHODS},
                    root / "fig3",
                )
                fig4_paths = figure4.plot(
                    {method: {"2": 1.0} for method in figure4.METHODS},
                    root / "fig4",
                )
            self.assertTrue(all(path.exists() for path in (*fig3_paths, *fig4_paths)))


class Figure56Test(unittest.TestCase):
    def test_fig5_dbm_conversion_matches_transmit_power(self):
        config = figure5._config_for_power(
            SystemConfig(),
            20.0,
        )
        self.assertAlmostEqual(config.p_max, 0.1, places=14)

    def test_fig5_axis_is_transmit_power(self):
        data = figure5.load_data(RELEASE_ROOT / "results" / "fig5.json")
        self.assertEqual(figure5.x_values(data), [-5, 0, 5, 10, 15, 20, 25])

    def test_fig6_axis_is_grid_size(self):
        data = figure6.load_data(RELEASE_ROOT / "results" / "fig6.json")
        self.assertEqual(figure6.x_values(data), [3, 4, 5, 6, 7])

    def test_fig5_and_fig6_plot_only_do_not_recompute(self):
        with patch("code.experiments.figure5.run_experiment") as run5, patch(
            "code.experiments.figure6.run_experiment"
        ) as run6:
            fig5_paths = figure5.plot_existing(
                RELEASE_ROOT / "results" / "fig5.json",
                RELEASE_ROOT / "outputs" / "fig5_test",
            )
            fig6_paths = figure6.plot_existing(
                RELEASE_ROOT / "results" / "fig6.json",
                RELEASE_ROOT / "outputs" / "fig6_test",
            )
        run5.assert_not_called()
        run6.assert_not_called()
        self.assertTrue(all(path.exists() for path in (*fig5_paths, *fig6_paths)))


if __name__ == "__main__":
    unittest.main()
