"""Reproduce manuscript Figure 3 from existing data or a new Bessel run."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.experiments import figure3
from scripts import figure_parser, print_paths, quick_config, validate_figure_args


def main() -> None:
    parser = figure_parser(
        3,
        ROOT / "results" / "fig3_fig4.json",
        ROOT / "outputs" / "fig3",
    )
    parser.set_defaults(num_samples=20)
    args = parser.parse_args()
    validate_figure_args(parser, args)
    if not args.recompute:
        print_paths(figure3.plot_existing(args.data, args.output_prefix))
        return
    data = figure3.run_experiment(
        config=quick_config() if args.quick else None,
        m_t_values=(2,) if args.quick else (2, 3, 4, 5, 6),
        num_samples=args.num_samples,
        seed_base=args.seed_base,
    )
    json_path = args.output_prefix.with_suffix(".json")
    figure3.save_data(data, json_path)
    print_paths((json_path, *figure3.plot(data, args.output_prefix)))


if __name__ == "__main__":
    main()
