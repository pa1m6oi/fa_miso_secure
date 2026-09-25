"""Reproduce manuscript Figure 5 from existing data or a new Bessel run."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.experiments import figure5
from scripts import figure_parser, print_paths, quick_config, validate_figure_args


def main() -> None:
    parser = figure_parser(
        5,
        ROOT / "results" / "fig5.json",
        ROOT / "outputs" / "fig5",
    )
    args = parser.parse_args()
    validate_figure_args(parser, args)
    if not args.recompute:
        print_paths(figure5.plot_existing(args.data, args.output_prefix))
        return
    data = figure5.run_experiment(
        config=quick_config() if args.quick else None,
        power_dbm_values=(-5,) if args.quick else figure5.DEFAULT_POWER_DBM,
        num_samples=args.num_samples,
        seed_base=args.seed_base,
    )
    json_path = args.output_prefix.with_suffix(".json")
    figure5.save_data(data, json_path)
    print_paths((json_path, *figure5.plot(data, args.output_prefix)))


if __name__ == "__main__":
    main()
