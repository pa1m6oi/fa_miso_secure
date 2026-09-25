"""Reproduce manuscript Figure 2 from existing data or a new Bessel run."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.experiments import figure2
from scripts import figure_parser, print_paths, quick_config, validate_figure_args


def main() -> None:
    parser = figure_parser(2, ROOT / "results", ROOT / "outputs" / "fig2")
    parser.set_defaults(num_samples=20)
    args = parser.parse_args()
    validate_figure_args(parser, args)
    if not args.recompute:
        print_paths(figure2.plot_existing(args.data, args.output_prefix))
        return
    data = figure2.run_experiment(
        config=quick_config() if args.quick else None,
        num_samples=args.num_samples,
        seed_base=args.seed_base,
        quick=args.quick,
    )
    json_paths = tuple(
        args.output_prefix.with_name(f"{args.output_prefix.name}_{name}.json")
        for name in ("L", "rho", "varsigma")
    )
    for name, path in zip(("L", "rho", "varsigma"), json_paths, strict=True):
        figure2.save_data(data[name], path)
    print_paths((*json_paths, *figure2.plot(data, args.output_prefix)))


if __name__ == "__main__":
    main()
