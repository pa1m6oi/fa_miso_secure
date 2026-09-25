"""Small helpers shared by the public command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from code.common.config import CeoConfig, PgdConfig, SystemConfig


def figure_parser(number: int, data: Path, output_prefix: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Plot or recompute manuscript Figure {number}."
    )
    parser.add_argument("--data", type=Path, default=data, help="existing JSON data path")
    parser.add_argument("--output-prefix", type=Path, default=output_prefix)
    parser.add_argument("--recompute", action="store_true", help="run algorithms before plotting")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="with --recompute, run one small representative sweep point",
    )
    return parser


def validate_figure_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.quick and not args.recompute:
        parser.error("--quick requires --recompute")
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")


def quick_config() -> SystemConfig:
    return SystemConfig(
        grid_size=3,
        m_t=2,
        ceo=CeoConfig(max_iterations=1),
        pgd=PgdConfig(iterations=1),
    )


def print_paths(paths: tuple[Path, ...]) -> None:
    for path in paths:
        print(path)
