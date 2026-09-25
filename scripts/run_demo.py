"""Run selected methods on a small Bessel-channel experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.common.config import SystemConfig
from code.experiments.runner import SUPPORTED_METHODS, run_method_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--m-t", type=int, default=2)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=["Proposed CEO", "Proposed Greedy", "FPA"],
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    config = SystemConfig(grid_size=args.grid_size, m_t=args.m_t)
    report = run_method_suite(
        config,
        tuple(args.methods),
        args.num_samples,
        args.seed_base,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
