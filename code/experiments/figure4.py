"""PGD-initialization sweep for manuscript Figure 4."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from code.common.config import SystemConfig
from code.experiments.runner import run_method_suite
from code.plotting.paper_style import save_figure, style_context

import matplotlib.pyplot as plt


METHODS = (
    "PGD",
    "PGD(Random)",
    "PGD(Random init)",
    "PGD(CEO init)",
    "PGD(Greedy init)",
    "UB",
)
RUNNER_NAMES = {
    method: "PGD(Multiple)" if method == "PGD(Random)" else method
    for method in METHODS
}
STYLES = {
    "PGD": {"label": "PGD", "color": "#D55E00", "marker": "d", "linestyle": "-"},
    "PGD(Random)": {
        "label": "PGD(Multiple)",
        "color": "#CC79A7",
        "marker": "P",
        "linestyle": "-",
    },
    "PGD(Random init)": {
        "label": "PGD(Random init)",
        "color": "#E69F00",
        "marker": "h",
        "linestyle": "-",
    },
    "PGD(CEO init)": {
        "label": "PGD(CEO init)",
        "color": "#0072B2",
        "marker": "X",
        "linestyle": "-",
    },
    "PGD(Greedy init)": {
        "label": "PGD(Greedy init)",
        "color": "#009E73",
        "marker": "*",
        "linestyle": "-",
    },
    "UB": {
        "label": "PGD(Exhaustive init)",
        "color": "#000000",
        "marker": "^",
        "linestyle": "-",
    },
}


def load_data(path: Path) -> dict[str, dict[str, float]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Figure 4 data file not found: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must contain an object of method series")
    data: dict[str, dict[str, float]] = {}
    for method, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"{source} series {method!r} must be an object")
        try:
            data[str(method)] = {str(x): float(y) for x, y in values.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} series {method!r} must contain numeric rates") from exc
    return data


def save_data(data: dict, path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def series_for_plot(data: dict) -> tuple[str, ...]:
    return tuple(method for method in METHODS if data.get(method))


def run_experiment(
    config: SystemConfig | None = None,
    m_t_values: tuple[int, ...] = (2, 3, 4, 5, 6),
    num_samples: int = 20,
    seed_base: int = 42,
) -> dict[str, dict[str, float]]:
    base = config or SystemConfig()
    data = {method: {} for method in METHODS}
    for m_t in m_t_values:
        cfg = replace(base, m_t=int(m_t))
        report = run_method_suite(
            cfg,
            tuple(RUNNER_NAMES[method] for method in METHODS),
            num_samples,
            seed_base,
        )
        for method in METHODS:
            data[method][str(m_t)] = float(
                report["series"][RUNNER_NAMES[method]]["mean"]
            )
    return data


def plot(data: dict, output_prefix: Path) -> tuple[Path, Path]:
    methods = series_for_plot(data)
    if not methods:
        raise ValueError("Figure 4 data contain no PGD-initialization series")
    x_values = sorted({int(x) for method in methods for x in data[method]})
    with style_context():
        figure, axis = plt.subplots(figsize=(5.0, 4.0), dpi=300)
        for method in methods:
            method_x = [x for x in x_values if str(x) in data[method]]
            method_y = [float(data[method][str(x)]) for x in method_x]
            axis.plot(
                method_x,
                method_y,
                label=STYLES[method]["label"],
                color=STYLES[method]["color"],
                marker=STYLES[method]["marker"],
                linestyle=STYLES[method]["linestyle"],
                linewidth=1.2,
                markersize=5.5,
            )
        axis.set_xlabel(r"$M_T$", fontsize=16)
        axis.set_ylabel(r"$R_s$ (bps/Hz)", fontsize=16)
        if len(x_values) > 1:
            axis.set_xlim(min(x_values), max(x_values))
        axis.set_xticks(x_values)
        axis.tick_params(axis="both", labelsize=16)
        axis.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        axis.legend(
            fontsize=10,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=2,
            frameon=True,
            borderaxespad=0.2,
            borderpad=0.2,
            labelspacing=0.2,
            handletextpad=0.4,
            columnspacing=0.7,
            handlelength=2.1,
        )
        figure.tight_layout()
        paths = save_figure(figure, Path(output_prefix))
        plt.close(figure)
    return paths


def plot_existing(data_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    return plot(load_data(Path(data_path)), Path(output_prefix))
