"""Transmit-power sweep for manuscript Figure 5."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from code.common.config import SystemConfig
from code.experiments.runner import run_method_suite
from code.plotting.paper_style import save_figure, style_context


METHODS = (
    "CEO",
    "Greedy",
    "Exhaustive",
    "Random",
    "FPA",
    "PGD",
    "PGD(Exhaustive init)",
)
RUNNER_NAMES = {
    "CEO": "Proposed CEO",
    "Greedy": "Proposed Greedy",
    "Exhaustive": "Exhaustive",
    "Random": "Random",
    "FPA": "FPA",
    "PGD": "PGD",
    "PGD(Exhaustive init)": "UB",
}
STYLES = {
    "CEO": {"label": "Proposed CEO", "color": "#D62728", "marker": "o", "linestyle": "-"},
    "Greedy": {"label": "Proposed Greedy", "color": "#1F77B4", "marker": "s", "linestyle": "-"},
    "Exhaustive": {"label": "Exhaustive", "color": "#4D4D4D", "marker": "^", "linestyle": "-"},
    "Random": {"label": "Random", "color": "#FF7F0E", "marker": "D", "linestyle": "-"},
    "FPA": {"label": "FPA", "color": "#2CA02C", "marker": "v", "linestyle": "-"},
    "PGD": {"label": "PGD", "color": "#CC79A7", "marker": "d", "linestyle": "--"},
    "PGD(Exhaustive init)": {
        "label": "PGD(Exhaustive init)",
        "color": "#000000",
        "marker": "X",
        "linestyle": "--",
    },
}
DEFAULT_POWER_DBM = (-5, 0, 5, 10, 15, 20, 25)


def load_data(path: Path) -> dict[str, dict[str, float]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Figure 5 data file not found: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must contain an object of method series")
    raw_data = raw.get("data", raw)
    if not isinstance(raw_data, dict):
        raise ValueError(f"{source} data must be an object of method series")
    data: dict[str, dict[str, float]] = {}
    for method, values in raw_data.items():
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
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def series_for_plot(data: dict) -> tuple[str, ...]:
    return tuple(method for method in METHODS if data.get(method))


def x_values(data: dict) -> list[int]:
    return sorted({int(x) for method in series_for_plot(data) for x in data[method]})


def _config_for_power(base: SystemConfig, power_dbm: float) -> SystemConfig:
    p_max = 10.0 ** ((power_dbm - 30.0) / 10.0)
    return replace(base, p_max=p_max)


def run_experiment(
    config: SystemConfig | None = None,
    power_dbm_values: tuple[int, ...] = DEFAULT_POWER_DBM,
    num_samples: int = 50,
    seed_base: int = 42,
) -> dict[str, dict[str, float]]:
    base = config or SystemConfig()
    data = {method: {} for method in METHODS}
    runner_methods = tuple(RUNNER_NAMES[method] for method in METHODS)
    for power_dbm in power_dbm_values:
        report = run_method_suite(
            _config_for_power(base, float(power_dbm)),
            runner_methods,
            num_samples,
            seed_base,
        )
        for method in METHODS:
            data[method][str(power_dbm)] = float(
                report["series"][RUNNER_NAMES[method]]["mean"]
            )
    return data


def plot(data: dict, output_prefix: Path) -> tuple[Path, Path]:
    methods = series_for_plot(data)
    powers = x_values(data)
    if not methods or not powers:
        raise ValueError("Figure 5 data contain no supported method series")
    with style_context():
        figure, axis = plt.subplots(figsize=(5.0, 4.0), dpi=300)
        for method in methods:
            method_x = [x for x in powers if str(x) in data[method]]
            method_y = [float(data[method][str(x)]) for x in method_x]
            style = STYLES[method]
            axis.plot(
                method_x,
                method_y,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.4,
                markersize=5.5,
            )
        axis.set_xlabel(r"$P_{\max}$ (dBm)", fontsize=16)
        axis.set_ylabel(r"$R_s$ (bps/Hz)", fontsize=16)
        if len(powers) > 1:
            axis.set_xlim(min(powers), max(powers))
        axis.set_xticks(powers)
        axis.tick_params(axis="both", labelsize=16)
        axis.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        axis.legend(
            fontsize=12,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.06),
            ncol=2,
            frameon=True,
            borderaxespad=0.2,
            borderpad=0.2,
            labelspacing=0.2,
            handletextpad=0.4,
            columnspacing=0.8,
            handlelength=1.8,
        )
        figure.tight_layout()
        paths = save_figure(figure, Path(output_prefix))
        plt.close(figure)
    return paths


def plot_existing(data_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    return plot(load_data(Path(data_path)), Path(output_prefix))
