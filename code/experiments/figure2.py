"""Load, recompute, and plot manuscript Figure 2."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from code.common.channel import discrete_channel_matrices, sample_channel
from code.common.config import CeoConfig, SystemConfig
from code.common.geometry import generate_grid
from code.solvers.ceo import solve_ceo
from code.solvers.context import DiscreteProblem
from code.plotting.paper_style import save_figure, style_context


EXPECTED_KEYS = ("meta", "sample_size", "elite_ratio", "smoothing")
CURVE_STYLES = (
    {"color": "#FF0101", "marker": "v", "linestyle": "-"},
    {"color": "#0072B2", "marker": "s", "linestyle": (0, (6, 2))},
    {"color": "#009E73", "marker": "o", "linestyle": "-."},
    {"color": "#CC79A7", "marker": "D", "linestyle": ":"},
    {"color": "#000000", "marker": "X", "linestyle": (0, (8, 2, 2, 2))},
)


def load_data(path: Path) -> dict:
    """Load one sensitivity JSON and validate its public top-level schema."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Figure 2 data file not found: {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc
    missing = [key for key in EXPECTED_KEYS if key not in data]
    if missing:
        expected = ", ".join(EXPECTED_KEYS)
        raise ValueError(f"{source} must contain expected keys: {expected}")
    if not isinstance(data["meta"], dict):
        raise ValueError(f"{source} key 'meta' must be an object")
    for key in EXPECTED_KEYS[1:]:
        if not isinstance(data[key], list):
            raise ValueError(f"{source} key {key!r} must be a list")
    return data


def save_data(data: dict, path: Path) -> None:
    """Write newly computed data without touching the shipped results."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _pad_history(history: tuple[object, ...], length: int) -> list[dict]:
    records = [dict(item) for item in history]
    if not records:
        raise RuntimeError("CEO returned no convergence history")
    while len(records) < length:
        repeated = dict(records[-1])
        repeated["iteration"] = len(records) + 1
        records.append(repeated)
    return records[:length]


def _convergence_record(
    parameter: str,
    value: int | float,
    config: SystemConfig,
    num_samples: int,
    seed_base: int,
) -> dict:
    num_positions = config.grid_size * config.grid_size
    if parameter == "sample_size":
        if int(value) % num_positions:
            raise ValueError("sample_size must be an integer multiple of grid positions")
        ceo = replace(config.ceo, sample_factor=int(value) // num_positions)
    elif parameter == "smoothing":
        ceo = replace(config.ceo, smoothing=float(value))
    elif parameter == "elite_ratio":
        ceo = replace(config.ceo, elite_ratio=float(value))
    else:
        raise ValueError(f"unsupported Figure 2 parameter: {parameter}")
    run_config = replace(config, ceo=ceo)

    histories: list[list[dict]] = []
    final_rates: list[float] = []
    evaluations: list[int] = []
    iterations: list[int] = []
    for sample_index in range(num_samples):
        seed = seed_base + sample_index
        realization = sample_channel(np.random.default_rng(seed), run_config)
        h_b, h_e = discrete_channel_matrices(
            realization,
            run_config,
            generate_grid(run_config.grid_size),
        )
        result = solve_ceo(
            DiscreteProblem(h_b, h_e, run_config),
            np.random.default_rng(np.random.SeedSequence([seed, 0])),
        )
        histories.append(_pad_history(result.history, ceo.max_iterations))
        final_rates.append(result.secrecy_rate)
        evaluations.append(result.evaluations)
        iterations.append(result.iterations)

    curve_names = (
        "iteration_best",
        "global_best",
        "elite_mean",
        "sample_mean",
    )
    curves: dict[str, list[float] | list[int]] = {
        "iteration": list(range(1, ceo.max_iterations + 1))
    }
    for name in curve_names:
        matrix = np.asarray(
            [[float(record[name]) for record in history] for history in histories]
        )
        curves[f"{name}_mean"] = np.mean(matrix, axis=0).tolist()
        curves[f"{name}_std"] = np.std(matrix, axis=0).tolist()
    return {
        "parameter": parameter,
        "value": value,
        "summary": {
            "m_t": run_config.m_t,
            "num_samples": num_samples,
            "grid_size": run_config.grid_size,
            "num_eves": run_config.num_eavesdroppers,
            "ceo_max_iter": ceo.max_iterations,
            "sample_size": ceo.sample_factor * num_positions,
            "elite_ratio": ceo.elite_ratio,
            "smoothing": ceo.smoothing,
            "tol": ceo.tolerance,
            "avg_final_objective": float(np.mean(final_rates)),
            "std_final_objective": float(np.std(final_rates)),
            "avg_iterations": float(np.mean(iterations)),
            "avg_evaluations": float(np.mean(evaluations)),
        },
        "curves": curves,
    }


def _sweep_report(
    parameter: str,
    values: list[int] | list[float],
    config: SystemConfig,
    num_samples: int,
    seed_base: int,
) -> dict:
    records = [
        _convergence_record(parameter, value, config, num_samples, seed_base)
        for value in values
    ]
    return {
        "meta": {
            "m_t": config.m_t,
            "num_samples": num_samples,
            "grid_size": config.grid_size,
            "num_eves": config.num_eavesdroppers,
            "ceo_max_iter": config.ceo.max_iterations,
            "ceo_tol": config.ceo.tolerance,
            "seed_base": seed_base,
        },
        "sample_size": records if parameter == "sample_size" else [],
        "elite_ratio": records if parameter == "elite_ratio" else [],
        "smoothing": records if parameter == "smoothing" else [],
    }


def run_experiment(
    config: SystemConfig | None = None,
    num_samples: int = 20,
    seed_base: int = 42,
    quick: bool = False,
) -> dict[str, dict]:
    """Compute new Bessel sensitivity data; full published data are not implicit."""
    cfg = config or SystemConfig()
    positions = cfg.grid_size * cfg.grid_size
    sample_sizes = [positions] if quick else [positions * factor for factor in range(1, 6)]
    smoothing = [cfg.ceo.smoothing] if quick else [0.2, 0.3, 0.4, 0.5, 0.6]
    elite = [cfg.ceo.elite_ratio] if quick else [0.05, 0.1, 0.15, 0.2, 0.25]
    return {
        "L": _sweep_report("sample_size", sample_sizes, cfg, num_samples, seed_base),
        "rho": _sweep_report("smoothing", smoothing, cfg, num_samples, seed_base),
        "varsigma": _sweep_report("elite_ratio", elite, cfg, num_samples, seed_base),
    }


def _format_label(parameter: str, value: int | float, report: dict) -> str:
    if parameter == "sample_size":
        positions = int(report["meta"]["grid_size"]) ** 2
        ratio = float(value) / positions
        if np.isclose(ratio, 1.0):
            return r"$L=N_T$"
        if np.isclose(ratio, round(ratio)):
            return rf"$L={int(round(ratio))}N_T$"
        return rf"$L={value}$"
    if parameter == "smoothing":
        return rf"$\rho={value}$"
    return rf"$\varsigma={value}$"


def _plot_panel(axis, report: dict, parameter: str, title: str) -> tuple[float, float]:
    minimum = np.inf
    maximum = -np.inf
    for index, record in enumerate(report[parameter]):
        curve = record["curves"]
        x_values = np.asarray(curve["iteration"], dtype=int)
        y_values = np.asarray(curve["global_best_mean"], dtype=float)
        mask = x_values <= 20
        x_values, y_values = x_values[mask], y_values[mask]
        style = CURVE_STYLES[index % len(CURVE_STYLES)]
        axis.plot(
            x_values,
            y_values,
            label=_format_label(parameter, record["value"], report),
            linewidth=1.7,
            markersize=5.5,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markevery=max(1, len(x_values) // 4),
            **style,
        )
        minimum = min(minimum, float(np.min(y_values)))
        maximum = max(maximum, float(np.max(y_values)))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError(f"Figure 2 report has no curves for {parameter}")
    axis.set_title(title, fontsize=16, pad=6)
    axis.set_xlim(1, 20)
    axis.set_xticks([5, 10, 15, 20])
    axis.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    axis.tick_params(axis="both", labelsize=16)
    axis.legend(fontsize=14, loc="lower right", frameon=True)
    return minimum, maximum


def plot(data: dict[str, dict], output_prefix: Path) -> tuple[Path, Path]:
    """Render the three sensitivity panels as manuscript Figure 2."""
    with style_context():
        figure, axes = plt.subplots(1, 3, figsize=(8.6, 6.2), dpi=300, sharey=True)
        specifications = (
            (axes[0], data["L"], "sample_size", r"(a) Different $L$"),
            (axes[1], data["rho"], "smoothing", r"(b) Different $\rho$"),
            (axes[2], data["varsigma"], "elite_ratio", r"(c) Different $\varsigma$"),
        )
        limits = [
            _plot_panel(axis, report, parameter, title)
            for axis, report, parameter, title in specifications
        ]
        y_min = min(item[0] for item in limits)
        y_max = max(item[1] for item in limits)
        margin = max(0.05, 0.05 * (y_max - y_min))
        axes[0].set_ylim(y_min - margin, y_max + margin)
        axes[0].set_ylabel(r"$R_s$ (bps/Hz)", fontsize=24)
        figure.supxlabel("The number of iterations", fontsize=24, y=0.02)
        figure.tight_layout(rect=(0.02, 0.03, 1.0, 1.0), w_pad=0.45)
        paths = save_figure(figure, Path(output_prefix))
        plt.close(figure)
    return paths


def plot_existing(results_dir: Path, output_prefix: Path) -> tuple[Path, Path]:
    """Plot shipped JSON directly, without calling any solver."""
    root = Path(results_dir)
    data = {
        "L": load_data(root / "fig2_L.json"),
        "rho": load_data(root / "fig2_rho.json"),
        "varsigma": load_data(root / "fig2_varsigma.json"),
    }
    return plot(data, Path(output_prefix))
