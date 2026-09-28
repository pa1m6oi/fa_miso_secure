# FA-MISO Secure Communications

This repository contains the code and data for reproducing the results in:

> Miao Jiang, Ruijie Huang, Yiqing Li, Guangchi Zhang, and Liang Yang, “Position Optimization for FA-Enabled MISO Secure Communications: Discrete Selection or Continuous Optimization?”

## Overview

We study secure communication with movable antennas in a multiple-input single-output (MISO) system. This release provides the channel model, discrete port-selection methods, continuous position optimization, baselines, and scripts for reproducing Figs. 2–6.

## Methods

| Component | Description | Location |
| --- | --- | --- |
| Channel model | Bessel-correlated multipath channel following the manuscript | [`code/common/channel.py`](code/common/channel.py) |
| CEO | Cross-entropy optimization for discrete port selection | [`code/solvers/ceo.py`](code/solvers/ceo.py) |
| Greedy | Greedy discrete port selection | [`code/solvers/greedy.py`](code/solvers/greedy.py) |
| Exhaustive search | Exhaustive discrete port-selection baseline | [`code/solvers/exhaustive.py`](code/solvers/exhaustive.py) |
| Random search | Random discrete port-selection baseline | [`code/solvers/random_search.py`](code/solvers/random_search.py) |
| FPA | Fixed-position antenna baseline | [`code/solvers/fpa.py`](code/solvers/fpa.py) |
| PGD | Continuous position optimization, including multiple initializations | [`code/solvers/pgd.py`](code/solvers/pgd.py) |
| Figure experiments | Experiment setup and plotting for Figs. 2–6 | [`code/experiments/`](code/experiments/) |

## Repository structure

```text
github_release/
├── code/
│   ├── common/       # Channel, geometry, beamforming, metrics, and configuration
│   ├── experiments/  # Figure experiments and shared runner
│   ├── plotting/     # Paper plotting style
│   └── solvers/      # CEO, greedy, exhaustive, random, FPA, and PGD
├── results/          # Existing JSON data for Figs. 2–6
├── scripts/          # Figure reproduction and demo entry points
├── tests/            # Unit and smoke tests
├── outputs/           # Generated figures
├── requirements.txt
└── LICENSE
```

## Requirements

The release was tested with Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1, and Matplotlib 3.9.2.

```bash
python -m pip install -r requirements.txt
```

## Quick start

Run a small end-to-end demo:

```bash
python scripts/run_demo.py --num-samples 1 --grid-size 3 --m-t 2 --methods FPA
```

To reproduce the paper plots from the included data, run (for example):

```bash
python scripts/reproduce_fig2.py
```

This loads the existing JSON data and writes the figure to `outputs/`; it does not rerun the optimization experiments.

## Figure reproduction

Run the corresponding script for each figure:

```bash
python scripts/reproduce_fig2.py
python scripts/reproduce_fig3.py
python scripts/reproduce_fig4.py
python scripts/reproduce_fig5.py
python scripts/reproduce_fig6.py
```

The scripts use the JSON files in `results/` by default. Fig. 2 uses `fig2_L.json`, `fig2_rho.json`, and `fig2_varsigma.json`; Figs. 3 and 4 use `fig3_fig4.json`; Figs. 5 and 6 use `fig5.json` and `fig6.json`, respectively. Generated plots are saved under `outputs/` in PNG and PDF formats. Existing JSON data are retained as provided and are not overwritten by default.

To explicitly rerun an experiment, add `--recompute`. For a small smoke run, for example:

```bash
python scripts/reproduce_fig5.py --recompute --quick --num-samples 1 --seed-base 42 --output-prefix outputs/smoke_fig5
```

## Channel model

The implementation in `code/common/channel.py` follows the paper’s Bessel-correlation model in Eqs. (19)–(20). For continuous PGD positions between receive-grid ports, the sampled uncorrelated port field is bilinearly interpolated before applying the distance-based Bessel correlation.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
