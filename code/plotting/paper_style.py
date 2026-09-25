"""Shared typography and figure-saving utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
}


def style_context():
    """Return an isolated Matplotlib context for paper figures."""
    return plt.rc_context(PAPER_RC)


def save_figure(figure, output_prefix: Path) -> tuple[Path, Path]:
    """Save matching high-resolution PNG and vector PDF artifacts."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = prefix.with_suffix(".png")
    pdf_path = prefix.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path
