"""Grid construction and antenna-position feasibility checks."""

from __future__ import annotations

from itertools import combinations
from typing import Protocol

import numpy as np


class GeometryConfig(Protocol):
    grid_size: int
    m_t: int
    d_min: float
    area_half_width: float


def generate_grid(grid_size: int) -> np.ndarray:
    """Return the row-major normalized square grid in ``[-1, 1]^2``."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    axis = np.linspace(-1.0, 1.0, grid_size)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack((x_grid.ravel(), y_grid.ravel()))


def physical_grid(config: GeometryConfig) -> np.ndarray:
    """Map the normalized port grid to physical offsets in metres."""
    if config.area_half_width <= 0:
        raise ValueError("area_half_width must be positive")
    return generate_grid(config.grid_size) * float(config.area_half_width)


def index_to_normalized(index: int, grid_size: int) -> np.ndarray:
    """Map a row-major grid index to its normalized two-dimensional point."""
    points = generate_grid(grid_size)
    if not 0 <= int(index) < len(points):
        raise ValueError("index is outside the port grid")
    return points[int(index)].copy()


def fixed_port_indices(m_t: int, grid_size: int) -> tuple[tuple[int, ...], int]:
    """Return the paper's fixed transmit selection and top-left receive port."""
    if m_t < 1:
        raise ValueError("m_t must be positive")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    count = grid_size * grid_size
    if m_t > count:
        raise ValueError("m_t exceeds the number of grid positions")

    top_left = 0
    top_right = grid_size - 1
    bottom_left = grid_size * (grid_size - 1)
    bottom_right = count - 1
    top_center = grid_size // 2
    bottom_center = bottom_left + grid_size // 2
    preferred = (
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        top_center,
        bottom_center,
    )
    if m_t <= len(preferred):
        return tuple(preferred[:m_t]), top_left
    remaining = tuple(index for index in range(count) if index not in preferred)
    return tuple((*preferred, *remaining[: m_t - len(preferred)])), top_left


def is_feasible_indices(
    indices: tuple[int, ...] | list[int],
    points: np.ndarray,
    d_min: float,
) -> bool:
    """Check bounds, uniqueness, and pairwise minimum spacing."""
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("points must have shape (num_positions, 2)")
    if d_min < 0:
        raise ValueError("d_min must be nonnegative")
    selected = tuple(int(index) for index in indices)
    if len(set(selected)) != len(selected):
        return False
    if any(index < 0 or index >= len(point_array) for index in selected):
        return False
    for left, right in combinations(selected, 2):
        if np.linalg.norm(point_array[left] - point_array[right]) + 1e-12 < d_min:
            return False
    return True


def is_feasible_continuous(x: np.ndarray, m_t: int, config: GeometryConfig) -> bool:
    """Check normalized continuous transmit positions and optional receiver."""
    values = np.asarray(x, dtype=float).reshape(-1)
    if m_t < 1 or values.size not in (2 * m_t, 2 * m_t + 2):
        return False
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1.0 + 1e-12):
        return False
    tx_positions = values[: 2 * m_t].reshape(m_t, 2) * config.area_half_width
    for left, right in combinations(range(m_t), 2):
        if np.linalg.norm(tx_positions[left] - tx_positions[right]) + 1e-12 < config.d_min:
            return False
    return True


def has_feasible_selection(config: GeometryConfig) -> bool:
    """Return whether at least one ``m_t``-port subset satisfies spacing."""
    if config.grid_size < 2 or config.m_t < 1:
        return False
    points = physical_grid(config)
    if config.m_t > len(points):
        return False

    compatible = np.ones((len(points), len(points)), dtype=bool)
    for left, right in combinations(range(len(points)), 2):
        valid = np.linalg.norm(points[left] - points[right]) + 1e-12 >= config.d_min
        compatible[left, right] = compatible[right, left] = valid

    def search(start: int, selected: tuple[int, ...]) -> bool:
        needed = config.m_t - len(selected)
        if needed == 0:
            return True
        last_start = len(points) - needed
        for candidate in range(start, last_start + 1):
            if all(compatible[candidate, previous] for previous in selected):
                if search(candidate + 1, (*selected, candidate)):
                    return True
        return False

    return search(0, ())
