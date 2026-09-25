"""Immutable configuration for the FA-MISO experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CeoConfig:
    sample_factor: int = 3
    elite_ratio: float = 0.1
    smoothing: float = 0.3
    max_iterations: int = 20
    tolerance: float = 1e-6


@dataclass(frozen=True)
class GreedyConfig:
    max_iterations: int = 10
    tolerance: float = 1e-6


@dataclass(frozen=True)
class PgdConfig:
    iterations: int = 60
    initial_step: float = 0.05
    minimum_step: float = 1e-4
    gradient_epsilon: float = 5e-4


@dataclass(frozen=True)
class SystemConfig:
    wavelength: float = 0.06
    d_bob: float = 100.0
    d_eve: float = 100.0
    num_eavesdroppers: int = 3
    num_bob_paths: int = 16
    num_eve_paths: int = 16
    path_loss_db: float = -30.0
    path_loss_exponent: float = 3.7
    p_max: float = 0.1
    noise_power: float = 1.26e-13
    d_min: float = 0.03
    area_half_width: float = 0.06
    grid_size: int = 7
    m_t: int = 4
    ceo: CeoConfig = field(default_factory=CeoConfig)
    greedy: GreedyConfig = field(default_factory=GreedyConfig)
    pgd: PgdConfig = field(default_factory=PgdConfig)

    def validate(self) -> None:
        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive")
        if self.d_bob <= 0 or self.d_eve <= 0:
            raise ValueError("link distances must be positive")
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if self.m_t < 1:
            raise ValueError("m_t must be positive")
        if self.num_eavesdroppers < 1:
            raise ValueError("num_eavesdroppers must be positive")
        if self.num_bob_paths < 1 or self.num_eve_paths < 1:
            raise ValueError("channel path counts must be positive")
        if self.p_max <= 0 or self.noise_power <= 0:
            raise ValueError("p_max and noise_power must be positive")
        if self.d_min < 0:
            raise ValueError("d_min must be nonnegative")
        if self.area_half_width <= 0:
            raise ValueError("area_half_width must be positive")
        if self.ceo.sample_factor < 1:
            raise ValueError("CEO sample_factor must be positive")
        if not 0.0 < self.ceo.elite_ratio <= 1.0:
            raise ValueError("CEO elite_ratio must lie in (0, 1]")
        if not 0.0 <= self.ceo.smoothing <= 1.0:
            raise ValueError("CEO smoothing must lie in [0, 1]")
        if self.ceo.max_iterations < 1 or self.greedy.max_iterations < 1:
            raise ValueError("algorithm iteration counts must be positive")
        if self.ceo.tolerance < 0 or self.greedy.tolerance < 0:
            raise ValueError("algorithm tolerances must be nonnegative")
        if self.pgd.iterations < 1:
            raise ValueError("PGD iterations must be positive")
        if self.pgd.initial_step <= 0 or self.pgd.minimum_step <= 0:
            raise ValueError("PGD step sizes must be positive")
        if self.pgd.minimum_step > self.pgd.initial_step:
            raise ValueError("PGD minimum_step cannot exceed initial_step")
        if self.pgd.gradient_epsilon <= 0:
            raise ValueError("PGD gradient_epsilon must be positive")
        self.validate_geometry()

    def validate_geometry(self) -> None:
        from code.common.geometry import has_feasible_selection

        if not has_feasible_selection(self):
            raise ValueError("minimum spacing leaves no feasible transmit-port set")
