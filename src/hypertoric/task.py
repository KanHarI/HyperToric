"""Task abstraction: target tracking with difficulty progression."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from hypertoric.config import TrainingConfig


class Task(Protocol):
    """Structural protocol for task implementations."""

    def reset(self) -> None: ...

    def get_target(self) -> float: ...

    def step(self, action: float) -> float: ...


class TargetTracking1D:
    """1D target tracking with 4 difficulty levels.

    Levels:
        0 - Static: target fixed at num_positions // 2
        1 - Slow step: target jumps to random position every step_interval ticks
        2 - Slow ramp: target moves ±1 every ramp_interval ticks (bouncing)
        3 - Sine wave: target follows a sine wave with configurable period
    """

    def __init__(self, training_config: TrainingConfig, seed: int = 42) -> None:
        self._cfg = training_config
        self._num_positions = training_config.num_positions
        self._rng = np.random.default_rng(seed)

        self._target: int = 0
        self._cursor: int = 0
        self._tick: int = 0
        self._level: int = 0
        self._ramp_direction: int = 1

        self.reset()

    @property
    def level(self) -> int:
        return self._level

    @level.setter
    def level(self, value: int) -> None:
        if value < 0 or value > 3:
            msg = f"level must be in {{0, 1, 2, 3}}, got {value}"
            raise ValueError(msg)
        self._level = value

    def reset(self) -> None:
        """Reset cursor to 0, tick to 0, set target per level."""
        self._cursor = 0
        self._tick = 0
        self._ramp_direction = 1
        if self._level == 0:
            self._target = self._num_positions // 2
        else:
            self._target = int(self._rng.integers(0, self._num_positions))

    def get_target(self) -> float:
        """Return target position normalized to [0, 1]."""
        return self._target / (self._num_positions - 1)

    def step(self, action: float) -> float:
        """Move cursor by action (clamped), update target, return distance."""
        # Move cursor
        if action > 0:
            self._cursor = min(self._cursor + 1, self._num_positions - 1)
        elif action < 0:
            self._cursor = max(self._cursor - 1, 0)

        self._tick += 1

        # Update target based on level
        if self._level == 1:
            # Slow step: jump to random position every step_interval ticks
            if self._tick % self._cfg.step_interval == 0:
                self._target = int(self._rng.integers(0, self._num_positions))
        elif self._level == 2:
            # Slow ramp: move ±1 every ramp_interval ticks, bouncing at edges
            if self._tick % self._cfg.ramp_interval == 0:
                self._target += self._ramp_direction
                if self._target >= self._num_positions - 1:
                    self._target = self._num_positions - 1
                    self._ramp_direction = -1
                elif self._target <= 0:
                    self._target = 0
                    self._ramp_direction = 1
        elif self._level == 3:
            # Sine wave: target follows sin with configurable period
            t = self._tick / self._cfg.sine_period
            normalized = (math.sin(2.0 * math.pi * t) + 1.0) / 2.0
            self._target = int(round(normalized * (self._num_positions - 1)))

        return float(abs(self._cursor - self._target))
