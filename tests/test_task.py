"""Tests for the Task abstraction and training loop smoke test."""

from __future__ import annotations

import numpy as np
import pytest

from hypertoric.config import (
    IOConfig,
    NeuronConfig,
    SimConfig,
    TorusConfig,
    TrainingConfig,
)
from hypertoric.task import TargetTracking1D


def _make_training(
    **overrides: object,
) -> TrainingConfig:
    defaults: dict[str, object] = {
        "num_positions": 8,
        "step_interval": 100,
        "ramp_interval": 20,
        "sine_period": 200.0,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


class TestTargetTracking1D:
    def test_static_target_no_movement(self) -> None:
        """Step with action=0 doesn't move cursor; distance stays constant."""
        tcfg = _make_training()
        task = TargetTracking1D(tcfg, seed=1)
        task.level = 0
        task.reset()

        d0 = task.step(0)
        d1 = task.step(0)
        d2 = task.step(0)
        assert d0 == d1 == d2

    def test_cursor_clamp_low(self) -> None:
        """Cursor at 0 stays at 0 when action is -1."""
        tcfg = _make_training()
        task = TargetTracking1D(tcfg, seed=1)
        task.level = 0
        task.reset()
        # Cursor starts at 0
        task.step(-1)
        task.step(-1)
        # Still at 0 — distance should be same as initial
        d = task.step(0)
        assert d == float(tcfg.num_positions // 2)

    def test_cursor_clamp_high(self) -> None:
        """Cursor at max stays at max when action is +1."""
        tcfg = _make_training(num_positions=4)
        task = TargetTracking1D(tcfg, seed=1)
        task.level = 0
        task.reset()
        # Move cursor to max (3)
        for _ in range(10):
            task.step(1)
        # Should stay at max=3, target at 2 (4//2)
        d = task.step(1)
        assert d == 1.0

    def test_distance_calculation(self) -> None:
        """Known target and cursor should give correct distance."""
        tcfg = _make_training(num_positions=8)
        task = TargetTracking1D(tcfg, seed=1)
        task.level = 0
        task.reset()
        # Target is at 4 (8//2), cursor starts at 0
        # Move cursor to 2
        task.step(1)
        d = task.step(1)
        assert d == 2.0  # |2 - 4| = 2

    def test_get_target_in_unit_range(self) -> None:
        """get_target() returns value in [0, 1]."""
        tcfg = _make_training()
        task = TargetTracking1D(tcfg, seed=1)
        for level in range(4):
            task.level = level
            task.reset()
            for _ in range(50):
                t = task.get_target()
                assert 0.0 <= t <= 1.0
                task.step(0)

    def test_slow_step_target_changes(self) -> None:
        """Level 1: target changes after step_interval ticks."""
        tcfg = _make_training(step_interval=10, num_positions=8)
        task = TargetTracking1D(tcfg, seed=42)
        task.level = 1
        task.reset()

        # Record initial target
        initial_target = task.get_target()

        # Step through enough ticks that at least one jump occurs
        targets_seen = {initial_target}
        for _ in range(100):
            task.step(0)
            targets_seen.add(task.get_target())

        # Should have seen more than one target
        assert len(targets_seen) > 1

    def test_ramp_bounces(self) -> None:
        """Level 2: target ramps up and bounces at edges."""
        tcfg = _make_training(ramp_interval=1, num_positions=4)
        task = TargetTracking1D(tcfg, seed=1)
        task.level = 2
        task.reset()

        targets: list[float] = []
        for _ in range(10):
            task.step(0)
            targets.append(task.get_target())

        # Should see target move and stay in [0, 1]
        assert all(0.0 <= t <= 1.0 for t in targets)
        # Should see at least 2 different values
        assert len(set(targets)) > 1

    def test_sine_wave(self) -> None:
        """Level 3: target follows sine wave."""
        tcfg = _make_training(sine_period=20.0, num_positions=8)
        task = TargetTracking1D(tcfg, seed=1)
        task.level = 3
        task.reset()

        targets: list[float] = []
        for _ in range(40):
            task.step(0)
            targets.append(task.get_target())

        # All in [0, 1]
        assert all(0.0 <= t <= 1.0 for t in targets)
        # Should see variation (sine moves)
        assert max(targets) > min(targets)

    def test_reset(self) -> None:
        """Reset sets tick=0, cursor=0."""
        tcfg = _make_training()
        task = TargetTracking1D(tcfg, seed=1)
        task.level = 0
        task.reset()

        # Move cursor
        task.step(1)
        task.step(1)
        task.step(1)

        task.reset()
        # After reset, cursor=0, target=num_positions//2
        d = task.step(0)
        assert d == float(tcfg.num_positions // 2)

    def test_level_validation(self) -> None:
        """Setting invalid level raises ValueError."""
        tcfg = _make_training()
        task = TargetTracking1D(tcfg, seed=1)
        with pytest.raises(ValueError, match="level"):
            task.level = 5
        with pytest.raises(ValueError, match="level"):
            task.level = -1


class TestTrainingLoopSmoke:
    """Smoke test: wire Simulator + IOManager + Task together."""

    @pytest.mark.timeout(60)
    def test_smoke_tiny(self, ti_runtime: None) -> None:
        cfg = SimConfig(
            torus=TorusConfig(ndim=2, grid_size=2, neurons_per_block=8),
            neuron=NeuronConfig(dt=0.5),
            io=IOConfig(
                sensory_axis=0,
                sensory_position=0,
                motor_axis=0,
                motor_position=1,
                sensory_cluster_size=4,
                motor_cluster_size=4,
            ),
            training=_make_training(
                game_tick_steps=5,
                ticks_per_epoch=10,
                max_epochs=1,
            ),
        )

        from hypertoric.io import IOManager
        from hypertoric.simulator import Simulator

        sim = Simulator(cfg)
        io_mgr = IOManager(sim.topology, cfg.io, cfg.torus.neurons_per_block, 42)
        task = TargetTracking1D(cfg.training, seed=42)

        dt = cfg.neuron.dt

        task.reset()
        for _tick in range(cfg.training.ticks_per_epoch):
            target_pos = task.get_target()
            io_mgr.encode_sensory(target_pos, sim.fields)

            for _step in range(cfg.training.game_tick_steps):
                sim.step()
                io_mgr.update_motor_rates(sim.fields, dt)

            action = io_mgr.decode_motor()
            distance = task.step(action)
            io_mgr.deliver_feedback(int(distance), sim.fields)

        # No NaN in voltages
        v = sim.read_voltage()
        assert not np.any(np.isnan(v)), "NaN detected in voltages"
        assert not np.any(np.isinf(v)), "Inf detected in voltages"
