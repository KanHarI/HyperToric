"""Integration smoke tests: full pipeline on small configs, NaN provenance."""

from __future__ import annotations

import numpy as np
import pytest

from hypertoric.config import (
    IOConfig,
    NeuronConfig,
    SimConfig,
    STDPConfig,
    TorusConfig,
    TrainingConfig,
)
from hypertoric.io import IOManager
from hypertoric.simulator import Simulator
from hypertoric.task import TargetTracking1D

SEED = 42


def _make_training() -> TrainingConfig:
    return TrainingConfig(
        game_tick_steps=50,
        ticks_per_epoch=20,
        max_epochs=1,
    )


def _io_for_ndim(ndim: int) -> IOConfig:
    """Build an IOConfig whose motor_position is valid for the given grid_size=2."""
    return IOConfig(
        sensory_axis=0,
        sensory_position=0,
        motor_axis=0,
        motor_position=1,
        sensory_cluster_size=4,
        motor_cluster_size=4,
    )


_CONFIGS = [
    pytest.param(
        SimConfig(
            torus=TorusConfig(ndim=2, grid_size=2, neurons_per_block=8),
            neuron=NeuronConfig(dt=0.5),
            io=_io_for_ndim(2),
            training=_make_training(),
            seed=SEED,
        ),
        id="minimal_2d",
    ),
    pytest.param(
        SimConfig(
            torus=TorusConfig(ndim=3, grid_size=2, neurons_per_block=16),
            neuron=NeuronConfig(dt=0.5),
            io=_io_for_ndim(3),
            training=_make_training(),
            seed=SEED,
        ),
        id="small_3d",
    ),
    pytest.param(
        SimConfig(
            torus=TorusConfig(ndim=4, grid_size=2, neurons_per_block=8),
            neuron=NeuronConfig(dt=0.5),
            io=_io_for_ndim(4),
            training=_make_training(),
            seed=SEED,
        ),
        id="tiny_4d",
    ),
]


class TestSmoke:
    @pytest.mark.integration
    @pytest.mark.timeout(10)
    @pytest.mark.parametrize("cfg", _CONFIGS)
    def test_full_pipeline_no_crash(self, ti_runtime: None, cfg: SimConfig) -> None:
        """Full pipeline runs without exceptions on small configs."""
        sim = Simulator(cfg)
        io_mgr = IOManager(sim.topology, cfg.io, cfg.torus.neurons_per_block, cfg.seed)
        task = TargetTracking1D(cfg.training, seed=cfg.seed)
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

        # All fields finite
        v = sim.read_voltage()
        assert np.all(np.isfinite(v)), "NaN/Inf in voltage"

        # Membrane potentials bounded (spike reset at 30)
        assert np.all(v <= 30.0), f"Voltage exceeded 30: max={np.max(v)}"

        assert sim.step_count > 0
        # At least check that the simulation ran the expected number of steps
        expected_steps = cfg.training.ticks_per_epoch * cfg.training.game_tick_steps
        assert sim.step_count == expected_steps


class TestNaNProvenance:
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.timeout(30)
    def test_nan_provenance_step_by_step(self, ti_runtime: None) -> None:
        """Run step-by-step checking for NaN after each major phase."""
        cfg = SimConfig(
            torus=TorusConfig(ndim=2, grid_size=2, neurons_per_block=16),
            neuron=NeuronConfig(dt=0.5, excitatory_ratio=0.9),
            stdp=STDPConfig(inter_mode="all"),
            io=_io_for_ndim(2),
            training=_make_training(),
            seed=SEED,
        )

        sim = Simulator(cfg)
        io_mgr = IOManager(sim.topology, cfg.io, cfg.torus.neurons_per_block, cfg.seed)

        # Inject sensory input
        io_mgr.encode_sensory(0.5, sim.fields)

        for step in range(500):
            sim.step()

            if step % 50 == 0:
                v = sim.fields.v.to_numpy()
                assert np.all(np.isfinite(v)), f"NaN/Inf in v at step {step}"
                u = sim.fields.u.to_numpy()
                assert np.all(np.isfinite(u)), f"NaN/Inf in u at step {step}"
                i_syn = sim.fields.I_syn.to_numpy()
                assert np.all(np.isfinite(i_syn)), f"NaN/Inf in I_syn at step {step}"
                w_intra = sim.fields.W_intra.to_numpy()
                assert np.all(np.isfinite(w_intra)), (
                    f"NaN/Inf in W_intra at step {step}"
                )
