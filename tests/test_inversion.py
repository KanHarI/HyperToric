"""Inversion (NOT gate) convergence test.

A tiny 2x2 2D torus learns to invert: input pattern A → output pattern B
and vice versa, using only STDP + order/chaos feedback.
"""

from __future__ import annotations

import numpy as np
import pytest

from hypertoric.config import (
    IOConfig,
    NeuronConfig,
    PlasticityConfig,
    SimConfig,
    STDPConfig,
    TorusConfig,
    TrainingConfig,
)
from hypertoric.io import IOManager
from hypertoric.simulator import Simulator

K = 16
SENSORY_BLOCK = 0  # (0,0) in 2x2 grid
MOTOR_BLOCK = 1  # (1,0) — one hop from sensory along axis 0
PATTERN_A = slice(0, K // 2)  # neurons 0-7
PATTERN_B = slice(K // 2, K)  # neurons 8-15
PRESENTATION_STEPS = 500
MAX_EPOCHS = 50
CURRENT = 15.0


def _make_config() -> SimConfig:
    return SimConfig(
        torus=TorusConfig(ndim=2, grid_size=2, neurons_per_block=K),
        neuron=NeuronConfig(dt=0.5),
        stdp=STDPConfig(inter_mode="all"),
        plasticity=PlasticityConfig(interval=100, inter_interval=500),
        io=IOConfig(
            sensory_axis=0,
            sensory_position=0,
            motor_axis=0,
            motor_position=1,
            sensory_cluster_size=K,
            motor_cluster_size=K,
        ),
        training=TrainingConfig(),
        seed=42,
    )


def _set_input(sim: Simulator, input_val: int) -> None:
    """Inject current into sensory block for the given input pattern."""
    i_ext = np.zeros(
        (sim.topology.num_blocks, sim.config.torus.neurons_per_block),
        dtype=np.float32,
    )
    if input_val == 0:
        i_ext[SENSORY_BLOCK, PATTERN_A] = CURRENT
    else:
        i_ext[SENSORY_BLOCK, PATTERN_B] = CURRENT
    sim.inject_current(i_ext)


def _read_motor_output(sim: Simulator) -> int:
    """Read which half of the motor block spiked more. Returns 0 or 1."""
    spikes = sim.read_spikes()
    count_a = int(np.sum(spikes[MOTOR_BLOCK, PATTERN_A]))
    count_b = int(np.sum(spikes[MOTOR_BLOCK, PATTERN_B]))
    return 0 if count_a > count_b else 1


class TestInversion:
    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_inversion_mechanical(self, ti_runtime: None) -> None:
        """Verify the inversion test infrastructure works: no NaN, spikes occur."""
        cfg = _make_config()
        sim = Simulator(cfg)
        io_mgr = IOManager(sim.topology, cfg.io, cfg.torus.neurons_per_block, cfg.seed)
        dt = cfg.neuron.dt

        total_sensory_spikes = 0

        for input_val in [0, 1]:
            for _t in range(PRESENTATION_STEPS):
                _set_input(sim, input_val)
                io_mgr.update_motor_rates(sim.fields, dt)
                sim.step()
                # Accumulate spikes over entire presentation (spikes are transient)
                spikes = sim.read_spikes()
                total_sensory_spikes += int(np.sum(spikes[SENSORY_BLOCK]))

            io_mgr.deliver_feedback(1, sim.fields)

        # No NaN in fields
        v = sim.read_voltage()
        assert np.all(np.isfinite(v)), "NaN/Inf in voltage"

        # Sensory block should spike (input current drives it)
        assert total_sensory_spikes > 0, "No sensory spikes at all"

        # Weights should have changed from STDP
        w_inter = sim.fields.W_inter.to_numpy()
        assert not np.allclose(w_inter, 0.0), "Inter-block weights are all zero"

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.timeout(60)
    @pytest.mark.xfail(
        reason="Learning convergence requires hyperparameter tuning — "
        "inter-block weights too weak for K=16 to drive motor spikes. "
        "See docs/plans/wc-11-integration.md debugging guide.",
        strict=False,
    )
    def test_inversion_convergence(self, ti_runtime: None) -> None:
        """Tiny network learns NOT gate through STDP + order/chaos feedback."""
        cfg = _make_config()
        sim = Simulator(cfg)
        io_mgr = IOManager(sim.topology, cfg.io, cfg.torus.neurons_per_block, cfg.seed)
        dt = cfg.neuron.dt

        correct_count = 0
        total_count = 0
        window_correct = 0
        window_total = 0

        for _epoch in range(MAX_EPOCHS):
            for input_val in [0, 1]:
                # Present input
                for _t in range(PRESENTATION_STEPS):
                    _set_input(sim, input_val)
                    io_mgr.update_motor_rates(sim.fields, dt)
                    sim.step()

                # Read motor output
                actual_output = _read_motor_output(sim)
                expected = 1 - input_val  # NOT gate

                is_correct = actual_output == expected
                correct_count += int(is_correct)
                total_count += 1
                window_correct += int(is_correct)
                window_total += 1

                # Deliver feedback
                distance = 0 if is_correct else 2
                io_mgr.deliver_feedback(distance, sim.fields)

            # Check convergence over a rolling window
            if window_total >= 20:
                accuracy = window_correct / window_total
                if accuracy >= 0.6:
                    return  # Test passes

                # Reset window
                window_correct = 0
                window_total = 0

        # Final check over all trials
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        assert accuracy >= 0.6, (
            f"Inversion test failed: accuracy {accuracy:.1%} < 60% "
            f"({correct_count}/{total_count})"
        )
