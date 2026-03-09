"""Tests for the Simulator orchestration layer."""

import numpy as np
import pytest

from hypertoric.config import (
    NeuronConfig,
    PlasticityConfig,
    SimConfig,
    STDPConfig,
    TorusConfig,
)
from hypertoric.simulator import Simulator

NDIM = 2
GRID_SIZE = 2
K = 8
B = GRID_SIZE**NDIM


def _make_cfg() -> SimConfig:
    """Create a small test config with short plasticity intervals."""
    return SimConfig(
        torus=TorusConfig(ndim=NDIM, grid_size=GRID_SIZE, neurons_per_block=K),
        neuron=NeuronConfig(dt=0.5),
        stdp=STDPConfig(inter_mode="rotating"),
        plasticity=PlasticityConfig(
            interval=10,
            inter_interval=50,
            w_max=1.0,
            weight_threshold=0.001,
            init_weight=0.01,
        ),
    )


@pytest.fixture(scope="module")
def sim(ti_runtime: None) -> Simulator:
    return Simulator(_make_cfg())


class TestConstruction:
    """Tests for simulator construction."""

    def test_construction_succeeds(self, ti_runtime: None) -> None:
        """Simulator can be constructed with valid config."""
        s = Simulator(_make_cfg())
        assert s is not None

    def test_step_count_starts_at_zero(self, sim: Simulator) -> None:
        assert sim.step_count == 0

    def test_field_shapes(self, ti_runtime: None) -> None:
        """Fields have correct shapes."""
        s = Simulator(_make_cfg())
        assert s.fields.v.shape == (B, K)
        assert s.fields.spikes.shape == (B, K)
        assert s.fields.W_intra.shape == (B, K, K)
        assert s.fields.W_inter.shape == (B, 2 * NDIM, K, K)


class TestStepping:
    """Tests for simulation stepping."""

    def test_step_increments_count(self, ti_runtime: None) -> None:
        s = Simulator(_make_cfg())
        s.step()
        assert s.step_count == 1
        s.step()
        assert s.step_count == 2

    def test_100_steps_no_nan(self, ti_runtime: None) -> None:
        """100 steps with small constant current produces no NaN."""
        s = Simulator(_make_cfg())
        current = np.full((B, K), 5.0, dtype=np.float32)
        s.inject_current(current)

        for _ in range(100):
            s.step()

        v = s.read_voltage()
        assert not np.any(np.isnan(v))

    def test_spikes_with_sufficient_current(self, ti_runtime: None) -> None:
        """Strong current should produce spikes."""
        s = Simulator(_make_cfg())
        current = np.full((B, K), 20.0, dtype=np.float32)
        s.inject_current(current)

        any_spike = False
        for _ in range(100):
            s.step()
            spikes = s.read_spikes()
            if np.any(spikes == 1):
                any_spike = True
                break

        assert any_spike


class TestIO:
    """Tests for inject_current and read_spikes."""

    def test_inject_current_sets_field(self, ti_runtime: None) -> None:
        s = Simulator(_make_cfg())
        current = np.random.default_rng(0).standard_normal((B, K)).astype(np.float32)
        s.inject_current(current)

        result = s.fields.I_ext.to_numpy()
        np.testing.assert_allclose(result, current, atol=1e-6)

    def test_read_spikes_shape(self, ti_runtime: None) -> None:
        s = Simulator(_make_cfg())
        spikes = s.read_spikes()
        assert spikes.shape == (B, K)

    def test_inject_strong_current_produces_spikes(self, ti_runtime: None) -> None:
        """Injecting strong current should eventually produce spikes."""
        s = Simulator(_make_cfg())
        current = np.full((B, K), 30.0, dtype=np.float32)
        s.inject_current(current)

        for _ in range(50):
            s.step()

        spikes = s.read_spikes()
        assert np.any(spikes == 1)


class TestSTDPRotation:
    """Tests for STDP direction rotation."""

    def test_stdp_direction_cycles(self, ti_runtime: None) -> None:
        """_stdp_direction cycles through 0..num_neighbors-1."""
        s = Simulator(_make_cfg())
        num_neighbors = s.topology.num_neighbors

        directions_seen = set()
        for _ in range(num_neighbors):
            directions_seen.add(s._stdp_direction)
            s.step()

        assert directions_seen == set(range(num_neighbors))
