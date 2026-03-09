"""Tests for the IOManager: plane selection, encoding, decoding, feedback."""

import math

import numpy as np
import pytest

from hypertoric.config import IOConfig, NeuronConfig, SimConfig, TorusConfig
from hypertoric.fields import SimFields, build_fields, init_fields
from hypertoric.io import IOManager
from hypertoric.topology import Topology

NDIM = 2
GRID_SIZE = 4
K = 16
B = GRID_SIZE**NDIM
SEED = 42


def _make_cfg() -> SimConfig:
    return SimConfig(
        torus=TorusConfig(ndim=NDIM, grid_size=GRID_SIZE, neurons_per_block=K),
        neuron=NeuronConfig(dt=0.5),
        io=IOConfig(
            sensory_axis=0,
            sensory_position=0,
            motor_position=3,
            sensory_cluster_size=4,
            motor_cluster_size=4,
        ),
    )


def _make_io(cfg: SimConfig, topo: Topology, seed: int = SEED) -> IOManager:
    return IOManager(topo, cfg.io, cfg.torus.neurons_per_block, seed)


@pytest.fixture(scope="module")
def cfg() -> SimConfig:
    return _make_cfg()


@pytest.fixture(scope="module")
def topo(cfg: SimConfig) -> Topology:
    return Topology(cfg.torus.ndim, cfg.torus.grid_size)


@pytest.fixture(scope="module")
def fields(cfg: SimConfig, topo: Topology, ti_runtime: None) -> SimFields:
    f = build_fields(cfg, topo)
    init_fields(f, cfg, topo)
    return f


@pytest.fixture()
def io_mgr(cfg: SimConfig, topo: Topology, fields: SimFields) -> IOManager:
    return _make_io(cfg, topo)


class TestPlaneSelection:
    """Verify sensory and motor plane block selection."""

    def test_sensory_plane_size(self, io_mgr: IOManager) -> None:
        expected = GRID_SIZE ** (NDIM - 1)
        assert len(io_mgr.sensory_blocks) == expected

    def test_motor_plane_size(self, io_mgr: IOManager) -> None:
        expected = GRID_SIZE ** (NDIM - 1)
        assert len(io_mgr.motor_blocks) == expected

    def test_sensory_coords(self, io_mgr: IOManager, topo: Topology) -> None:
        for blk in io_mgr.sensory_blocks:
            assert topo.flat_to_coord(blk)[0] == 0

    def test_motor_coords(self, io_mgr: IOManager, topo: Topology) -> None:
        for blk in io_mgr.motor_blocks:
            assert topo.flat_to_coord(blk)[0] == 3

    def test_disjoint(self, io_mgr: IOManager) -> None:
        assert set(io_mgr.sensory_blocks).isdisjoint(set(io_mgr.motor_blocks))

    def test_different_axes(self, ti_runtime: None) -> None:
        """Motor blocks use motor_axis when it differs from sensory_axis."""
        ndim = 3
        grid = 4
        c = SimConfig(
            torus=TorusConfig(ndim=ndim, grid_size=grid, neurons_per_block=K),
            io=IOConfig(
                sensory_axis=0,
                sensory_position=0,
                motor_axis=1,
                motor_position=2,
            ),
        )
        t = Topology(ndim, grid)
        mgr = IOManager(t, c.io, K, seed=SEED)

        for blk in mgr.sensory_blocks:
            assert t.flat_to_coord(blk)[0] == 0
        for blk in mgr.motor_blocks:
            assert t.flat_to_coord(blk)[1] == 2


class TestSensoryEncoding:
    """Verify Gaussian place-code encoding on sensory blocks."""

    def test_center_target_peak(self, io_mgr: IOManager, fields: SimFields) -> None:
        """Block closest to target=0.5 gets highest current."""
        io_mgr.encode_sensory(0.5, fields)
        i_ext = fields.I_ext.to_numpy()
        currents = [i_ext[blk, 0] for blk in io_mgr.sensory_blocks]
        # Find block with preferred closest to 0.5
        n = len(io_mgr.sensory_blocks)
        preferred = [i / n for i in range(n)]
        dists = [min(abs(0.5 - p), 1.0 - abs(0.5 - p)) for p in preferred]
        best_idx = int(np.argmin(dists))
        assert currents[best_idx] == max(currents)

    def test_different_targets_different_patterns(
        self, io_mgr: IOManager, fields: SimFields
    ) -> None:
        io_mgr.encode_sensory(0.1, fields)
        pat_a = fields.I_ext.to_numpy().copy()
        io_mgr.encode_sensory(0.9, fields)
        pat_b = fields.I_ext.to_numpy()
        assert not np.allclose(pat_a, pat_b)

    def test_total_current_positive(self, io_mgr: IOManager, fields: SimFields) -> None:
        io_mgr.encode_sensory(0.5, fields)
        i_ext = fields.I_ext.to_numpy()
        total = sum(i_ext[blk, 0] for blk in io_mgr.sensory_blocks)
        assert total > 0

    def test_only_cluster_neurons_receive_current(
        self, io_mgr: IOManager, fields: SimFields, cfg: SimConfig
    ) -> None:
        # Zero out first
        i_ext = np.zeros_like(fields.I_ext.to_numpy())
        fields.I_ext.from_numpy(i_ext)

        io_mgr.encode_sensory(0.5, fields)
        i_ext = fields.I_ext.to_numpy()
        cluster = cfg.io.sensory_cluster_size
        for blk in io_mgr.sensory_blocks:
            # Cluster neurons should have current
            assert np.any(i_ext[blk, :cluster] > 0)
            # Non-cluster neurons should be zero
            if cluster < K:
                np.testing.assert_array_equal(i_ext[blk, cluster:], 0.0)


class TestMotorDecoding:
    """Verify motor rate decay and decoding."""

    def test_rate_decay_no_spikes(self, io_mgr: IOManager, fields: SimFields) -> None:
        """Rates decay toward zero with no spikes."""
        io_mgr.rate_up = 10.0
        io_mgr.rate_down = 10.0
        # Zero spikes
        spikes = np.zeros_like(fields.spikes.to_numpy())
        fields.spikes.from_numpy(spikes)

        dt = 0.5
        io_mgr.update_motor_rates(fields, dt)
        expected = 10.0 * math.exp(-dt / io_mgr._cfg.tau_motor)
        assert abs(io_mgr.rate_up - expected) < 1e-6
        assert abs(io_mgr.rate_down - expected) < 1e-6

    def test_up_spikes_decode_positive(
        self, cfg: SimConfig, topo: Topology, fields: SimFields
    ) -> None:
        """Injecting spikes into up population should decode +1."""
        io_mgr = IOManager(topo, cfg.io, K, seed=123)
        # Reset stats for clean test
        io_mgr.diff_mean = 0.0
        io_mgr.diff_var = 1.0

        cluster = min(cfg.io.motor_cluster_size, K)
        up_count = cluster // 2 + (cluster % 2)
        dt = 0.5

        for _ in range(100):
            spikes = np.zeros((B, K), dtype=np.int32)
            for blk in io_mgr.motor_blocks:
                spikes[blk, :up_count] = 1  # all up neurons fire
            fields.spikes.from_numpy(spikes)
            io_mgr.update_motor_rates(fields, dt)

        result = io_mgr.decode_motor()
        assert result == 1

    def test_equal_spikes_decode_zero(
        self, cfg: SimConfig, topo: Topology, fields: SimFields
    ) -> None:
        """Equal spikes in both populations should decode 0."""
        io_mgr = IOManager(topo, cfg.io, K, seed=456)
        dt = 0.5

        for _ in range(100):
            spikes = np.zeros((B, K), dtype=np.int32)
            for blk in io_mgr.motor_blocks:
                spikes[blk, :] = 1  # all neurons fire
            fields.spikes.from_numpy(spikes)
            io_mgr.update_motor_rates(fields, dt)

        result = io_mgr.decode_motor()
        assert result == 0


class TestFeedback:
    """Verify order/chaos feedback delivery."""

    def test_distance_zero_always_ordered(
        self, cfg: SimConfig, topo: Topology, fields: SimFields
    ) -> None:
        """distance=0 -> always ordered pulse."""
        io_mgr = IOManager(topo, cfg.io, K, seed=0)
        cluster = cfg.io.sensory_cluster_size
        for _ in range(50):
            io_mgr.deliver_feedback(0, fields)
            i_ext = fields.I_ext.to_numpy()
            for blk in io_mgr.sensory_blocks:
                vals = i_ext[blk, :cluster]
                np.testing.assert_array_equal(vals, cfg.io.base_current)

    def test_distance_large_always_chaotic(
        self, cfg: SimConfig, topo: Topology, fields: SimFields
    ) -> None:
        """distance>=3 -> always chaotic (p=0.0)."""
        io_mgr = IOManager(topo, cfg.io, K, seed=0)
        cluster = cfg.io.sensory_cluster_size
        for _ in range(50):
            io_mgr.deliver_feedback(100, fields)
            i_ext = fields.I_ext.to_numpy()
            # Chaotic: varied amplitudes
            all_vals = np.concatenate(
                [i_ext[blk, :cluster] for blk in io_mgr.sensory_blocks]
            )
            assert np.std(all_vals) > 0

    def test_distance_one_mixed(
        self, cfg: SimConfig, topo: Topology, fields: SimFields
    ) -> None:
        """distance=1 -> ~70% ordered (check 50-90% range over 200 trials)."""
        io_mgr = IOManager(topo, cfg.io, K, seed=77)
        cluster = cfg.io.sensory_cluster_size
        ordered_count = 0
        n_trials = 200
        for _ in range(n_trials):
            io_mgr.deliver_feedback(1, fields)
            i_ext = fields.I_ext.to_numpy()
            vals = i_ext[io_mgr.sensory_blocks[0], :cluster]
            if np.all(vals == cfg.io.base_current):
                ordered_count += 1

        ratio = ordered_count / n_trials
        assert 0.50 < ratio < 0.90, f"ordered ratio {ratio:.2f} outside (0.50, 0.90)"

    def test_ordered_uniform_amplitude(
        self, cfg: SimConfig, topo: Topology, fields: SimFields
    ) -> None:
        """Ordered pulses have uniform amplitude equal to base_current."""
        io_mgr = IOManager(topo, cfg.io, K, seed=0)
        io_mgr.deliver_feedback(0, fields)
        i_ext = fields.I_ext.to_numpy()
        cluster = cfg.io.sensory_cluster_size
        for blk in io_mgr.sensory_blocks:
            np.testing.assert_array_equal(i_ext[blk, :cluster], cfg.io.base_current)

    def test_chaotic_varied_amplitude(
        self, cfg: SimConfig, topo: Topology, fields: SimFields
    ) -> None:
        """Chaotic pulses have varied amplitudes (std > 0)."""
        io_mgr = IOManager(topo, cfg.io, K, seed=0)
        io_mgr.deliver_feedback(100, fields)
        i_ext = fields.I_ext.to_numpy()
        cluster = cfg.io.sensory_cluster_size
        all_vals = np.concatenate(
            [i_ext[blk, :cluster] for blk in io_mgr.sensory_blocks]
        )
        assert np.std(all_vals) > 0
