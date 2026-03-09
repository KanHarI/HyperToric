"""Tests for spike propagation kernel."""

import math

import numpy as np
import pytest
import taichi as ti

from hypertoric.kernels.propagate import make_spike_propagate
from hypertoric.topology import Topology

# Small torus: 2D, grid_size=2, K=4 → 4 blocks, 4 neighbors each
NDIM = 2
GRID_SIZE = 2
K = 4
B = GRID_SIZE**NDIM  # 4
TAU_SYN = 5.0
DT = 0.5
DECAY = math.exp(-DT / TAU_SYN)


@pytest.fixture(scope="module")
def propagate_kernel(ti_runtime: None) -> object:
    """Build propagation kernel once per module."""
    return make_spike_propagate(B, K, GRID_SIZE, NDIM)


@pytest.fixture()
def fields(ti_runtime: None) -> tuple[ti.Field, ti.Field, ti.Field, ti.Field]:
    """Allocate fresh fields for each test."""
    i_syn = ti.field(dtype=ti.f32, shape=(B, K))
    spikes = ti.field(dtype=ti.i32, shape=(B, K))
    w_intra = ti.field(dtype=ti.f32, shape=(B, K, K))
    n_neighbors = 2 * NDIM
    w_inter = ti.field(dtype=ti.f32, shape=(B, n_neighbors, K, K))
    return i_syn, spikes, w_intra, w_inter


class TestIntraBlock:
    """Intra-block spike propagation tests."""

    def test_excitatory(
        self,
        propagate_kernel: object,
        fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Excitatory weight: spike increases I_syn."""
        i_syn, spikes, w_intra, w_inter = fields

        # Set W_intra[0, 0, 1] = 0.5 (neuron 0 → neuron 1 in block 0)
        w_inter.from_numpy(np.zeros((B, 2 * NDIM, K, K), dtype=np.float32))
        np_w = np.zeros((B, K, K), dtype=np.float32)
        np_w[0, 0, 1] = 0.5
        w_intra.from_numpy(np_w)

        # Spike neuron 0 in block 0
        np_spikes = np.zeros((B, K), dtype=np.int32)
        np_spikes[0, 0] = 1
        spikes.from_numpy(np_spikes)

        i_syn.from_numpy(np.zeros((B, K), dtype=np.float32))

        propagate_kernel(i_syn, spikes, w_intra, w_inter, DECAY)

        result = i_syn.to_numpy()
        assert result[0, 1] == pytest.approx(0.5, abs=1e-6)

    def test_inhibitory(
        self,
        propagate_kernel: object,
        fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Inhibitory weight: spike decreases I_syn."""
        i_syn, spikes, w_intra, w_inter = fields

        np_w = np.zeros((B, K, K), dtype=np.float32)
        np_w[0, 0, 1] = -0.3
        w_intra.from_numpy(np_w)
        w_inter.from_numpy(np.zeros((B, 2 * NDIM, K, K), dtype=np.float32))

        np_spikes = np.zeros((B, K), dtype=np.int32)
        np_spikes[0, 0] = 1
        spikes.from_numpy(np_spikes)

        i_syn.from_numpy(np.zeros((B, K), dtype=np.float32))

        propagate_kernel(i_syn, spikes, w_intra, w_inter, DECAY)

        result = i_syn.to_numpy()
        assert result[0, 1] == pytest.approx(-0.3, abs=1e-6)


class TestInterBlock:
    """Inter-block spike propagation tests."""

    def test_inter_block(
        self,
        propagate_kernel: object,
        fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Spike in neighbor block propagates through inter-block weights."""
        i_syn, spikes, w_intra, w_inter = fields
        topo = Topology(NDIM, GRID_SIZE)

        # Neighbor of block 0 in direction 0
        nb = topo.get_neighbor_flat(0, 0)

        w_intra.from_numpy(np.zeros((B, K, K), dtype=np.float32))
        np_w_inter = np.zeros((B, 2 * NDIM, K, K), dtype=np.float32)
        # Weight from neuron 0 in neighbor → neuron 1 in block 0
        np_w_inter[0, 0, 0, 1] = 0.7
        w_inter.from_numpy(np_w_inter)

        # Spike neuron 0 in the neighbor block
        np_spikes = np.zeros((B, K), dtype=np.int32)
        np_spikes[nb, 0] = 1
        spikes.from_numpy(np_spikes)

        i_syn.from_numpy(np.zeros((B, K), dtype=np.float32))

        propagate_kernel(i_syn, spikes, w_intra, w_inter, DECAY)

        result = i_syn.to_numpy()
        assert result[0, 1] == pytest.approx(0.7, abs=1e-6)

    def test_periodic_wrapping(
        self,
        propagate_kernel: object,
        fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Periodic boundary: block at edge wraps to opposite side."""
        i_syn, spikes, w_intra, w_inter = fields
        topo = Topology(NDIM, GRID_SIZE)

        # In 2x2 2D torus, block (1,0)=flat 1, +x neighbor wraps to (0,0)=flat 0
        src_block = topo.coord_to_flat((1, 0))
        dst_block = topo.coord_to_flat((0, 0))
        # Find which direction from dst_block leads to src_block
        direction = -1
        for d in range(topo.num_neighbors):
            if topo.get_neighbor_flat(dst_block, d) == src_block:
                direction = d
                break
        assert direction >= 0, "Could not find wrap direction"

        w_intra.from_numpy(np.zeros((B, K, K), dtype=np.float32))
        np_w_inter = np.zeros((B, 2 * NDIM, K, K), dtype=np.float32)
        np_w_inter[dst_block, direction, 2, 3] = 0.9
        w_inter.from_numpy(np_w_inter)

        np_spikes = np.zeros((B, K), dtype=np.int32)
        np_spikes[src_block, 2] = 1
        spikes.from_numpy(np_spikes)

        i_syn.from_numpy(np.zeros((B, K), dtype=np.float32))

        propagate_kernel(i_syn, spikes, w_intra, w_inter, DECAY)

        result = i_syn.to_numpy()
        assert result[dst_block, 3] == pytest.approx(0.9, abs=1e-6)


class TestDecayAndSuperposition:
    """Exponential decay and superposition tests."""

    def test_exponential_decay(
        self,
        propagate_kernel: object,
        fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """I_syn decays exponentially with no spikes."""
        i_syn, spikes, w_intra, w_inter = fields

        w_intra.from_numpy(np.zeros((B, K, K), dtype=np.float32))
        w_inter.from_numpy(np.zeros((B, 2 * NDIM, K, K), dtype=np.float32))
        spikes.from_numpy(np.zeros((B, K), dtype=np.int32))

        # Set initial I_syn
        np_isyn = np.zeros((B, K), dtype=np.float32)
        np_isyn[0, 0] = 10.0
        i_syn.from_numpy(np_isyn)

        # Run 20 steps with no spikes
        for _ in range(20):
            propagate_kernel(i_syn, spikes, w_intra, w_inter, DECAY)

        result = i_syn.to_numpy()
        expected = 10.0 * DECAY**20
        assert result[0, 0] == pytest.approx(expected, rel=1e-4)

    def test_superposition(
        self,
        propagate_kernel: object,
        fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Two simultaneous spikes: currents add linearly."""
        i_syn, spikes, w_intra, w_inter = fields

        np_w = np.zeros((B, K, K), dtype=np.float32)
        np_w[0, 0, 2] = 0.4  # neuron 0 → neuron 2
        np_w[0, 1, 2] = 0.6  # neuron 1 → neuron 2
        w_intra.from_numpy(np_w)
        w_inter.from_numpy(np.zeros((B, 2 * NDIM, K, K), dtype=np.float32))

        np_spikes = np.zeros((B, K), dtype=np.int32)
        np_spikes[0, 0] = 1
        np_spikes[0, 1] = 1
        spikes.from_numpy(np_spikes)

        i_syn.from_numpy(np.zeros((B, K), dtype=np.float32))

        propagate_kernel(i_syn, spikes, w_intra, w_inter, DECAY)

        result = i_syn.to_numpy()
        assert result[0, 2] == pytest.approx(1.0, abs=1e-6)

    def test_zero_weights(
        self,
        propagate_kernel: object,
        fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Zero weights: spikes produce no current, only decay."""
        i_syn, spikes, w_intra, w_inter = fields

        w_intra.from_numpy(np.zeros((B, K, K), dtype=np.float32))
        w_inter.from_numpy(np.zeros((B, 2 * NDIM, K, K), dtype=np.float32))

        # All neurons spike
        spikes.from_numpy(np.ones((B, K), dtype=np.int32))

        np_isyn = np.zeros((B, K), dtype=np.float32)
        np_isyn[0, 0] = 5.0
        i_syn.from_numpy(np_isyn)

        propagate_kernel(i_syn, spikes, w_intra, w_inter, DECAY)

        result = i_syn.to_numpy()
        # Only decay, no new current
        assert result[0, 0] == pytest.approx(5.0 * DECAY, abs=1e-5)
        # Other neurons should remain at 0
        assert result[0, 1] == pytest.approx(0.0, abs=1e-6)
