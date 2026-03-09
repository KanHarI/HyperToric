"""Tests for structural plasticity kernels: calcium update, intra-block, inter-block."""

import math

import numpy as np
import pytest
import taichi as ti

from hypertoric.kernels.plasticity import (
    make_calcium_update,
    make_structural_inter,
    make_structural_intra,
)

# Small torus: 2D, grid_size=2, K=8
NDIM = 2
GRID_SIZE = 2
K = 8
B = GRID_SIZE**NDIM  # 4
N_NEIGHBORS = 2 * NDIM  # 4

# Plasticity parameters
CALCIUM_TAU = 50.0
DT = 0.5
CALCIUM_LOW = 0.2
CALCIUM_HIGH = 0.8
WEIGHT_THRESHOLD = 0.001
INIT_WEIGHT = 0.01
W_MAX = 1.0


@pytest.fixture(scope="module")
def calcium_kernel(ti_runtime: None) -> object:
    return make_calcium_update(B, K)


@pytest.fixture(scope="module")
def intra_kernel(ti_runtime: None) -> object:
    return make_structural_intra(B, K)


@pytest.fixture(scope="module")
def inter_kernel(ti_runtime: None) -> object:
    return make_structural_inter(B, K)


@pytest.fixture()
def calcium_fields(ti_runtime: None) -> tuple[ti.Field, ti.Field]:
    calcium = ti.field(dtype=ti.f32, shape=(B, K))
    spikes = ti.field(dtype=ti.i32, shape=(B, K))
    return calcium, spikes


@pytest.fixture()
def intra_fields(ti_runtime: None) -> tuple[ti.Field, ti.Field]:
    w_intra = ti.field(dtype=ti.f32, shape=(B, K, K))
    calcium = ti.field(dtype=ti.f32, shape=(B, K))
    return w_intra, calcium


@pytest.fixture()
def inter_fields(ti_runtime: None) -> tuple[ti.Field, ti.Field]:
    w_inter = ti.field(dtype=ti.f32, shape=(B, N_NEIGHBORS, K, K))
    calcium = ti.field(dtype=ti.f32, shape=(B, K))
    return w_inter, calcium


class TestCalciumUpdate:
    """Tests for calcium decay and accumulation."""

    def test_calcium_decay(
        self,
        calcium_kernel: object,
        calcium_fields: tuple[ti.Field, ti.Field],
    ) -> None:
        """Calcium decays exponentially with no spikes."""
        calcium, spikes = calcium_fields

        np_ca = np.zeros((B, K), dtype=np.float32)
        np_ca[0, 0] = 1.0
        calcium.from_numpy(np_ca)
        spikes.from_numpy(np.zeros((B, K), dtype=np.int32))

        n_steps = 20
        for _ in range(n_steps):
            calcium_kernel(calcium, spikes, CALCIUM_TAU, DT)

        result = calcium.to_numpy()
        expected = math.exp(-n_steps * DT / CALCIUM_TAU)
        assert result[0, 0] == pytest.approx(expected, rel=0.01)

    def test_spike_increment(
        self,
        calcium_kernel: object,
        calcium_fields: tuple[ti.Field, ti.Field],
    ) -> None:
        """Calcium increments by 1.0 on spike."""
        calcium, spikes = calcium_fields

        calcium.from_numpy(np.zeros((B, K), dtype=np.float32))
        np_spikes = np.zeros((B, K), dtype=np.int32)
        np_spikes[0, 0] = 1
        spikes.from_numpy(np_spikes)

        calcium_kernel(calcium, spikes, CALCIUM_TAU, DT)

        result = calcium.to_numpy()
        # decay from 0 is 0, then +1.0
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_accumulation(
        self,
        calcium_kernel: object,
        calcium_fields: tuple[ti.Field, ti.Field],
    ) -> None:
        """Calcium accumulates across multiple spikes."""
        calcium, spikes = calcium_fields

        calcium.from_numpy(np.zeros((B, K), dtype=np.float32))

        # First spike
        np_spikes = np.zeros((B, K), dtype=np.int32)
        np_spikes[0, 0] = 1
        spikes.from_numpy(np_spikes)
        calcium_kernel(calcium, spikes, CALCIUM_TAU, DT)

        # 10 steps no spikes
        spikes.from_numpy(np.zeros((B, K), dtype=np.int32))
        for _ in range(10):
            calcium_kernel(calcium, spikes, CALCIUM_TAU, DT)

        # Second spike
        np_spikes[0, 0] = 1
        spikes.from_numpy(np_spikes)
        calcium_kernel(calcium, spikes, CALCIUM_TAU, DT)

        result = calcium.to_numpy()
        expected = math.exp(-11 * DT / CALCIUM_TAU) + 1.0
        assert result[0, 0] == pytest.approx(expected, rel=0.01)


class TestStructuralIntra:
    """Tests for intra-block structural plasticity."""

    def test_threshold_pruning(
        self,
        intra_kernel: object,
        intra_fields: tuple[ti.Field, ti.Field],
    ) -> None:
        """Synapses with |weight| < threshold are pruned."""
        w_intra, calcium = intra_fields

        np_w = np.zeros((B, K, K), dtype=np.float32)
        np_w[0, 0, 1] = WEIGHT_THRESHOLD / 2  # below threshold
        np_w[0, 2, 3] = -WEIGHT_THRESHOLD / 2  # negative, below threshold
        np_w[0, 1, 2] = 0.5  # above threshold, should survive
        w_intra.from_numpy(np_w)

        # Calcium in dead zone (no growth/pruning in pass 2)
        np_ca = np.full((B, K), (CALCIUM_LOW + CALCIUM_HIGH) / 2, dtype=np.float32)
        calcium.from_numpy(np_ca)

        intra_kernel(
            w_intra,
            calcium,
            CALCIUM_LOW,
            CALCIUM_HIGH,
            WEIGHT_THRESHOLD,
            INIT_WEIGHT,
            W_MAX,
        )

        result = w_intra.to_numpy()
        assert result[0, 0, 1] == 0.0
        assert result[0, 2, 3] == 0.0
        assert result[0, 1, 2] == pytest.approx(0.5, abs=1e-6)

    def test_growth_low_calcium(
        self,
        intra_kernel: object,
        intra_fields: tuple[ti.Field, ti.Field],
    ) -> None:
        """Low calcium triggers synapse growth (statistical test)."""
        w_intra, calcium = intra_fields

        grew = 0
        for _ in range(100):
            w_intra.from_numpy(np.zeros((B, K, K), dtype=np.float32))
            np_ca = np.full((B, K), CALCIUM_LOW / 2, dtype=np.float32)
            calcium.from_numpy(np_ca)

            intra_kernel(
                w_intra,
                calcium,
                CALCIUM_LOW,
                CALCIUM_HIGH,
                WEIGHT_THRESHOLD,
                INIT_WEIGHT,
                W_MAX,
            )

            result = w_intra.to_numpy()
            if np.any(result != 0.0):
                grew += 1

        # Should grow in a significant fraction of trials
        assert grew > 10

    def test_pruning_high_calcium(
        self,
        intra_kernel: object,
        intra_fields: tuple[ti.Field, ti.Field],
    ) -> None:
        """High calcium triggers synapse pruning."""
        w_intra, calcium = intra_fields

        pruned = 0
        for _ in range(100):
            # Fill all off-diagonal with weights
            np_w = np.full((B, K, K), 0.5, dtype=np.float32)
            for blk in range(B):
                np.fill_diagonal(np_w[blk], 0.0)
            w_intra.from_numpy(np_w)
            total_before = np.count_nonzero(np_w)

            np_ca = np.full((B, K), CALCIUM_HIGH * 2, dtype=np.float32)
            calcium.from_numpy(np_ca)

            intra_kernel(
                w_intra,
                calcium,
                CALCIUM_LOW,
                CALCIUM_HIGH,
                WEIGHT_THRESHOLD,
                INIT_WEIGHT,
                W_MAX,
            )

            result = w_intra.to_numpy()
            total_after = np.count_nonzero(result)
            if total_after < total_before:
                pruned += 1

        assert pruned > 10

    def test_no_self_connections(
        self,
        intra_kernel: object,
        intra_fields: tuple[ti.Field, ti.Field],
    ) -> None:
        """Growth should not create self-connections."""
        w_intra, calcium = intra_fields

        for _ in range(100):
            w_intra.from_numpy(np.zeros((B, K, K), dtype=np.float32))
            np_ca = np.full((B, K), CALCIUM_LOW / 2, dtype=np.float32)
            calcium.from_numpy(np_ca)

            intra_kernel(
                w_intra,
                calcium,
                CALCIUM_LOW,
                CALCIUM_HIGH,
                WEIGHT_THRESHOLD,
                INIT_WEIGHT,
                W_MAX,
            )

            result = w_intra.to_numpy()
            for blk in range(B):
                assert np.all(np.diag(result[blk]) == 0.0)

    def test_dead_zone_stability(
        self,
        intra_kernel: object,
        intra_fields: tuple[ti.Field, ti.Field],
    ) -> None:
        """Calcium in dead zone: no structural changes (except threshold pruning)."""
        w_intra, calcium = intra_fields

        np_w = np.zeros((B, K, K), dtype=np.float32)
        np_w[0, 0, 1] = 0.5
        np_w[0, 2, 3] = 0.3
        w_intra.from_numpy(np_w)

        np_ca = np.full((B, K), (CALCIUM_LOW + CALCIUM_HIGH) / 2, dtype=np.float32)
        calcium.from_numpy(np_ca)

        intra_kernel(
            w_intra,
            calcium,
            CALCIUM_LOW,
            CALCIUM_HIGH,
            WEIGHT_THRESHOLD,
            INIT_WEIGHT,
            W_MAX,
        )

        result = w_intra.to_numpy()
        np.testing.assert_allclose(result[0, 0, 1], 0.5, atol=1e-6)
        np.testing.assert_allclose(result[0, 2, 3], 0.3, atol=1e-6)


class TestStructuralInter:
    """Tests for inter-block structural plasticity."""

    def test_threshold_pruning(
        self,
        inter_kernel: object,
        inter_fields: tuple[ti.Field, ti.Field],
    ) -> None:
        """Synapses below threshold are pruned."""
        w_inter, calcium = inter_fields

        np_w = np.zeros((B, N_NEIGHBORS, K, K), dtype=np.float32)
        np_w[0, 0, 0, 1] = WEIGHT_THRESHOLD / 2
        np_w[0, 0, 2, 3] = 0.5  # above threshold
        w_inter.from_numpy(np_w)

        np_ca = np.full((B, K), (CALCIUM_LOW + CALCIUM_HIGH) / 2, dtype=np.float32)
        calcium.from_numpy(np_ca)

        inter_kernel(
            w_inter,
            calcium,
            CALCIUM_LOW,
            CALCIUM_HIGH,
            WEIGHT_THRESHOLD,
            INIT_WEIGHT,
            W_MAX,
            0,
        )

        result = w_inter.to_numpy()
        assert result[0, 0, 0, 1] == 0.0
        assert result[0, 0, 2, 3] == pytest.approx(0.5, abs=1e-6)

    def test_growth_positive_weight(
        self,
        inter_kernel: object,
        inter_fields: tuple[ti.Field, ti.Field],
    ) -> None:
        """Growth produces positive (excitatory) weights."""
        w_inter, calcium = inter_fields

        grew = 0
        for _ in range(100):
            w_inter.from_numpy(np.zeros((B, N_NEIGHBORS, K, K), dtype=np.float32))
            np_ca = np.full((B, K), CALCIUM_LOW / 2, dtype=np.float32)
            calcium.from_numpy(np_ca)

            inter_kernel(
                w_inter,
                calcium,
                CALCIUM_LOW,
                CALCIUM_HIGH,
                WEIGHT_THRESHOLD,
                INIT_WEIGHT,
                W_MAX,
                0,
            )

            result = w_inter.to_numpy()
            nonzero = result[result != 0.0]
            if len(nonzero) > 0:
                grew += 1
                assert np.all(nonzero > 0.0)

        # Ensure growth actually happened in a significant fraction of trials
        assert grew > 10

    def test_direction_isolation(
        self,
        inter_kernel: object,
        inter_fields: tuple[ti.Field, ti.Field],
    ) -> None:
        """Only the specified direction is modified."""
        w_inter, calcium = inter_fields

        # Set weights in all directions
        np_w = np.full((B, N_NEIGHBORS, K, K), 0.5, dtype=np.float32)
        w_inter.from_numpy(np_w)

        # High calcium to trigger pruning in direction 0 only
        np_ca = np.full((B, K), CALCIUM_HIGH * 2, dtype=np.float32)
        calcium.from_numpy(np_ca)

        inter_kernel(
            w_inter,
            calcium,
            CALCIUM_LOW,
            CALCIUM_HIGH,
            WEIGHT_THRESHOLD,
            INIT_WEIGHT,
            W_MAX,
            0,
        )

        result = w_inter.to_numpy()
        # Directions 1, 2, 3 should be unchanged
        for d in range(1, N_NEIGHBORS):
            np.testing.assert_allclose(result[:, d, :, :], 0.5)
