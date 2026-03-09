"""Tests for STDP kernels: trace update, intra-block, and inter-block."""

import math

import numpy as np
import pytest
import taichi as ti

from hypertoric.kernels.stdp import (
    make_stdp_inter,
    make_stdp_intra,
    make_trace_update,
)
from hypertoric.topology import Topology

# Small torus: 2D, grid_size=2, K=4
NDIM = 2
GRID_SIZE = 2
K = 4
B = GRID_SIZE**NDIM  # 4
N_NEIGHBORS = 2 * NDIM  # 4

# STDP parameters
A_PLUS = 0.01
A_MINUS = 0.012
TAU_PRE = 20.0
TAU_POST = 20.0
W_MAX = 1.0
DT = 0.5


@pytest.fixture(scope="module")
def trace_kernel(ti_runtime: None) -> object:
    return make_trace_update(B, K)


@pytest.fixture(scope="module")
def intra_kernel(ti_runtime: None) -> object:
    return make_stdp_intra(B, K)


@pytest.fixture(scope="module")
def inter_kernel(ti_runtime: None) -> object:
    return make_stdp_inter(B, K, GRID_SIZE, NDIM)


@pytest.fixture()
def trace_fields(ti_runtime: None) -> tuple[ti.Field, ti.Field, ti.Field]:
    trace_pre = ti.field(dtype=ti.f32, shape=(B, K))
    trace_post = ti.field(dtype=ti.f32, shape=(B, K))
    spikes = ti.field(dtype=ti.i32, shape=(B, K))
    return trace_pre, trace_post, spikes


@pytest.fixture()
def intra_fields(
    ti_runtime: None,
) -> tuple[ti.Field, ti.Field, ti.Field, ti.Field, ti.Field]:
    w_intra = ti.field(dtype=ti.f32, shape=(B, K, K))
    spikes = ti.field(dtype=ti.i32, shape=(B, K))
    trace_pre = ti.field(dtype=ti.f32, shape=(B, K))
    trace_post = ti.field(dtype=ti.f32, shape=(B, K))
    return w_intra, spikes, trace_pre, trace_post


@pytest.fixture()
def inter_fields(
    ti_runtime: None,
) -> tuple[ti.Field, ti.Field, ti.Field, ti.Field]:
    w_inter = ti.field(dtype=ti.f32, shape=(B, N_NEIGHBORS, K, K))
    spikes = ti.field(dtype=ti.i32, shape=(B, K))
    trace_pre = ti.field(dtype=ti.f32, shape=(B, K))
    trace_post = ti.field(dtype=ti.f32, shape=(B, K))
    return w_inter, spikes, trace_pre, trace_post


class TestTraceUpdate:
    """Tests for trace decay and accumulation."""

    def test_trace_decay(
        self,
        trace_kernel: object,
        trace_fields: tuple[ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Traces decay exponentially with no spikes."""
        trace_pre, trace_post, spikes = trace_fields

        np_pre = np.zeros((B, K), dtype=np.float32)
        np_pre[0, 0] = 1.0
        trace_pre.from_numpy(np_pre)
        trace_post.from_numpy(np.zeros((B, K), dtype=np.float32))
        spikes.from_numpy(np.zeros((B, K), dtype=np.int32))

        n_steps = 20
        for _ in range(n_steps):
            trace_kernel(trace_pre, trace_post, spikes, TAU_PRE, TAU_POST, DT)

        result = trace_pre.to_numpy()
        expected = math.exp(-n_steps * DT / TAU_PRE)
        assert result[0, 0] == pytest.approx(expected, rel=0.01)

    def test_trace_accumulation(
        self,
        trace_kernel: object,
        trace_fields: tuple[ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Traces accumulate across multiple spikes."""
        trace_pre, trace_post, spikes = trace_fields

        trace_pre.from_numpy(np.zeros((B, K), dtype=np.float32))
        trace_post.from_numpy(np.zeros((B, K), dtype=np.float32))

        # First spike
        np_spikes = np.zeros((B, K), dtype=np.int32)
        np_spikes[0, 0] = 1
        spikes.from_numpy(np_spikes)
        trace_kernel(trace_pre, trace_post, spikes, TAU_PRE, TAU_POST, DT)

        # 10 steps with no spikes
        spikes.from_numpy(np.zeros((B, K), dtype=np.int32))
        for _ in range(10):
            trace_kernel(trace_pre, trace_post, spikes, TAU_PRE, TAU_POST, DT)

        # Second spike
        np_spikes[0, 0] = 1
        spikes.from_numpy(np_spikes)
        trace_kernel(trace_pre, trace_post, spikes, TAU_PRE, TAU_POST, DT)

        result = trace_pre.to_numpy()
        # After first spike: trace = 1.0
        # After 10 no-spike steps: trace = exp(-10*dt/tau)
        # At second spike: decay then increment: exp(-11*dt/tau) + 1.0
        expected = math.exp(-11 * DT / TAU_PRE) + 1.0
        assert result[0, 0] == pytest.approx(expected, rel=0.01)


class TestSTDPIntra:
    """Tests for intra-block STDP."""

    def test_ltp_pre_before_post(
        self,
        intra_kernel: object,
        intra_fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """LTP: pre fired recently, post fires now → weight increases."""
        w_intra, spikes, trace_pre, trace_post = intra_fields

        # Set up weight from neuron 0 → neuron 1
        np_w = np.zeros((B, K, K), dtype=np.float32)
        np_w[0, 0, 1] = 0.5
        w_intra.from_numpy(np_w)

        # Pre neuron 0 has residual trace (fired recently)
        np_trace_pre = np.zeros((B, K), dtype=np.float32)
        trace_val = math.exp(-5 * DT / TAU_PRE)
        np_trace_pre[0, 0] = trace_val
        trace_pre.from_numpy(np_trace_pre)
        trace_post.from_numpy(np.zeros((B, K), dtype=np.float32))

        # Post neuron 1 spikes now
        np_spikes = np.zeros((B, K), dtype=np.int32)
        np_spikes[0, 1] = 1
        spikes.from_numpy(np_spikes)

        intra_kernel(w_intra, spikes, trace_pre, trace_post, A_PLUS, A_MINUS, W_MAX)

        result = w_intra.to_numpy()
        expected = 0.5 + A_PLUS * trace_val
        assert result[0, 0, 1] == pytest.approx(expected, abs=1e-6)

    def test_ltd_post_before_pre(
        self,
        intra_kernel: object,
        intra_fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """LTD: post fired recently, pre fires now → weight decreases."""
        w_intra, spikes, trace_pre, trace_post = intra_fields

        np_w = np.zeros((B, K, K), dtype=np.float32)
        np_w[0, 0, 1] = 0.5
        w_intra.from_numpy(np_w)

        # Post neuron 1 has residual trace
        np_trace_post = np.zeros((B, K), dtype=np.float32)
        trace_val = math.exp(-5 * DT / TAU_POST)
        np_trace_post[0, 1] = trace_val
        trace_post.from_numpy(np_trace_post)
        trace_pre.from_numpy(np.zeros((B, K), dtype=np.float32))

        # Pre neuron 0 spikes now
        np_spikes = np.zeros((B, K), dtype=np.int32)
        np_spikes[0, 0] = 1
        spikes.from_numpy(np_spikes)

        intra_kernel(w_intra, spikes, trace_pre, trace_post, A_PLUS, A_MINUS, W_MAX)

        result = w_intra.to_numpy()
        expected = 0.5 - A_MINUS * trace_val
        assert result[0, 0, 1] == pytest.approx(expected, abs=1e-6)

    def test_asymmetry(
        self,
        intra_kernel: object,
        intra_fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """A_minus > A_plus means |LTD| > |LTP| for same timing."""
        w_intra, spikes, trace_pre, trace_post = intra_fields

        trace_val = math.exp(-5 * DT / TAU_PRE)

        # LTP scenario
        np_w = np.zeros((B, K, K), dtype=np.float32)
        np_w[0, 0, 1] = 0.5
        w_intra.from_numpy(np_w)
        np_tp = np.zeros((B, K), dtype=np.float32)
        np_tp[0, 0] = trace_val
        trace_pre.from_numpy(np_tp)
        trace_post.from_numpy(np.zeros((B, K), dtype=np.float32))
        np_sp = np.zeros((B, K), dtype=np.int32)
        np_sp[0, 1] = 1
        spikes.from_numpy(np_sp)

        intra_kernel(w_intra, spikes, trace_pre, trace_post, A_PLUS, A_MINUS, W_MAX)
        ltp_change = w_intra.to_numpy()[0, 0, 1] - 0.5

        # LTD scenario
        np_w[0, 0, 1] = 0.5
        w_intra.from_numpy(np_w)
        trace_pre.from_numpy(np.zeros((B, K), dtype=np.float32))
        np_tp2 = np.zeros((B, K), dtype=np.float32)
        np_tp2[0, 1] = trace_val
        trace_post.from_numpy(np_tp2)
        np_sp2 = np.zeros((B, K), dtype=np.int32)
        np_sp2[0, 0] = 1
        spikes.from_numpy(np_sp2)

        intra_kernel(w_intra, spikes, trace_pre, trace_post, A_PLUS, A_MINUS, W_MAX)
        ltd_change = 0.5 - w_intra.to_numpy()[0, 0, 1]

        assert ltd_change > ltp_change

    def test_weight_clamping_exc_upper(
        self,
        intra_kernel: object,
        intra_fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Excitatory weight clamps to w_max."""
        w_intra, spikes, trace_pre, trace_post = intra_fields

        np_w = np.zeros((B, K, K), dtype=np.float32)
        np_w[0, 0, 1] = W_MAX - 0.001
        w_intra.from_numpy(np_w)

        np_tp = np.zeros((B, K), dtype=np.float32)
        np_tp[0, 0] = 1.0  # Large trace to push past max
        trace_pre.from_numpy(np_tp)
        trace_post.from_numpy(np.zeros((B, K), dtype=np.float32))

        np_sp = np.zeros((B, K), dtype=np.int32)
        np_sp[0, 1] = 1
        spikes.from_numpy(np_sp)

        intra_kernel(w_intra, spikes, trace_pre, trace_post, A_PLUS, A_MINUS, W_MAX)

        result = w_intra.to_numpy()
        assert result[0, 0, 1] == pytest.approx(W_MAX, abs=1e-6)

    def test_weight_clamping_exc_lower(
        self,
        intra_kernel: object,
        intra_fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Excitatory weight clamps to 0.0 (doesn't go negative)."""
        w_intra, spikes, trace_pre, trace_post = intra_fields

        np_w = np.zeros((B, K, K), dtype=np.float32)
        np_w[0, 0, 1] = 0.001
        w_intra.from_numpy(np_w)

        np_tp = np.zeros((B, K), dtype=np.float32)
        np_tp[0, 1] = 1.0  # Large trace to push past zero
        trace_post.from_numpy(np_tp)
        trace_pre.from_numpy(np.zeros((B, K), dtype=np.float32))

        np_sp = np.zeros((B, K), dtype=np.int32)
        np_sp[0, 0] = 1  # Pre spikes → LTD
        spikes.from_numpy(np_sp)

        intra_kernel(w_intra, spikes, trace_pre, trace_post, A_PLUS, A_MINUS, W_MAX)

        result = w_intra.to_numpy()
        assert result[0, 0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_weight_clamping_inh(
        self,
        intra_kernel: object,
        intra_fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Inhibitory weight clamps to -w_max."""
        w_intra, spikes, trace_pre, trace_post = intra_fields

        np_w = np.zeros((B, K, K), dtype=np.float32)
        np_w[0, 0, 1] = -W_MAX + 0.001  # inhibitory, close to -w_max
        w_intra.from_numpy(np_w)

        # LTP on a negative weight → makes it less negative (toward 0)
        # We want LTD to push it more negative → pre spikes, post has trace
        np_tp = np.zeros((B, K), dtype=np.float32)
        np_tp[0, 1] = 1.0
        trace_post.from_numpy(np_tp)
        trace_pre.from_numpy(np.zeros((B, K), dtype=np.float32))

        np_sp = np.zeros((B, K), dtype=np.int32)
        np_sp[0, 0] = 1  # Pre spikes → LTD
        spikes.from_numpy(np_sp)

        intra_kernel(w_intra, spikes, trace_pre, trace_post, A_PLUS, A_MINUS, W_MAX)

        result = w_intra.to_numpy()
        # LTD subtracts from negative weight → more negative → clamp to -w_max
        assert result[0, 0, 1] >= -W_MAX - 1e-6

    def test_synapse_gate(
        self,
        intra_kernel: object,
        intra_fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Zero weight (no synapse) stays zero — STDP doesn't create synapses."""
        w_intra, spikes, trace_pre, trace_post = intra_fields

        w_intra.from_numpy(np.zeros((B, K, K), dtype=np.float32))

        np_tp = np.ones((B, K), dtype=np.float32)
        trace_pre.from_numpy(np_tp)
        trace_post.from_numpy(np_tp)

        # All neurons spike
        spikes.from_numpy(np.ones((B, K), dtype=np.int32))

        intra_kernel(w_intra, spikes, trace_pre, trace_post, A_PLUS, A_MINUS, W_MAX)

        result = w_intra.to_numpy()
        assert np.all(result == 0.0)


class TestSTDPInter:
    """Tests for inter-block STDP."""

    def test_rotating_mode_isolation(
        self,
        inter_kernel: object,
        inter_fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Only the specified direction's weights change."""
        w_inter, spikes, trace_pre, trace_post = inter_fields

        # Set non-zero weights in all directions
        np_w = np.ones((B, N_NEIGHBORS, K, K), dtype=np.float32) * 0.5
        w_inter.from_numpy(np_w)

        # Set up traces and spikes
        np_tp = np.ones((B, K), dtype=np.float32) * 0.5
        trace_pre.from_numpy(np_tp)
        trace_post.from_numpy(np_tp)
        spikes.from_numpy(np.ones((B, K), dtype=np.int32))

        # Run STDP only for direction 0
        inter_kernel(w_inter, spikes, trace_pre, trace_post, 0, A_PLUS, A_MINUS, W_MAX)

        result = w_inter.to_numpy()

        # Direction 0 should have changed
        assert not np.allclose(result[:, 0, :, :], 0.5)

        # Directions 1, 2, 3 should be unchanged
        for d in range(1, N_NEIGHBORS):
            np.testing.assert_allclose(result[:, d, :, :], 0.5)

    def test_inter_ltp(
        self,
        inter_kernel: object,
        inter_fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Inter-block LTP: post spikes in this block with pre trace in neighbor."""
        w_inter, spikes, trace_pre, trace_post = inter_fields
        topo = Topology(NDIM, GRID_SIZE)

        nb = topo.get_neighbor_flat(0, 0)

        np_w = np.zeros((B, N_NEIGHBORS, K, K), dtype=np.float32)
        np_w[0, 0, 0, 1] = 0.5  # Weight from nb:0 → block0:1
        w_inter.from_numpy(np_w)

        # Pre neuron 0 in neighbor has trace
        np_tp = np.zeros((B, K), dtype=np.float32)
        np_tp[nb, 0] = 0.8
        trace_pre.from_numpy(np_tp)
        trace_post.from_numpy(np.zeros((B, K), dtype=np.float32))

        # Post neuron 1 in block 0 spikes
        np_sp = np.zeros((B, K), dtype=np.int32)
        np_sp[0, 1] = 1
        spikes.from_numpy(np_sp)

        inter_kernel(w_inter, spikes, trace_pre, trace_post, 0, A_PLUS, A_MINUS, W_MAX)

        result = w_inter.to_numpy()
        expected = 0.5 + A_PLUS * 0.8
        assert result[0, 0, 0, 1] == pytest.approx(expected, abs=1e-6)

    def test_inter_ltd(
        self,
        inter_kernel: object,
        inter_fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field],
    ) -> None:
        """Inter-block LTD: pre spikes in neighbor with post trace in this block."""
        w_inter, spikes, trace_pre, trace_post = inter_fields
        topo = Topology(NDIM, GRID_SIZE)

        nb = topo.get_neighbor_flat(0, 0)

        np_w = np.zeros((B, N_NEIGHBORS, K, K), dtype=np.float32)
        np_w[0, 0, 0, 1] = 0.5
        w_inter.from_numpy(np_w)

        # Post neuron 1 in block 0 has trace
        np_tp = np.zeros((B, K), dtype=np.float32)
        np_tp[0, 1] = 0.8
        trace_post.from_numpy(np_tp)
        trace_pre.from_numpy(np.zeros((B, K), dtype=np.float32))

        # Pre neuron 0 in neighbor spikes
        np_sp = np.zeros((B, K), dtype=np.int32)
        np_sp[nb, 0] = 1
        spikes.from_numpy(np_sp)

        inter_kernel(w_inter, spikes, trace_pre, trace_post, 0, A_PLUS, A_MINUS, W_MAX)

        result = w_inter.to_numpy()
        expected = 0.5 - A_MINUS * 0.8
        assert result[0, 0, 0, 1] == pytest.approx(expected, abs=1e-6)
