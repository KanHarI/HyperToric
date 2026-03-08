"""Tests for the Izhikevich neuron kernel."""

from __future__ import annotations

import numpy as np
import pytest
import taichi as ti

from hypertoric.kernels.neuron_models import NEURON_MODELS, get_neuron_factory
from hypertoric.kernels.neuron_models.izhikevich import make_izhikevich_update

# Izhikevich parameters: (a, b, c, d)
PARAMS = {
    "RS": (0.02, 0.2, -65.0, 8.0),
    "IB": (0.02, 0.2, -55.0, 4.0),
    "CH": (0.02, 0.2, -50.0, 2.0),
    "FS": (0.1, 0.2, -65.0, 2.0),
}


def _alloc_fields(b: int, k: int) -> tuple:
    """Allocate Taichi fields for neuron state."""
    v = ti.field(dtype=ti.f32, shape=(b, k))
    u = ti.field(dtype=ti.f32, shape=(b, k))
    spikes = ti.field(dtype=ti.i32, shape=(b, k))
    i_ext = ti.field(dtype=ti.f32, shape=(b, k))
    param_a = ti.field(dtype=ti.f32, shape=(b, k))
    param_b = ti.field(dtype=ti.f32, shape=(b, k))
    param_c = ti.field(dtype=ti.f32, shape=(b, k))
    param_d = ti.field(dtype=ti.f32, shape=(b, k))
    return v, u, spikes, i_ext, param_a, param_b, param_c, param_d


def _init_fields(
    v: ti.Field,
    u: ti.Field,
    spikes: ti.Field,
    i_ext: ti.Field,
    param_a: ti.Field,
    param_b: ti.Field,
    param_c: ti.Field,
    param_d: ti.Field,
    ntype: str,
    current: float,
    b: int,
    k: int,
) -> None:
    """Initialize fields with given neuron type and current."""
    a, bv, c, d = PARAMS[ntype]
    np_v = np.full((b, k), -65.0, dtype=np.float32)
    np_u = np.full((b, k), bv * -65.0, dtype=np.float32)
    np_spikes = np.zeros((b, k), dtype=np.int32)
    np_i_ext = np.full((b, k), current, dtype=np.float32)
    np_a = np.full((b, k), a, dtype=np.float32)
    np_b = np.full((b, k), bv, dtype=np.float32)
    np_c = np.full((b, k), c, dtype=np.float32)
    np_d = np.full((b, k), d, dtype=np.float32)

    v.from_numpy(np_v)
    u.from_numpy(np_u)
    spikes.from_numpy(np_spikes)
    i_ext.from_numpy(np_i_ext)
    param_a.from_numpy(np_a)
    param_b.from_numpy(np_b)
    param_c.from_numpy(np_c)
    param_d.from_numpy(np_d)


class TestRegistry:
    """Test neuron model registry."""

    def test_izhikevich_registered(self) -> None:
        assert "izhikevich" in NEURON_MODELS

    def test_get_neuron_factory(self) -> None:
        factory = get_neuron_factory("izhikevich")
        assert factory is make_izhikevich_update

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown neuron model"):
            get_neuron_factory("nonexistent")


@pytest.mark.parametrize("ntype", ["RS", "FS", "CH", "IB"])
class TestSpiking:
    """Test spike behavior across neuron types."""

    def test_suprathreshold_spikes(self, ti_cpu: None, ntype: str) -> None:
        """Suprathreshold current (I=15) should produce spikes within 100 steps."""
        b, k = 1, 4
        fields = _alloc_fields(b, k)
        _init_fields(*fields, ntype=ntype, current=15.0, b=b, k=k)
        v, u, spikes, i_ext, pa, pb, pc, pd = fields

        kernel = make_izhikevich_update(b, k)
        total_spikes = 0
        for _ in range(100):
            kernel(v, u, spikes, i_ext, pa, pb, pc, pd, 0.5)
            total_spikes += int(np.sum(spikes.to_numpy()))

        assert total_spikes > 0, f"{ntype}: no spikes with I=15 in 100 steps"

    def test_subthreshold_no_spikes(self, ti_cpu: None, ntype: str) -> None:
        """Subthreshold current (I=3) should produce no spikes over 10k steps."""
        b, k = 1, 4
        fields = _alloc_fields(b, k)
        _init_fields(*fields, ntype=ntype, current=3.0, b=b, k=k)
        v, u, spikes, i_ext, pa, pb, pc, pd = fields

        kernel = make_izhikevich_update(b, k)
        total_spikes = 0
        for _ in range(10_000):
            kernel(v, u, spikes, i_ext, pa, pb, pc, pd, 0.5)
            total_spikes += int(np.sum(spikes.to_numpy()))

        assert total_spikes == 0, f"{ntype}: unexpected spikes with I=3"


class TestResetMechanics:
    """Test spike reset behavior."""

    def test_reset_v_to_c(self, ti_cpu: None) -> None:
        """After spike, v should be reset to c."""
        b, k = 1, 4
        fields = _alloc_fields(b, k)
        _init_fields(*fields, ntype="RS", current=15.0, b=b, k=k)
        v, u, spikes, i_ext, pa, pb, pc, pd = fields

        kernel = make_izhikevich_update(b, k)
        for _ in range(200):
            kernel(v, u, spikes, i_ext, pa, pb, pc, pd, 0.5)
            s = spikes.to_numpy()
            if np.any(s > 0):
                v_np = v.to_numpy()
                # Where spike occurred, v should be c=-65
                spike_mask = s > 0
                assert np.allclose(v_np[spike_mask], -65.0), (
                    "v not reset to c after spike"
                )
                return

        pytest.fail("No spike observed in 200 steps")

    def test_v_never_exceeds_30(self, ti_cpu: None) -> None:
        """v should always be <= 30 (resets enforced)."""
        b, k = 1, 8
        fields = _alloc_fields(b, k)
        _init_fields(*fields, ntype="RS", current=15.0, b=b, k=k)
        v, u, spikes, i_ext, pa, pb, pc, pd = fields

        kernel = make_izhikevich_update(b, k)
        for step in range(10_000):
            kernel(v, u, spikes, i_ext, pa, pb, pc, pd, 0.5)
            v_np = v.to_numpy()
            assert np.all(v_np <= 30.0), f"v exceeded 30 at step {step}"


class TestNumericalStability:
    """Test numerical stability over many steps."""

    def test_no_nan_divergence(self, ti_cpu: None) -> None:
        """No NaN or divergence over 10k steps with various inputs."""
        b, k = 1, 8
        fields = _alloc_fields(b, k)
        _init_fields(*fields, ntype="RS", current=10.0, b=b, k=k)
        v, u, spikes, i_ext, pa, pb, pc, pd = fields

        kernel = make_izhikevich_update(b, k)
        for step in range(10_000):
            kernel(v, u, spikes, i_ext, pa, pb, pc, pd, 0.5)
            v_np = v.to_numpy()
            u_np = u.to_numpy()
            assert not np.any(np.isnan(v_np)), f"NaN in v at step {step}"
            assert not np.any(np.isnan(u_np)), f"NaN in u at step {step}"
            assert not np.any(np.isinf(v_np)), f"Inf in v at step {step}"
            assert not np.any(np.isinf(u_np)), f"Inf in u at step {step}"


class TestIsolation:
    """Test that blocks are independent (no cross-block coupling in neuron update)."""

    def test_block_isolation(self, ti_cpu: None) -> None:
        """Inject current in block 0 only — block 1 should have zero spikes."""
        b, k = 2, 8
        fields = _alloc_fields(b, k)
        _init_fields(*fields, ntype="RS", current=0.0, b=b, k=k)
        v, u, spikes, i_ext, pa, pb, pc, pd = fields

        # Set current only for block 0
        np_i = np.zeros((b, k), dtype=np.float32)
        np_i[0, :] = 15.0
        i_ext.from_numpy(np_i)

        kernel = make_izhikevich_update(b, k)
        block1_spikes = 0
        for _ in range(500):
            kernel(v, u, spikes, i_ext, pa, pb, pc, pd, 0.5)
            block1_spikes += int(np.sum(spikes.to_numpy()[1, :]))

        assert block1_spikes == 0, "Block 1 spiked despite receiving no current"
