"""Tests for SimFields allocation and initialization."""

from __future__ import annotations

import numpy as np
import pytest

from hypertoric.config import SimConfig, TorusConfig
from hypertoric.fields import build_fields, init_fields
from hypertoric.topology import Topology


def _make_cfg(ndim: int, grid_size: int, k: int, seed: int = 42) -> SimConfig:
    return SimConfig(
        torus=TorusConfig(ndim=ndim, grid_size=grid_size, neurons_per_block=k),
        seed=seed,
    )


def _make_topo(cfg: SimConfig) -> Topology:
    return Topology(cfg.torus.ndim, cfg.torus.grid_size)


@pytest.mark.parametrize(
    ("ndim", "grid_size", "k"),
    [
        (2, 2, 16),
        (3, 3, 32),
    ],
    ids=["small_2d", "medium_3d"],
)
class TestFieldShapes:
    """Verify field shapes for various configs."""

    def test_state_shapes(
        self, ti_cpu: None, ndim: int, grid_size: int, k: int
    ) -> None:
        cfg = _make_cfg(ndim, grid_size, k)
        topo = _make_topo(cfg)
        fields = build_fields(cfg, topo)
        b = topo.num_blocks

        assert fields.v.shape == (b, k)
        assert fields.u.shape == (b, k)
        assert fields.spikes.shape == (b, k)
        assert fields.I_syn.shape == (b, k)
        assert fields.I_ext.shape == (b, k)
        assert fields.calcium.shape == (b, k)
        assert fields.trace_pre.shape == (b, k)
        assert fields.trace_post.shape == (b, k)

    def test_param_shapes(
        self, ti_cpu: None, ndim: int, grid_size: int, k: int
    ) -> None:
        cfg = _make_cfg(ndim, grid_size, k)
        topo = _make_topo(cfg)
        fields = build_fields(cfg, topo)
        b = topo.num_blocks

        assert fields.param_a.shape == (b, k)
        assert fields.param_b.shape == (b, k)
        assert fields.param_c.shape == (b, k)
        assert fields.param_d.shape == (b, k)

    def test_weight_shapes(
        self, ti_cpu: None, ndim: int, grid_size: int, k: int
    ) -> None:
        cfg = _make_cfg(ndim, grid_size, k)
        topo = _make_topo(cfg)
        fields = build_fields(cfg, topo)
        b = topo.num_blocks
        n = topo.num_neighbors

        assert fields.W_intra.shape == (b, k, k)
        assert fields.W_inter.shape == (b, n, k, k)


class TestFieldInit:
    """Verify initialization values."""

    def test_voltage_init(self, ti_cpu: None) -> None:
        cfg = _make_cfg(2, 2, 16)
        topo = _make_topo(cfg)
        fields = build_fields(cfg, topo)
        init_fields(fields, cfg, topo)

        v = fields.v.to_numpy()
        assert np.allclose(v, -65.0)

    def test_u_equals_b_times_v(self, ti_cpu: None) -> None:
        cfg = _make_cfg(2, 2, 16)
        topo = _make_topo(cfg)
        fields = build_fields(cfg, topo)
        init_fields(fields, cfg, topo)

        u = fields.u.to_numpy()
        b_param = fields.param_b.to_numpy()
        v = fields.v.to_numpy()
        expected = b_param * v
        assert np.allclose(u, expected, atol=1e-5)

    def test_no_nan(self, ti_cpu: None) -> None:
        cfg = _make_cfg(2, 2, 16)
        topo = _make_topo(cfg)
        fields = build_fields(cfg, topo)
        init_fields(fields, cfg, topo)

        for arr in [
            fields.v.to_numpy(),
            fields.u.to_numpy(),
            fields.param_a.to_numpy(),
            fields.param_b.to_numpy(),
            fields.param_c.to_numpy(),
            fields.param_d.to_numpy(),
            fields.W_intra.to_numpy(),
            fields.W_inter.to_numpy(),
        ]:
            assert not np.any(np.isnan(arr)), "Found NaN in initialized fields"

    def test_no_self_connections(self, ti_cpu: None) -> None:
        cfg = _make_cfg(2, 2, 32)
        topo = _make_topo(cfg)
        fields = build_fields(cfg, topo)
        init_fields(fields, cfg, topo)

        w = fields.W_intra.to_numpy()
        for blk in range(w.shape[0]):
            diag = np.diag(w[blk])
            assert np.allclose(diag, 0.0), f"Self-connections found in block {blk}"

    def test_excitatory_ratio(self, ti_cpu: None) -> None:
        """~80% excitatory neurons (±5% tolerance)."""
        cfg = _make_cfg(2, 2, 100)  # K=100 for better statistics
        topo = _make_topo(cfg)
        fields = build_fields(cfg, topo)
        init_fields(fields, cfg, topo)

        # FS neurons have param_a=0.1, all excitatory have param_a=0.02
        a = fields.param_a.to_numpy()
        exc_count = np.sum(np.isclose(a, 0.02))
        total = a.size
        exc_ratio = exc_count / total
        assert abs(exc_ratio - 0.80) < 0.05, (
            f"Excitatory ratio {exc_ratio:.3f} not ~0.80"
        )

    def test_weight_signs_excitatory(self, ti_cpu: None) -> None:
        """Excitatory neurons should have non-negative outgoing weights."""
        cfg = _make_cfg(2, 2, 32)
        topo = _make_topo(cfg)
        fields = build_fields(cfg, topo)
        init_fields(fields, cfg, topo)

        w = fields.W_intra.to_numpy()
        a = fields.param_a.to_numpy()

        for blk in range(w.shape[0]):
            for pre in range(w.shape[1]):
                if np.isclose(a[blk, pre], 0.02):  # excitatory
                    assert np.all(w[blk, pre, :] >= 0.0), (
                        f"Excitatory neuron ({blk},{pre}) has negative outgoing weight"
                    )

    def test_weight_signs_inhibitory(self, ti_cpu: None) -> None:
        """Inhibitory neurons should have non-positive outgoing weights."""
        cfg = _make_cfg(2, 2, 32)
        topo = _make_topo(cfg)
        fields = build_fields(cfg, topo)
        init_fields(fields, cfg, topo)

        w = fields.W_intra.to_numpy()
        a = fields.param_a.to_numpy()

        for blk in range(w.shape[0]):
            for pre in range(w.shape[1]):
                if np.isclose(a[blk, pre], 0.1):  # inhibitory (FS)
                    assert np.all(w[blk, pre, :] <= 0.0), (
                        f"Inhibitory neuron ({blk},{pre}) has positive outgoing weight"
                    )

    def test_inter_weights_positive(self, ti_cpu: None) -> None:
        """W_inter should be non-negative (excitatory long-range)."""
        cfg = _make_cfg(2, 2, 16)
        topo = _make_topo(cfg)
        fields = build_fields(cfg, topo)
        init_fields(fields, cfg, topo)

        w = fields.W_inter.to_numpy()
        assert np.all(w >= 0.0), "W_inter has negative weights"

    def test_different_seeds(self, ti_cpu: None) -> None:
        """Different seeds should produce different weights."""
        cfg1 = _make_cfg(2, 2, 16, seed=1)
        cfg2 = _make_cfg(2, 2, 16, seed=2)
        topo = _make_topo(cfg1)

        f1 = build_fields(cfg1, topo)
        init_fields(f1, cfg1, topo)
        f2 = build_fields(cfg2, topo)
        init_fields(f2, cfg2, topo)

        w1 = f1.W_intra.to_numpy()
        w2 = f2.W_intra.to_numpy()
        assert not np.allclose(w1, w2), "Different seeds produced identical weights"
