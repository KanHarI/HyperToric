"""Simulator orchestration: owns topology, fields, kernels, and step logic."""

import math
from typing import TYPE_CHECKING, Any

import taichi as ti

from hypertoric.config import SimConfig, validate_config
from hypertoric.fields import SimFields, build_fields, init_fields
from hypertoric.kernels import build_kernels
from hypertoric.topology import Topology

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class Simulator:
    """Simulation orchestrator for the HyperToric neural network."""

    def __init__(self, cfg: SimConfig) -> None:
        validate_config(cfg)

        self._cfg = cfg
        self._topo = Topology(cfg.torus.ndim, cfg.torus.grid_size)

        b = self._topo.num_blocks
        k = cfg.torus.neurons_per_block

        self._fields = build_fields(cfg, self._topo)
        init_fields(self._fields, cfg, self._topo)

        self._kernels = build_kernels(
            b, k, cfg.torus.grid_size, cfg.torus.ndim, cfg.neuron.model
        )

        self._decay_factor = math.exp(-cfg.neuron.dt / cfg.neuron.tau_syn)

        # Scratch field for combined currents
        self._i_total: ti.Field = ti.field(dtype=ti.f32, shape=(b, k))
        self._combine_kernel = _make_combine_currents(b, k)

        self._step_count = 0
        self._stdp_direction = 0
        self._num_neighbors = self._topo.num_neighbors

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def fields(self) -> "SimFields":
        return self._fields

    @property
    def config(self) -> "SimConfig":
        return self._cfg

    @property
    def topology(self) -> "Topology":
        return self._topo

    def inject_current(self, current: "NDArray[np.float32]") -> None:
        """Set external current field from numpy array of shape (B, K)."""
        self._fields.I_ext.from_numpy(current)

    def read_spikes(self) -> "NDArray[np.int32]":
        """Read spike field as numpy array of shape (B, K)."""
        result: NDArray[np.int32] = self._fields.spikes.to_numpy()
        return result

    def read_voltage(self) -> "NDArray[np.float32]":
        """Read voltage field as numpy array of shape (B, K)."""
        result: NDArray[np.float32] = self._fields.v.to_numpy()
        return result

    def step(self) -> None:
        """Advance simulation by one timestep."""
        f = self._fields
        k = self._kernels
        cfg = self._cfg
        dt = cfg.neuron.dt

        # 1. Combine currents: I_total = I_syn + I_ext
        self._combine_kernel(self._i_total, f.I_syn, f.I_ext)

        # 2. Neuron update (produces spikes)
        k.neuron_update(
            f.v,
            f.u,
            f.spikes,
            self._i_total,
            f.param_a,
            f.param_b,
            f.param_c,
            f.param_d,
            dt,
        )

        # 3. Spike propagation (updates I_syn for next step)
        k.spike_propagate(
            f.I_syn,
            f.spikes,
            f.W_intra,
            f.W_inter,
            self._decay_factor,
        )

        # 4. Trace update
        k.trace_update(
            f.trace_pre,
            f.trace_post,
            f.spikes,
            cfg.stdp.tau_pre,
            cfg.stdp.tau_post,
            dt,
        )

        # 5. STDP intra
        k.stdp_intra(
            f.W_intra,
            f.spikes,
            f.trace_pre,
            f.trace_post,
            cfg.stdp.a_plus,
            cfg.stdp.a_minus,
            cfg.plasticity.w_max,
        )

        # 6. STDP inter (rotating or all)
        if cfg.stdp.inter_mode == "all":
            for d in range(self._num_neighbors):
                k.stdp_inter(
                    f.W_inter,
                    f.spikes,
                    f.trace_pre,
                    f.trace_post,
                    d,
                    cfg.stdp.a_plus,
                    cfg.stdp.a_minus,
                    cfg.plasticity.w_max,
                )
        else:
            k.stdp_inter(
                f.W_inter,
                f.spikes,
                f.trace_pre,
                f.trace_post,
                self._stdp_direction,
                cfg.stdp.a_plus,
                cfg.stdp.a_minus,
                cfg.plasticity.w_max,
            )
            self._stdp_direction = (self._stdp_direction + 1) % self._num_neighbors

        # 7. Calcium update
        k.calcium_update(f.calcium, f.spikes, cfg.plasticity.calcium_tau, dt)

        # 8. Structural plasticity (periodic)
        self._step_count += 1

        if self._step_count % cfg.plasticity.interval == 0:
            k.structural_intra(
                f.W_intra,
                f.calcium,
                cfg.plasticity.calcium_threshold_low,
                cfg.plasticity.calcium_threshold_high,
                cfg.plasticity.weight_threshold,
                cfg.plasticity.init_weight,
                cfg.plasticity.w_max,
            )

        if self._step_count % cfg.plasticity.inter_interval == 0:
            for d in range(self._num_neighbors):
                k.structural_inter(
                    f.W_inter,
                    f.calcium,
                    cfg.plasticity.calcium_threshold_low,
                    cfg.plasticity.calcium_threshold_high,
                    cfg.plasticity.weight_threshold,
                    cfg.plasticity.init_weight,
                    cfg.plasticity.w_max,
                    d,
                )


def _make_combine_currents(b: int, k: int) -> Any:
    """Factory for a kernel that combines I_syn and I_ext into I_total."""

    @ti.kernel
    def combine_currents(  # type: ignore[no-untyped-def]
        i_total: ti.template(),  # type: ignore[valid-type]
        i_syn: ti.template(),  # type: ignore[valid-type]
        i_ext: ti.template(),  # type: ignore[valid-type]
    ):
        for block_idx, i in ti.ndrange(b, k):
            i_total[block_idx, i] = i_syn[block_idx, i] + i_ext[block_idx, i]

    return combine_currents
