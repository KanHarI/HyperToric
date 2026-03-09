"""STDP kernel factories: trace update, intra-block, and inter-block."""

from typing import Any

import taichi as ti


def make_trace_update(b: int, k: int) -> Any:
    """Factory returning a compiled trace update kernel."""

    @ti.kernel
    def trace_update(  # type: ignore[no-untyped-def]
        trace_pre: ti.template(),  # type: ignore[valid-type]
        trace_post: ti.template(),  # type: ignore[valid-type]
        spikes: ti.template(),  # type: ignore[valid-type]
        tau_pre: ti.f32,
        tau_post: ti.f32,
        dt: ti.f32,
    ):
        for block_idx, i in ti.ndrange(b, k):
            trace_pre[block_idx, i] *= ti.exp(-dt / tau_pre)
            trace_post[block_idx, i] *= ti.exp(-dt / tau_post)
            if spikes[block_idx, i] == 1:
                trace_pre[block_idx, i] += 1.0
                trace_post[block_idx, i] += 1.0

    return trace_update


def make_stdp_intra(b: int, k: int) -> Any:
    """Factory returning a compiled intra-block STDP kernel."""

    @ti.kernel
    def stdp_intra(  # type: ignore[no-untyped-def]
        w_intra: ti.template(),  # type: ignore[valid-type]
        spikes: ti.template(),  # type: ignore[valid-type]
        trace_pre: ti.template(),  # type: ignore[valid-type]
        trace_post: ti.template(),  # type: ignore[valid-type]
        a_plus: ti.f32,
        a_minus: ti.f32,
        w_max: ti.f32,
    ):
        # Pass 1: LTP — post-synaptic neuron i spiked
        for block_idx, i in ti.ndrange(b, k):
            if spikes[block_idx, i] == 1:
                for j in range(k):
                    w_orig = w_intra[block_idx, j, i]
                    if w_orig != 0.0:
                        dw = a_plus * trace_pre[block_idx, j]
                        w = w_orig + dw
                        # Clamp based on original sign
                        if w_orig > 0.0:  # noqa: SIM108
                            w = ti.min(w, w_max)
                        else:
                            w = ti.min(w, 0.0)
                        w_intra[block_idx, j, i] = w

        # Pass 2: LTD — pre-synaptic neuron j spiked
        for block_idx, j in ti.ndrange(b, k):
            if spikes[block_idx, j] == 1:
                for i in range(k):
                    w_orig = w_intra[block_idx, j, i]
                    if w_orig != 0.0:
                        dw = a_minus * trace_post[block_idx, i]
                        w = w_orig - dw
                        # Clamp based on original sign
                        if w_orig > 0.0:  # noqa: SIM108
                            w = ti.max(w, 0.0)
                        else:
                            w = ti.max(w, -w_max)
                        w_intra[block_idx, j, i] = w

    return stdp_intra


def make_stdp_inter(b: int, k: int, grid_size: int, ndim: int) -> Any:
    """Factory returning a compiled inter-block STDP kernel."""

    strides = tuple(grid_size**a for a in range(ndim))

    @ti.kernel
    def stdp_inter(  # type: ignore[no-untyped-def]
        w_inter: ti.template(),  # type: ignore[valid-type]
        spikes: ti.template(),  # type: ignore[valid-type]
        trace_pre: ti.template(),  # type: ignore[valid-type]
        trace_post: ti.template(),  # type: ignore[valid-type]
        direction: ti.i32,
        a_plus: ti.f32,
        a_minus: ti.f32,
        w_max: ti.f32,
    ):
        for block_idx, i in ti.ndrange(b, k):
            # Compute neighbor for this direction
            axis = direction // 2
            sign = 1 - 2 * (direction % 2)
            stride = ti.i32(1)
            for a in ti.static(range(ndim)):
                if axis == a:
                    stride = ti.static(strides[a])
            coord = (block_idx // stride) % grid_size
            new_coord = (coord + sign + grid_size) % grid_size
            nb = block_idx + (new_coord - coord) * stride

            # LTP: post i in this block spiked
            if spikes[block_idx, i] == 1:
                for j in range(k):
                    w = w_inter[block_idx, direction, j, i]
                    if w != 0.0:
                        dw = a_plus * trace_pre[nb, j]
                        w += dw
                        w = ti.min(w, w_max)
                        w_inter[block_idx, direction, j, i] = w

            # LTD: pre j in neighbor spiked
            for j in range(k):
                if spikes[nb, j] == 1:
                    w = w_inter[block_idx, direction, j, i]
                    if w != 0.0:
                        dw = a_minus * trace_post[block_idx, i]
                        w -= dw
                        w = ti.max(w, 0.0)
                        w_inter[block_idx, direction, j, i] = w

    return stdp_inter
