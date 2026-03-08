"""Izhikevich neuron model kernel factory."""

from typing import Any

import taichi as ti

from hypertoric.kernels.neuron_models import register


@register("izhikevich")
def make_izhikevich_update(b: int, k: int) -> Any:
    """Factory returning a compiled Izhikevich update kernel with B, K baked in."""

    @ti.kernel
    def izhikevich_update(  # type: ignore[no-untyped-def]
        v: ti.template(),  # type: ignore[valid-type]
        u: ti.template(),  # type: ignore[valid-type]
        spikes: ti.template(),  # type: ignore[valid-type]
        i_ext: ti.template(),  # type: ignore[valid-type]
        param_a: ti.template(),  # type: ignore[valid-type]
        param_b: ti.template(),  # type: ignore[valid-type]
        param_c: ti.template(),  # type: ignore[valid-type]
        param_d: ti.template(),  # type: ignore[valid-type]
        dt: ti.f32,
    ):
        for block, neuron in ti.ndrange(b, k):
            v_val = v[block, neuron]
            u_val = u[block, neuron]
            i_val = i_ext[block, neuron]
            a = param_a[block, neuron]
            b_param = param_b[block, neuron]
            c = param_c[block, neuron]
            d = param_d[block, neuron]

            # Half-step Euler for v (two 0.5*dt updates for numerical stability)
            half_dt = 0.5 * dt
            dv = 0.04 * v_val * v_val + 5.0 * v_val + 140.0 - u_val + i_val
            v_val += half_dt * dv
            dv = 0.04 * v_val * v_val + 5.0 * v_val + 140.0 - u_val + i_val
            v_val += half_dt * dv

            # Full step for u
            u_val += dt * a * (b_param * v_val - u_val)

            # Spike detection and reset
            if v_val >= 30.0:
                v[block, neuron] = c
                u[block, neuron] = u_val + d
                spikes[block, neuron] = 1
            else:
                v[block, neuron] = v_val
                u[block, neuron] = u_val
                spikes[block, neuron] = 0

    return izhikevich_update
