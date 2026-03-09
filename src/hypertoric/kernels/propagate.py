"""Spike propagation kernel factory."""

from typing import Any

import taichi as ti


def make_spike_propagate(b: int, k: int, grid_size: int, ndim: int) -> Any:
    """Factory returning a compiled spike propagation kernel."""

    num_neighbors = 2 * ndim
    strides = tuple(grid_size**a for a in range(ndim))

    @ti.kernel
    def spike_propagate(  # type: ignore[no-untyped-def]
        i_syn: ti.template(),  # type: ignore[valid-type]
        spikes: ti.template(),  # type: ignore[valid-type]
        w_intra: ti.template(),  # type: ignore[valid-type]
        w_inter: ti.template(),  # type: ignore[valid-type]
        decay_factor: ti.f32,
    ):
        for block_idx, i in ti.ndrange(b, k):
            acc = ti.cast(0.0, ti.f32)

            # Intra-block: spikes from all neurons in same block
            for j in range(k):
                acc += w_intra[block_idx, j, i] * ti.cast(spikes[block_idx, j], ti.f32)

            # Inter-block: spikes from all neighbor blocks
            for d in ti.static(range(num_neighbors)):
                # Inline neighbor computation with compile-time constants
                sign = ti.static(1 - 2 * (d % 2))
                stride = ti.static(strides[d // 2])
                coord = (block_idx // stride) % grid_size
                new_coord = (coord + sign + grid_size) % grid_size
                nb = block_idx + (new_coord - coord) * stride
                for j in range(k):
                    acc += w_inter[block_idx, d, j, i] * ti.cast(spikes[nb, j], ti.f32)

            # Exponential decay + new input
            i_syn[block_idx, i] = i_syn[block_idx, i] * decay_factor + acc

    return spike_propagate
