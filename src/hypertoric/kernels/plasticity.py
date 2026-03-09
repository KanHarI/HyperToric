"""Structural plasticity kernel factories: calcium update, intra-block, inter-block."""

from typing import Any

import taichi as ti


def make_calcium_update(b: int, k: int) -> Any:
    """Factory returning a compiled calcium update kernel."""

    @ti.kernel
    def calcium_update(  # type: ignore[no-untyped-def]
        calcium: ti.template(),  # type: ignore[valid-type]
        spikes: ti.template(),  # type: ignore[valid-type]
        calcium_tau: ti.f32,
        dt: ti.f32,
    ):
        for block_idx, i in ti.ndrange(b, k):
            calcium[block_idx, i] *= ti.exp(-dt / calcium_tau)
            if spikes[block_idx, i] == 1:
                calcium[block_idx, i] += 1.0

    return calcium_update


def make_structural_intra(b: int, k: int) -> Any:
    """Factory returning a compiled intra-block structural plasticity kernel."""

    @ti.kernel
    def structural_intra(  # type: ignore[no-untyped-def]
        w_intra: ti.template(),  # type: ignore[valid-type]
        calcium: ti.template(),  # type: ignore[valid-type]
        calcium_low: ti.f32,
        calcium_high: ti.f32,
        weight_threshold: ti.f32,
        init_weight: ti.f32,
        w_max: ti.f32,
    ):
        # Pass 1: threshold pruning
        for block_idx, j, i in ti.ndrange(b, k, k):
            w = w_intra[block_idx, j, i]
            if w != 0.0 and ti.abs(w) < weight_threshold:
                w_intra[block_idx, j, i] = 0.0

        # Pass 2: growth/pruning based on calcium
        for block_idx, i in ti.ndrange(b, k):
            ca = calcium[block_idx, i]
            if ca < calcium_low:
                # Grow one synapse: pick random slot
                j = ti.cast(ti.random() * k, ti.i32)
                j = ti.min(j, k - 1)
                if j != i and w_intra[block_idx, j, i] == 0.0:
                    w_intra[block_idx, j, i] = init_weight
            elif ca > calcium_high:
                # Prune one synapse: pick random non-zero slot
                j = ti.cast(ti.random() * k, ti.i32)
                j = ti.min(j, k - 1)
                if w_intra[block_idx, j, i] != 0.0:
                    w_intra[block_idx, j, i] = 0.0

    return structural_intra


def make_structural_inter(b: int, k: int, grid_size: int, ndim: int) -> Any:
    """Factory returning a compiled inter-block structural plasticity kernel."""

    @ti.kernel
    def structural_inter(  # type: ignore[no-untyped-def]
        w_inter: ti.template(),  # type: ignore[valid-type]
        calcium: ti.template(),  # type: ignore[valid-type]
        calcium_low: ti.f32,
        calcium_high: ti.f32,
        weight_threshold: ti.f32,
        init_weight: ti.f32,
        w_max: ti.f32,
        direction: ti.i32,
    ):
        # Pass 1: threshold pruning for this direction
        for block_idx, j, i in ti.ndrange(b, k, k):
            w = w_inter[block_idx, direction, j, i]
            if w != 0.0 and ti.abs(w) < weight_threshold:
                w_inter[block_idx, direction, j, i] = 0.0

        # Pass 2: growth/pruning based on calcium
        for block_idx, i in ti.ndrange(b, k):
            ca = calcium[block_idx, i]
            if ca < calcium_low:
                # Grow one synapse (always positive/excitatory)
                j = ti.cast(ti.random() * k, ti.i32)
                j = ti.min(j, k - 1)
                if w_inter[block_idx, direction, j, i] == 0.0:
                    w_inter[block_idx, direction, j, i] = init_weight
            elif ca > calcium_high:
                # Prune one synapse
                j = ti.cast(ti.random() * k, ti.i32)
                j = ti.min(j, k - 1)
                if w_inter[block_idx, direction, j, i] != 0.0:
                    w_inter[block_idx, direction, j, i] = 0.0

    return structural_inter
