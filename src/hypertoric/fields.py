"""Taichi field allocation and initialization for neural simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hypertoric.config import SimConfig
    from hypertoric.topology import Topology

# Izhikevich neuron type parameters: (a, b, c, d)
_NEURON_PARAMS: dict[str, tuple[float, float, float, float]] = {
    "RS": (0.02, 0.2, -65.0, 8.0),
    "IB": (0.02, 0.2, -55.0, 4.0),
    "CH": (0.02, 0.2, -50.0, 2.0),
    "FS": (0.1, 0.2, -65.0, 2.0),
}

# Excitatory subtypes and their relative proportions (must sum to 1.0)
_EXC_SUBTYPES: list[tuple[str, float]] = [("RS", 0.75), ("IB", 0.19), ("CH", 0.06)]


@dataclass
class SimFields:
    """Container for all Taichi fields used in the simulation."""

    # Per-neuron state (B, K)
    v: ti.Field
    u: ti.Field
    spikes: ti.Field
    I_syn: ti.Field
    I_ext: ti.Field
    calcium: ti.Field
    trace_pre: ti.Field
    trace_post: ti.Field

    # Per-neuron parameters (B, K)
    param_a: ti.Field
    param_b: ti.Field
    param_c: ti.Field
    param_d: ti.Field

    # Weights
    W_intra: ti.Field  # (B, K, K)
    W_inter: ti.Field  # (B, N, K, K) where N = 2*ndim


def build_fields(cfg: SimConfig, topo: Topology) -> SimFields:
    """Allocate Taichi fields based on config and topology."""
    b = topo.num_blocks
    k = cfg.torus.neurons_per_block
    n = topo.num_neighbors

    # State fields (B, K)
    v = ti.field(dtype=ti.f32, shape=(b, k))
    u = ti.field(dtype=ti.f32, shape=(b, k))
    spikes = ti.field(dtype=ti.i32, shape=(b, k))
    i_syn = ti.field(dtype=ti.f32, shape=(b, k))
    i_ext = ti.field(dtype=ti.f32, shape=(b, k))
    calcium = ti.field(dtype=ti.f32, shape=(b, k))
    trace_pre = ti.field(dtype=ti.f32, shape=(b, k))
    trace_post = ti.field(dtype=ti.f32, shape=(b, k))

    # Parameter fields (B, K)
    param_a = ti.field(dtype=ti.f32, shape=(b, k))
    param_b = ti.field(dtype=ti.f32, shape=(b, k))
    param_c = ti.field(dtype=ti.f32, shape=(b, k))
    param_d = ti.field(dtype=ti.f32, shape=(b, k))

    # Weight fields
    w_intra = ti.field(dtype=ti.f32, shape=(b, k, k))
    w_inter = ti.field(dtype=ti.f32, shape=(b, n, k, k))

    return SimFields(
        v=v,
        u=u,
        spikes=spikes,
        I_syn=i_syn,
        I_ext=i_ext,
        calcium=calcium,
        trace_pre=trace_pre,
        trace_post=trace_post,
        param_a=param_a,
        param_b=param_b,
        param_c=param_c,
        param_d=param_d,
        W_intra=w_intra,
        W_inter=w_inter,
    )


def init_fields(fields: SimFields, cfg: SimConfig, topo: Topology) -> None:
    """Initialize field values using numpy, then copy to Taichi fields."""
    b = topo.num_blocks
    k = cfg.torus.neurons_per_block
    n = topo.num_neighbors
    rng = np.random.default_rng(cfg.seed)

    # --- Neuron type assignment and parameter initialization ---
    exc_ratio = cfg.neuron.excitatory_ratio
    np_a = np.zeros((b, k), dtype=np.float32)
    np_b = np.zeros((b, k), dtype=np.float32)
    np_c = np.zeros((b, k), dtype=np.float32)
    np_d = np.zeros((b, k), dtype=np.float32)
    is_excitatory = np.zeros((b, k), dtype=bool)

    for blk in range(b):
        n_exc = int(round(exc_ratio * k))
        n_inh = k - n_exc

        # Assign excitatory subtypes
        exc_types = _assign_exc_subtypes(n_exc, rng)
        # Inhibitory neurons are all FS
        inh_types = ["FS"] * n_inh

        # Combine and shuffle within block
        all_types = exc_types + inh_types
        perm = rng.permutation(k)

        for local_idx in range(k):
            ntype = all_types[perm[local_idx]]
            a_val, b_val, c_val, d_val = _NEURON_PARAMS[ntype]
            np_a[blk, local_idx] = a_val
            np_b[blk, local_idx] = b_val
            np_c[blk, local_idx] = c_val
            np_d[blk, local_idx] = d_val
            is_excitatory[blk, local_idx] = ntype != "FS"

    # --- State initialization ---
    np_v = np.full((b, k), -65.0, dtype=np.float32)
    np_u = (np_b * np_v).astype(np.float32)

    fields.v.from_numpy(np_v)
    fields.u.from_numpy(np_u)
    fields.spikes.from_numpy(np.zeros((b, k), dtype=np.int32))
    fields.I_syn.from_numpy(np.zeros((b, k), dtype=np.float32))
    fields.I_ext.from_numpy(np.zeros((b, k), dtype=np.float32))
    fields.calcium.from_numpy(np.zeros((b, k), dtype=np.float32))
    fields.trace_pre.from_numpy(np.zeros((b, k), dtype=np.float32))
    fields.trace_post.from_numpy(np.zeros((b, k), dtype=np.float32))

    fields.param_a.from_numpy(np_a)
    fields.param_b.from_numpy(np_b)
    fields.param_c.from_numpy(np_c)
    fields.param_d.from_numpy(np_d)

    # --- W_intra: ~20-30% sparse, sign depends on presynaptic type ---
    np_w_intra = np.zeros((b, k, k), dtype=np.float32)
    for blk in range(b):
        density = rng.uniform(0.20, 0.30)
        mask = rng.random((k, k)) < density
        # No self-connections
        np.fill_diagonal(mask, False)
        weights = rng.uniform(0.0, 1.0, size=(k, k)).astype(np.float32)
        weights *= mask
        # Sign: excitatory pre → positive, inhibitory pre → negative
        for pre in range(k):
            if not is_excitatory[blk, pre]:
                weights[pre, :] = -np.abs(weights[pre, :])
        np_w_intra[blk] = weights

    fields.W_intra.from_numpy(np_w_intra)

    # --- W_inter: ~5-10% sparse, positive only (excitatory long-range) ---
    np_w_inter = np.zeros((b, n, k, k), dtype=np.float32)
    for blk in range(b):
        for nbr in range(n):
            density = rng.uniform(0.05, 0.10)
            mask = rng.random((k, k)) < density
            weights = rng.uniform(0.0, 1.0, size=(k, k)).astype(np.float32)
            weights *= mask
            np_w_inter[blk, nbr] = weights

    fields.W_inter.from_numpy(np_w_inter)


def _assign_exc_subtypes(n_exc: int, rng: np.random.Generator) -> list[str]:
    """Assign excitatory neuron subtypes based on fixed proportions."""
    types: list[str] = []
    remaining = n_exc
    for i, (subtype, proportion) in enumerate(_EXC_SUBTYPES):
        if i == len(_EXC_SUBTYPES) - 1:
            count = remaining
        else:
            count = int(round(proportion * n_exc))
            count = min(count, remaining)
        types.extend([subtype] * count)
        remaining -= count
    # Shuffle to avoid ordering artifacts
    shuffled: NDArray[np.intp] = rng.permutation(len(types))
    return [types[i] for i in shuffled]
