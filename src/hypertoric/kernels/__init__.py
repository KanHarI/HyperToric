"""Kernel set container and builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class KernelSet:
    """Container for all compiled Taichi kernels used in the simulation."""

    neuron_update: Any  # NeuronUpdateFn — compiled @ti.kernel
    trace_update: Any  # compiled @ti.kernel
    stdp_intra: Any  # compiled @ti.kernel
    stdp_inter: Any  # compiled @ti.kernel


def build_kernels(
    b: int,
    k: int,
    grid_size: int,
    ndim: int,
    model_name: str = "izhikevich",
) -> KernelSet:
    """Build all simulation kernels with dimensions baked in."""
    from hypertoric.kernels.neuron_models import get_neuron_factory
    from hypertoric.kernels.stdp import (
        make_stdp_inter,
        make_stdp_intra,
        make_trace_update,
    )

    factory = get_neuron_factory(model_name)
    neuron_update = factory(b, k)
    trace_update = make_trace_update(b, k)
    stdp_intra = make_stdp_intra(b, k)
    stdp_inter = make_stdp_inter(b, k, grid_size, ndim)
    return KernelSet(
        neuron_update=neuron_update,
        trace_update=trace_update,
        stdp_intra=stdp_intra,
        stdp_inter=stdp_inter,
    )
