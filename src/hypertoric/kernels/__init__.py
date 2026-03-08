"""Kernel set container and builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class KernelSet:
    """Container for all compiled Taichi kernels used in the simulation."""

    neuron_update: Any  # NeuronUpdateFn — compiled @ti.kernel


def build_kernels(b: int, k: int, model_name: str = "izhikevich") -> KernelSet:
    """Build all simulation kernels with dimensions baked in."""
    from hypertoric.kernels.neuron_models import get_neuron_factory

    factory = get_neuron_factory(model_name)
    neuron_update = factory(b, k)
    return KernelSet(neuron_update=neuron_update)
