"""Neuron model registry with factory pattern."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import taichi as ti


class NeuronUpdateFn(Protocol):
    """Protocol for a compiled neuron update kernel."""

    def __call__(
        self,
        v: ti.Field,
        u: ti.Field,
        spikes: ti.Field,
        i_ext: ti.Field,
        param_a: ti.Field,
        param_b: ti.Field,
        param_c: ti.Field,
        param_d: ti.Field,
        dt: float,
    ) -> None: ...


NeuronModelFactory = Callable[[int, int], Any]

NEURON_MODELS: dict[str, NeuronModelFactory] = {}

_F = Callable[[int, int], Any]


def register(name: str) -> Callable[[_F], _F]:
    """Decorator to register a neuron model factory."""

    def decorator(fn: _F) -> _F:
        NEURON_MODELS[name] = fn
        return fn

    return decorator


def get_neuron_factory(name: str) -> NeuronModelFactory:
    """Look up a neuron model factory by name."""
    if name not in NEURON_MODELS:
        available = ", ".join(sorted(NEURON_MODELS.keys()))
        msg = f"Unknown neuron model {name!r}. Available: {available}"
        raise KeyError(msg)
    return NEURON_MODELS[name]


# Import submodules to trigger registration
import hypertoric.kernels.neuron_models.izhikevich as _  # noqa: E402, F401
