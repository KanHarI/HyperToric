# WC-4: Neuron Kernel

**Branch:** `wc-4-neuron-kernel`
**Dependencies:** WC-1, WC-2

## Goal

Implement the Izhikevich neuron model as a Taichi kernel, wrapped in a registry-based factory pattern that allows swapping neuron models via config. This chapter focuses on the neuron update step only — synaptic input comes from WC-5.

## Files

| File | Action | Notes |
|------|--------|-------|
| `src/hypertoric/kernels/__init__.py` | Create | KernelSet dataclass, build_kernels stub |
| `src/hypertoric/kernels/neuron_models/__init__.py` | Create | Registry: NEURON_MODELS dict, get_neuron_factory() |
| `src/hypertoric/kernels/neuron_models/izhikevich.py` | Create | Izhikevich factory + kernel |
| `tests/test_neuron.py` | Create | Spike behavior, numerical stability |

## Architecture: Kernel Factory Pattern

### Why Factories?

Taichi kernels are compiled with fixed field shapes. `B` (num_blocks) and `K` (neurons_per_block) vary by config. We can't write a single kernel that works for all shapes — Taichi needs the loop bounds at compile time for GPU thread mapping.

The factory pattern solves this: a function takes `(B, K)` and returns a compiled kernel with those values baked in via closure.

```python
def make_izhikevich_update(B: int, K: int) -> Callable[..., None]:
    @ti.kernel
    def neuron_update(v: ti.template(), ...):
        for block_idx, i in ti.ndrange(B, K):  # B, K captured from closure
            ...
    return neuron_update
```

`ti.ndrange(B, K)` gets the values from the closure at compile time. Taichi traces through the Python closure during compilation and embeds the constants. This is equivalent to C++ template instantiation — zero runtime overhead.

### Neuron Model Registry

New neuron models (LIF, AdEx, etc.) should be addable without modifying existing code. The registry pattern:

```python
# kernels/neuron_models/__init__.py
NEURON_MODELS: dict[str, NeuronModelFactory] = {}

def register(name: str) -> Callable:
    def decorator(factory: NeuronModelFactory) -> NeuronModelFactory:
        NEURON_MODELS[name] = factory
        return factory
    return decorator

def get_neuron_factory(name: str) -> NeuronModelFactory:
    if name not in NEURON_MODELS:
        raise ValueError(f"Unknown neuron model: {name!r}. Available: {list(NEURON_MODELS)}")
    return NEURON_MODELS[name]
```

Each model file (izhikevich.py, future lif.py, etc.) imports `register` and decorates its factory. The `__init__.py` imports all model modules to trigger registration.

**NeuronUpdateFn protocol** — all factories return a callable with the same signature:
```python
class NeuronUpdateFn(Protocol):
    def __call__(
        self, v: Any, u: Any, spikes: Any,
        I_syn: Any, I_ext: Any,
        param_a: Any, param_b: Any, param_c: Any, param_d: Any,
        dt: float,
    ) -> None: ...
```

The simulator doesn't know or care which model is running. It calls `neuron_update(fields.v, fields.u, ...)` regardless.

### Config Integration

`NeuronConfig.model: str = "izhikevich"` selects the model. The simulator (WC-8) calls `get_neuron_factory(config.neuron.model)` at construction time.

## Izhikevich Implementation

### The Model

Two coupled ODEs (Izhikevich 2003):

```
dv/dt = 0.04v² + 5v + 140 - u + I
du/dt = a(bv - u)
if v ≥ 30: spike → v = c, u += d
```

### Numerical Integration: Half-Step Euler

Plain forward Euler with `dt = 0.5ms` is unstable for the Izhikevich model near the spike threshold because of the `0.04v²` term — v can overshoot 30mV and diverge to infinity in a single step.

Half-step Euler applies the voltage update twice with `0.5 * dt`:

```python
vv += 0.5 * dt * (0.04 * vv * vv + 5.0 * vv + 140.0 - uu + I_total)
vv += 0.5 * dt * (0.04 * vv * vv + 5.0 * vv + 140.0 - uu + I_total)
```

Note: this is NOT the same as RK2/midpoint. It's two half-sized Euler steps, which is slightly more stable than one full step because the second half uses the updated `v` from the first half. Izhikevich himself recommends this in his 2003 paper.

The recovery variable `u` uses a single full Euler step — it evolves slowly (governed by `a`, which is 0.02-0.1) and doesn't need the extra stability.

### Spike Detection and Reset

```python
if vv >= 30.0:
    spikes[block_idx, i] = 1
    vv = param_c[block_idx, i]    # reset voltage (type-dependent)
    uu += param_d[block_idx, i]   # recovery kick (type-dependent)
else:
    spikes[block_idx, i] = 0
```

Key detail: the spike flag is set to 1 for exactly one timestep. The propagation kernel (WC-5) reads `spikes` to compute synaptic input. The flag must be cleared (set to 0) on every non-spiking step. Don't rely on a separate "clear spikes" kernel — set it inline.

### Current Combination

```python
I_total = I_syn[block_idx, i] + I_ext[block_idx, i]
```

`I_syn` is the total synaptic current from other neurons (computed by propagation kernel). `I_ext` is externally injected current (set by the I/O manager for sensory/feedback input). They're separate fields because `I_syn` decays exponentially (managed by propagation) while `I_ext` is set directly each timestep.

### Neuron Types and Their Behavior

| Type | a | b | c | d | Behavior |
|------|-----|-----|------|-----|----------|
| RS | 0.02 | 0.2 | -65 | 8 | Adapting — spike rate decreases under sustained input |
| FS | 0.1 | 0.2 | -65 | 2 | Non-adapting — fires at constant rate, fast response |
| CH | 0.02 | 0.2 | -50 | 2 | Chattering — bursts of closely-spaced spikes |
| IB | 0.02 | 0.2 | -55 | 4 | Initial burst then regular spiking |

The `c` parameter (reset voltage) is the main differentiator. RS resets to -65 (deep reset, slow to spike again), while CH resets to -50 (shallow reset, fires again quickly → bursts).

### GPU Considerations

Each thread handles one `(block_idx, i)` pair — one neuron. The kernel is embarrassingly parallel with no data dependencies between neurons (synaptic input was computed in the previous step). Memory access is coalesced because consecutive threads access consecutive `i` indices within the same block.

No shared memory, no atomics, no synchronization. This kernel will always be memory-bandwidth-limited, not compute-limited — the arithmetic is trivial (~10 FLOPs) but each neuron touches 9 field reads and 3 field writes.

## Tests

All tests use CPU backend, small sizes (`B=1, K=4` or `B=2, K=8`) for speed.

### Spike Threshold Tests

**Suprathreshold input → spikes within expected range:**
- Set `I_ext[0, 0] = 15.0` (strong input for RS neuron).
- Run 100 timesteps. Assert at least one spike occurs.
- Verify spike timing: RS neuron with I=15 should spike within ~10-20ms (20-40 steps at dt=0.5).

**Subthreshold input → no spikes:**
- Set `I_ext[0, 0] = 3.0` (below rheobase for RS).
- Run 10,000 timesteps. Assert zero spikes.
- This tests that the equilibrium is stable — the neuron should settle to a resting potential, not slowly drift.

### Reset Mechanics

**After spike, v resets to c:**
- Drive a neuron to spike. On the timestep after the spike, check `v == c`.
- For RS: `v == -65`. For CH: `v == -50`.

**After spike, u incremented by d:**
- Record `u` before and after spike. Difference should be `d`.
- For RS: `d == 8` (big recovery kick → long inter-spike interval).

### Numerical Stability

**No NaN or divergence over 10k steps:**
- Run with various input levels (0, 5, 15, 50) for 10,000 steps.
- Assert no NaN in `v` or `u` fields at any point.
- Assert `v` stays bounded (e.g., `v <= 30.0` always — any value above 30 should have triggered a reset).

### Multi-Type Parametrization

Parametrize spike tests over all four types (RS, FS, CH, IB):
- Each type should spike with sufficient input.
- FS should have the shortest inter-spike interval (fastest response).
- CH should produce bursts (multiple spikes in quick succession, then pause).

### Isolation Test

Run two blocks (`B=2`), inject current into block 0 only. Block 1 should have zero spikes. This verifies the kernel doesn't accidentally cross block boundaries (no off-by-one in `ti.ndrange`).
