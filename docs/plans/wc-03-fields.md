# WC-3: Taichi Fields

**Branch:** `wc-3-fields`
**Dependencies:** WC-1, WC-2

## Goal

Allocate all Taichi fields (GPU-resident arrays) for the simulation and initialize them to biologically plausible starting values. This module is the bridge between Python config and GPU memory — it translates `SimConfig` into the flat-indexed field layout that all kernels operate on.

## Files

| File | Action | Notes |
|------|--------|-------|
| `src/hypertoric/fields.py` | Create | SimFields dataclass, build_fields, init_fields |
| `tests/test_fields.py` | Create | Shape checks, init value checks |

## Flat Indexing — Why and How

The architecture doc shows fields shaped `(GRID, GRID, GRID, K)` — 3D grid coordinates plus neuron index. The implementation uses flat 1D block indices instead: `(B, K)` where `B = grid_size^ndim`.

**Why flat:**
1. Kernels iterate `for block_idx, i in ti.ndrange(B, K)` — one loop, no ndim-specific code.
2. Inter-block weight arrays use `(B, N, K, K)` where `N = 2*ndim`. With grid coordinates, this would be `(G, G, G, 6, K, K)` for 3D — shape changes with ndim.
3. Neighbor computation in kernels uses the modular arithmetic from topology.py, which operates on flat indices directly.

**Memory layout:** Taichi allocates fields in row-major order. For `(B, K)`, neurons within a block are contiguous. This is good — the inner loops in propagation iterate over `K` within a block, hitting contiguous memory.

## SimFields — What Gets Allocated

### Per-Neuron State: shape `(B, K)`

| Field | dtype | Purpose | Initial Value |
|-------|-------|---------|---------------|
| `v` | f32 | Membrane potential (mV) | -65.0 |
| `u` | f32 | Recovery variable | `b * v` (param-dependent) |
| `spikes` | i32 | Binary spike flag this timestep | 0 |
| `I_syn` | f32 | Total synaptic current | 0.0 |
| `I_ext` | f32 | External current injection | 0.0 |
| `calcium` | f32 | Exponential spike trace for plasticity | 0.0 |
| `trace_pre` | f32 | Pre-synaptic STDP trace | 0.0 |
| `trace_post` | f32 | Post-synaptic STDP trace | 0.0 |

### Per-Neuron Parameters: shape `(B, K)`

| Field | dtype | Purpose | Value |
|-------|-------|---------|-------|
| `param_a` | f32 | Recovery time constant | Type-dependent |
| `param_b` | f32 | Recovery sensitivity | Type-dependent |
| `param_c` | f32 | Post-spike reset voltage | Type-dependent |
| `param_d` | f32 | Post-spike recovery increment | Type-dependent |

Parameter assignment by neuron type (Izhikevich 2003):

| Type | Fraction | a | b | c | d |
|------|----------|-----|-----|------|---|
| Regular Spiking (RS) | ~60% of exc | 0.02 | 0.2 | -65 | 8 |
| Intrinsically Bursting (IB) | ~15% of exc | 0.02 | 0.2 | -55 | 4 |
| Chattering (CH) | ~5% of exc | 0.02 | 0.2 | -50 | 2 |
| Fast Spiking (FS) | 100% of inh | 0.1 | 0.2 | -65 | 2 |

Excitatory neurons make up `excitatory_ratio` (default 80%) of each block. Within excitatory neurons, the RS/IB/CH split follows the fractions above. The assignment is deterministic given the seed — use `numpy.random.Generator` seeded per block for reproducibility.

### Weight Matrices

| Field | Shape | dtype | Purpose |
|-------|-------|-------|---------|
| `W_intra` | `(B, K, K)` | f32 | Intra-block synaptic weights |
| `W_inter` | `(B, N, K, K)` | f32 | Inter-block weights, per direction |

**N = num_neighbors = 2 * ndim** — this dimension varies with ndim, but the kernel doesn't care because it uses `ti.static(range(num_neighbors))`.

### Memory Budget

For `ndim=3, grid_size=4, K=256`:
- B = 64, N = 6, K = 256
- Per-neuron fields: 12 fields × 64 × 256 × 4 bytes = 786 KB
- W_intra: 64 × 256 × 256 × 4 = 16 MB
- W_inter: 64 × 6 × 256 × 256 × 4 = 96 MB
- **Total: ~113 MB** — fits easily on any modern GPU.

For `ndim=4, grid_size=3, K=128`:
- B = 81, N = 8, K = 128
- W_intra: 81 × 128 × 128 × 4 = 5.3 MB
- W_inter: 81 × 8 × 128 × 128 × 4 = 42 MB
- **Total: ~48 MB**

## Initialization Logic

### init_fields() Details

1. **Membrane potential**: all `v[b, i] = -65.0` (resting potential).
2. **Recovery variable**: `u[b, i] = param_b[b, i] * v[b, i]` (equilibrium for each neuron type).
3. **Neuron type assignment**:
   - Generate a permutation of neuron indices within each block (seeded RNG).
   - First `K * excitatory_ratio` neurons are excitatory, rest inhibitory.
   - Within excitatory: 75% RS, 19% IB, 6% CH (rough cortical proportions).
   - Set `param_a/b/c/d` fields accordingly.
4. **Intra-block weights**: initialize with small random values. Excitatory neurons get positive weights, inhibitory get negative. Use sparse initialization: only ~20-30% of possible connections are non-zero initially. This gives structural plasticity room to grow connections where needed.
5. **Inter-block weights**: sparse-initialize at ~5-10% connection probability with small positive values. Inter-block connections are excitatory only (long-range inhibition is rare in cortex).
6. **Self-connections**: `W_intra[b, i, i] = 0` for all neurons — no autapses.

### Why Initialize on CPU, Then Copy?

Taichi fields can be filled from numpy arrays via `field.from_numpy()`. The initialization logic (RNG, type assignment, sparse init) is complex branching code that's awkward in Taichi kernels. Do it in numpy, then bulk-copy to GPU. This runs once at startup — the copy latency is negligible.

Alternative: write a Taichi kernel for init. Only worth it if init takes >1s, which it won't for these sizes.

### Weight Sign Convention

Excitatory neurons have positive outgoing weights; inhibitory have negative. This is enforced at init and maintained by STDP (which clamps excitatory weights to `[0, w_max]` and inhibitory to `[-w_max, 0]`). The neuron model doesn't distinguish — it just sums `I_syn`, which can be positive or negative.

## build_fields() — The Factory

```python
def build_fields(config: SimConfig, topology: Topology) -> SimFields:
    B = topology.num_blocks
    K = config.torus.neurons_per_block
    N = topology.num_neighbors
    # Allocate all ti.fields with appropriate shapes
    # Return SimFields dataclass
```

This must be called AFTER `ti.init()` — Taichi fields can only be allocated after the backend is selected. The simulator (WC-8) is responsible for calling `ti.init()` before `build_fields()`.

### SimFields as a Dataclass

```python
@dataclass
class SimFields:
    v: ti.ScalarField
    u: ti.ScalarField
    # ... etc
```

Using a dataclass rather than a dict gives us:
- Attribute access with autocomplete
- mypy can check field names (even if the types are `Any` due to Taichi)
- Clear documentation of what fields exist

The Taichi field types don't have good Python stubs, so the type annotations will likely need `# type: ignore[misc]` or a thin type alias. Contain the `Any` here — downstream code that receives `SimFields` gets named attributes instead of stringly-typed dict lookups.

## Tests

### test_fields.py

All tests run on CPU (`conftest.py` fixture handles `ti.init(arch=ti.cpu)`).

**Shape tests:**
- `fields.v.shape == (B, K)` for various configs
- `fields.W_intra.shape == (B, K, K)`
- `fields.W_inter.shape == (B, N, K, K)` where N varies with ndim

**Init value tests:**
- All `v` values are -65.0
- All `u` values are `param_b * v` (check a sample)
- No NaN in any field after init
- `W_intra[b, i, i] == 0` for all neurons (no self-connections)

**Neuron type distribution tests:**
- ~80% excitatory (within ±5% due to integer rounding)
- Excitatory neurons have `param_a == 0.02` (all exc types share this)
- Inhibitory neurons have `param_a == 0.1`

**Weight properties:**
- Excitatory neurons' outgoing weights are all ≥ 0
- Inhibitory neurons' outgoing weights are all ≤ 0
- Inter-block weight sparsity is roughly 5-10% non-zero

**Parametrize over:**
- Small configs: `ndim=2, grid_size=2, K=16` (minimal)
- Medium configs: `ndim=3, grid_size=3, K=32`
- Check that different seeds produce different weight patterns
