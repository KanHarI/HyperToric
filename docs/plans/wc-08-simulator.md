# WC-8: Simulator Orchestration

**Branch:** `wc-8-simulator`
**Dependencies:** WC-5 (propagation), WC-6 (STDP)

## Goal

Build the `Simulator` class that owns all state (fields, topology, compiled kernels) and exposes a simple `step()` method. This is the central coordinator — it sequences kernel calls in the correct order, manages the rotating STDP schedule, triggers structural plasticity at the right intervals, and provides the read/write interface for the I/O layer.

## Files

| File | Action | Notes |
|------|--------|-------|
| `src/hypertoric/simulator.py` | Create | Simulator class |
| `src/hypertoric/kernels/__init__.py` | Update | KernelSet dataclass, build_kernels() |
| `tests/test_simulator.py` | Create | Construction, stepping, I/O interface |

## Simulator Class Design

### Construction

```python
class Simulator:
    def __init__(self, config: SimConfig) -> None:
        # 1. Initialize Taichi
        arch = self._resolve_arch(config.backend)
        ti.init(arch=arch, random_seed=config.seed)

        # 2. Build topology
        self.topology = Topology(config.torus.ndim, config.torus.grid_size)

        # 3. Allocate and initialize fields
        self.fields = build_fields(config, self.topology)
        init_fields(self.fields, config, self.topology)

        # 4. Compile kernels
        self.kernels = build_kernels(config, self.topology)

        # 5. State
        self._config = config
        self._step_count: int = 0
```

**Backend resolution:** `config.backend` is a string like `"gpu"`, `"cpu"`, `"cuda"`, `"metal"`. Map to Taichi arch:
- `"gpu"` → `ti.gpu` (auto-detect best available)
- `"cpu"` → `ti.cpu`
- `"cuda"` → `ti.cuda`
- `"metal"` → `ti.metal`
- `"vulkan"` → `ti.vulkan`

This mapping is important because tests always use `ti.cpu` (via conftest fixture), but production runs should auto-detect. The `"gpu"` default falls back gracefully: CUDA → Metal → Vulkan → CPU.

### Step Ordering

The `step()` method executes one simulation timestep (~0.5ms of simulated time). The order is critical — reordering breaks the simulation semantics.

```python
def step(self) -> None:
    dt = self._config.neuron.dt
    cfg = self._config

    # 1. Neuron update: v, u → new v, u, spikes
    self.kernels.neuron_update(
        self.fields.v, self.fields.u, self.fields.spikes,
        self.fields.I_syn, self.fields.I_ext,
        self.fields.param_a, self.fields.param_b,
        self.fields.param_c, self.fields.param_d,
        dt,
    )

    # 2. Spike propagation: spikes + weights → I_syn
    self.kernels.spike_propagate(
        self.fields.I_syn, self.fields.spikes,
        self.fields.W_intra, self.fields.W_inter,
        tau_syn, dt,
    )

    # 3. Trace update: decay + spike increment
    self.kernels.trace_update(
        self.fields.trace_pre, self.fields.trace_post,
        self.fields.spikes,
        cfg.stdp.tau_pre, cfg.stdp.tau_post, dt,
    )

    # 4. Intra-block STDP (every timestep)
    self.kernels.stdp_intra(
        self.fields.W_intra, self.fields.spikes,
        self.fields.trace_pre, self.fields.trace_post,
        cfg.stdp.a_plus, cfg.stdp.a_minus, cfg.plasticity.w_max,
    )

    # 5. Inter-block STDP (rotating or all)
    if cfg.stdp.inter_mode == "rotating":
        direction = self._step_count % self.topology.num_neighbors
        self.kernels.stdp_inter(
            self.fields.W_inter, self.fields.spikes,
            self.fields.trace_pre, self.fields.trace_post,
            direction,
            cfg.stdp.a_plus, cfg.stdp.a_minus, cfg.plasticity.w_max,
        )
    else:  # "all"
        for d in range(self.topology.num_neighbors):
            self.kernels.stdp_inter(
                self.fields.W_inter, self.fields.spikes,
                self.fields.trace_pre, self.fields.trace_post,
                d,
                cfg.stdp.a_plus, cfg.stdp.a_minus, cfg.plasticity.w_max,
            )

    # 6. Calcium update (every timestep)
    self.kernels.calcium_update(
        self.fields.calcium, self.fields.spikes,
        cfg.plasticity.calcium_tau, dt,
    )

    # 7. Structural plasticity (periodic)
    if self._step_count % cfg.plasticity.intra_interval == 0:
        self.kernels.structural_intra(
            self.fields.W_intra, self.fields.calcium,
            cfg.plasticity.calcium_target, cfg.plasticity.weight_threshold,
            0.01,  # init_weight
        )
    if self._step_count % cfg.plasticity.inter_interval == 0:
        self.kernels.structural_inter(
            self.fields.W_inter, self.fields.calcium,
            cfg.plasticity.calcium_target, cfg.plasticity.weight_threshold,
            0.01,
        )

    self._step_count += 1
```

### Why This Order

1. **Neuron update first** — reads I_syn from PREVIOUS timestep, produces spikes for THIS timestep.
2. **Propagation second** — reads THIS timestep's spikes, updates I_syn for NEXT timestep. Note: I_syn is both read (decay) and written (add new current) in this kernel.
3. **Trace update third** — must happen after spikes are computed but before STDP reads traces. Traces reflect the CURRENT spike state.
4. **STDP after traces** — reads current traces and current spikes. Updates weights in-place.
5. **Calcium after spikes** — accumulates spike into calcium trace.
6. **Structural plasticity last** — reads calcium (updated this step) and weights (updated by STDP this step). Runs infrequently.

Alternative: calcium update could happen before STDP — the calcium trace isn't used by STDP, only by structural plasticity. But keeping it after STDP and before structural plasticity ensures the most up-to-date calcium when structural decisions are made.

## KernelSet Dataclass

Holds all compiled kernel references. Built once at simulator construction.

```python
@dataclass
class KernelSet:
    neuron_update: NeuronUpdateFn
    spike_propagate: Callable[..., None]
    trace_update: Callable[..., None]
    stdp_intra: Callable[..., None]
    stdp_inter: Callable[..., None]
    calcium_update: Callable[..., None]
    structural_intra: Callable[..., None]
    structural_inter: Callable[..., None]
```

### build_kernels()

```python
def build_kernels(config: SimConfig, topology: Topology) -> KernelSet:
    B = topology.num_blocks
    K = config.torus.neurons_per_block
    N = topology.num_neighbors

    neuron_factory = get_neuron_factory(config.neuron.model)

    return KernelSet(
        neuron_update=neuron_factory(B, K),
        spike_propagate=make_spike_propagate(B, K, N, config.torus.grid_size, config.torus.ndim),
        trace_update=make_trace_update(B, K),
        stdp_intra=make_stdp_intra(B, K),
        stdp_inter=make_stdp_inter(B, K, N, config.torus.grid_size, config.torus.ndim),
        calcium_update=make_calcium_update(B, K),
        structural_intra=make_structural_intra(B, K),
        structural_inter=make_structural_inter(B, K, N, config.torus.grid_size, config.torus.ndim),
    )
```

Each factory bakes `B`, `K`, `N` as compile-time constants. The returned kernels are normal callables — no trace of the factory remains at runtime.

## External Interface

### inject_current()

```python
def inject_current(self, block_indices: Sequence[int], currents: NDArray[np.float32]) -> None:
    """Set I_ext for specified blocks. currents shape: (len(block_indices), K)."""
    for idx, block_idx in enumerate(block_indices):
        # Write currents[idx, :] into I_ext[block_idx, :]
        ...
```

This is a CPU→GPU transfer. Called once per game tick (every ~50-100 timesteps), NOT every simulation step. The I/O manager (WC-9) uses this to inject sensory input and feedback.

**Implementation options:**
1. `ti.field.from_numpy()` — bulk copy. Requires extracting a slice of I_ext, which Taichi fields don't support natively. May need a staging numpy array.
2. A small Taichi kernel that copies from a staging field. More overhead but avoids the slicing issue.
3. Direct `field[block_idx, i] = value` in a Python loop — O(K) per block, fine for K≤512 since this runs ~10x/second.

Option 3 is simplest and sufficient for initial implementation. Optimize to option 1 if profiling shows I/O as a bottleneck.

### read_spikes()

```python
def read_spikes(self, block_indices: Sequence[int]) -> NDArray[np.int32]:
    """Read spike state for specified blocks. Returns shape (len(block_indices), K)."""
    ...
```

GPU→CPU transfer. Also called per game tick. Same implementation options as `inject_current`.

**Critical performance note:** GPU→CPU transfers are expensive (~100µs each due to synchronization). Do NOT call these per simulation step — batch them per game tick. The I/O manager handles this batching.

### read_weights() (optional, for visualization)

```python
def read_weights(self, block_idx: int) -> tuple[NDArray, NDArray]:
    """Return (W_intra, W_inter) for a block. For dashboards only."""
```

Only used by visualization (WC-F). Not performance-critical.

## Tests

### Construction Tests

- `Simulator(SimConfig())` completes without error.
- Parametrize over several configs: 2D/3D/4D, various grid sizes, different neuron models.
- Simulator with `backend="cpu"` always works (even in CI).

### Stepping Tests

- Run `sim.step()` for 100 steps. No errors, no NaN in fields.
- Run 1000 steps with small constant current injection. Verify spikes occur.
- Run 1000 steps with zero current. Verify the network eventually goes silent (spontaneous activity dies out without input because A_minus > A_plus weakens random connections).

### inject_current / read_spikes

- Inject current into block 0, run 50 steps, read spikes from block 0. Should see non-zero spikes.
- Read spikes from a block with no current injection — should see zero (or very few) spikes.
- Verify shapes: `read_spikes([0, 1])` returns shape `(2, K)`.

### Step Count

- Verify `sim._step_count` increments by 1 per `step()` call.
- Verify structural plasticity triggers at the correct intervals (mock or count kernel invocations).

### Rotating STDP Schedule

- With `inter_mode="rotating"` and `ndim=2` (4 directions): verify the direction argument cycles 0, 1, 2, 3, 0, 1, ... across steps.
