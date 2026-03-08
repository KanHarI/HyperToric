# WC-2: Config + Topology

**Branch:** `wc-2-config-topology`
**Dependencies:** WC-1

## Goal

Define the entire configuration surface as typed Python dataclasses and implement the N-dimensional torus topology. After this chapter, every subsequent module can `from hypertoric.config import SimConfig` and have typed access to all parameters. The topology module provides the coordinate math that fields, kernels, and I/O all depend on.

## Files

| File | Action | Notes |
|------|--------|-------|
| `src/hypertoric/config.py` | Create | Hydra structured configs |
| `src/hypertoric/topology.py` | Create | N-dim torus coordinate math |
| `conf/config.yaml` | Create | Hydra defaults list |
| `conf/torus/*.yaml` | Create | Torus presets |
| `conf/neuron/*.yaml` | Create | Neuron presets |
| `conf/stdp/*.yaml` | Create | STDP mode presets |
| `conf/plasticity/*.yaml` | Create | Plasticity presets |
| `conf/io/*.yaml` | Create | I/O layout presets |
| `conf/training/*.yaml` | Create | Task presets |
| `tests/test_config.py` | Create | Config loading tests |
| `tests/test_topology.py` | Create | Topology correctness tests |

## config.py — Design Decisions

### Why Hydra Structured Configs

Plain dataclasses would work for config, but Hydra gives us:
- **CLI overrides for free**: `python -m hypertoric torus.grid_size=8 stdp.inter_mode=all` — no argparse boilerplate.
- **Config composition**: swap entire subsections via `torus=small_2d` or `stdp=all`.
- **Validation at load time**: OmegaConf enforces types from the dataclass annotations.
- **Reproducibility**: Hydra saves the resolved config to `outputs/` on every run.

The cost is the Hydra/OmegaConf dependency and the ConfigStore registration boilerplate. Worth it for a project with this many knobs.

### Dataclass Hierarchy

```
SimConfig
├── torus: TorusConfig        (ndim, grid_size, neurons_per_block)
├── neuron: NeuronConfig      (model, excitatory_ratio, dt)
├── stdp: STDPConfig          (a_plus, a_minus, tau_pre, tau_post, inter_mode)
├── plasticity: PlasticityConfig  (intervals, calcium params, w_max)
├── io: IOConfig              (sensory/motor axis and position, cluster sizes)
├── training: TrainingConfig  (task name, feedback_tau)
├── seed: int
└── backend: str
```

Each sub-config is its own dataclass so Hydra can compose them independently.

### Key Parameter Constraints

These should be validated in a `__post_init__` or a standalone `validate_config()`:

- `ndim` ∈ {2, 3, 4} — below 2 is degenerate, above 4 is untested and memory explodes.
- `grid_size` ≥ 2 — grid_size=1 means the single block is its own neighbor in every direction, which breaks STDP (self-loops).
- `neurons_per_block` must be ≥ 2 (need at least one excitatory, one inhibitory).
- `excitatory_ratio` ∈ (0, 1) — 0 or 1 gives a network with no inhibition or no excitation.
- `a_minus > a_plus` — ensures net depression under random spiking (critical for STDP stability). Warn but don't error if violated.
- `sensory_position != motor_position` — same plane for I/O defeats the purpose (see architecture doc).
- `sensory_axis < ndim` — can't slice along a non-existent axis.

### Hydra ConfigStore Registration

```python
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=SimConfig)
cs.store(group="torus", name="default_3d", node=TorusConfig())
cs.store(group="torus", name="small_2d", node=TorusConfig(ndim=2, grid_size=4, neurons_per_block=64))
# ... etc for each group
```

This runs at import time. The `conf/*.yaml` files override or extend these programmatic defaults.

### NeuronConfig.model Field

```python
@dataclass
class NeuronConfig:
    model: str = "izhikevich"  # key into NEURON_MODELS registry (WC-4)
    excitatory_ratio: float = 0.8
    dt: float = 0.5  # ms
```

The `model` field is a string key, not an enum, so new models can be registered without modifying config.py. The registry lives in `kernels/neuron_models/__init__.py` (WC-4) and validates the key at simulator construction time.

## topology.py — The N-Dimensional Torus

### Core Abstraction

The topology module converts between N-dimensional grid coordinates and flat 1D indices. Every module above it (fields, kernels, I/O) works exclusively with flat indices. This keeps the ndim-specific math in one place.

### Flat Indexing Scheme

For a grid of size `G` in `D` dimensions, flat index of coordinate `(c_0, c_1, ..., c_{D-1})`:

```
flat = c_0 * G^0 + c_1 * G^1 + ... + c_{D-1} * G^{D-1}
     = Σ c_i * stride_i    where stride_i = G^i
```

This is little-endian (axis 0 is the least significant). The choice is arbitrary but must be consistent everywhere.

Inverse: `c_i = (flat // stride_i) % G`

### Direction Encoding

`2 * ndim` directions, indexed 0 to `2*ndim - 1`:
- Direction `d` → axis `d // 2`, offset `+1` if `d % 2 == 0`, `-1` if `d % 2 == 1`

For 3D: directions 0-5 map to `+x, -x, +y, -y, +z, -z`.

This encoding matters because:
1. **Opposite directions are adjacent**: direction `d` and `d ^ 1` (XOR with 1) are opposites. Useful for neighbor symmetry checks.
2. **Rotating STDP** iterates `t % num_neighbors`, cycling through all directions uniformly.
3. **The kernel factory** uses `ti.static(range(num_neighbors))` to unroll the direction loop — the encoding must be the same in Python (topology) and Taichi (kernels).

### Neighbor Computation

```python
def get_neighbor_flat(self, flat_idx: int, direction: int) -> int:
    axis = direction // 2
    offset = 1 - 2 * (direction % 2)  # +1 or -1
    coord = list(self.flat_to_coord(flat_idx))
    coord[axis] = (coord[axis] + offset) % self.grid_size
    return self.coord_to_flat(tuple(coord))
```

Periodic wrapping is just `% grid_size`. No boundary conditionals — the torus has no boundaries.

### Precomputed Neighbor Table

The `Topology` class should precompute a neighbor table at construction time:

```python
self._neighbors: list[list[int]]  # _neighbors[flat_idx][direction] = neighbor_flat_idx
```

This is used in Python-side code (I/O manager, tests, visualization). The Taichi kernels do NOT use this table — they recompute neighbors inline via modular arithmetic on the flat index, which is cheaper than a global memory read. See `docs/experiments/perf/neighbor-lookup-vs-inline.md` for the rationale.

### Properties

- `num_blocks = grid_size ** ndim`
- `num_neighbors = 2 * ndim`
- `strides = tuple(grid_size ** a for a in range(ndim))` — useful for flat↔coord conversion

### Plane Selection (for I/O, WC-9)

Topology should provide a method to select all blocks in a hyperplane:

```python
def get_plane(self, axis: int, position: int) -> list[int]:
    """Return flat indices of all blocks where coord[axis] == position."""
```

This is used by the I/O manager to find sensory and motor blocks. For a 4×4×4 3D torus with `axis=0, position=0`, this returns the 16 blocks at x=0.

Implementation: iterate all `num_blocks` flat indices, convert to coord, filter by `coord[axis] == position`. Alternatively, generate all `grid_size^(ndim-1)` coordinates for the other axes and set the target axis to `position`. The latter is cleaner but both are fine — this runs once at init, not per timestep.

## Hydra Config Directory

```
conf/
├── config.yaml             # defaults list
├── torus/
│   ├── small_2d.yaml       # ndim: 2, grid_size: 4, neurons_per_block: 64
│   ├── default_3d.yaml     # ndim: 3, grid_size: 4, neurons_per_block: 256
│   └── large_4d.yaml       # ndim: 4, grid_size: 3, neurons_per_block: 128
├── neuron/
│   └── default.yaml        # model: izhikevich, excitatory_ratio: 0.8, dt: 0.5
├── stdp/
│   ├── rotating.yaml       # inter_mode: rotating
│   └── all.yaml            # inter_mode: all
├── plasticity/
│   └── default.yaml
├── io/
│   └── tracking_1d.yaml
└── training/
    ├── static_target.yaml
    └── slow_step.yaml
```

**`config.yaml`** is the top-level defaults list:
```yaml
defaults:
  - torus: default_3d
  - neuron: default
  - stdp: rotating
  - plasticity: default
  - io: tracking_1d
  - training: static_target

seed: 42
backend: gpu
```

CLI override example: `uv run python -m hypertoric torus=small_2d stdp=all torus.grid_size=8`

### Why YAML files AND structured configs?

The structured configs (Python dataclasses) define the schema and defaults. The YAML files provide named presets. A user can:
1. Use a preset: `torus=small_2d`
2. Override a field: `torus.grid_size=8`
3. Combine both: `torus=small_2d torus.neurons_per_block=128`

Without YAML presets, every configuration requires listing all fields on the CLI. Without structured configs, YAML typos are silent.

## Tests

### test_topology.py

Parametrize over `ndim ∈ {2, 3, 4}` and `grid_size ∈ {2, 3, 4, 5}` (12 combinations). Each test should be fast (pure Python, no Taichi).

**Correctness tests:**
- `num_blocks == grid_size ** ndim` for all configs
- `coord_to_flat(flat_to_coord(i)) == i` for all `i` in `range(num_blocks)` — round-trip
- `flat_to_coord(coord_to_flat(c)) == c` for all valid coordinates — reverse round-trip
- All flat indices in `range(num_blocks)` are distinct after `coord_to_flat`

**Topology tests:**
- Every block has exactly `2 * ndim` neighbors
- Neighbor symmetry: if `get_neighbor_flat(a, d) == b`, then `get_neighbor_flat(b, d ^ 1) == a` (opposite direction gets you back)
- No self-neighbors: `get_neighbor_flat(i, d) != i` for all `i, d` (requires `grid_size >= 2`)
- Periodic wrapping: for a corner block (all coords = 0), neighbor in the `-` direction wraps to `grid_size - 1`

**Plane selection tests:**
- `get_plane(axis, pos)` returns `grid_size^(ndim-1)` blocks for any valid axis/pos
- All returned blocks have `coord[axis] == pos`
- Union of `get_plane(axis, pos)` for all `pos` equals all blocks (partitions the grid)

**Edge cases:**
- `grid_size = 2`: every block is its own neighbor's neighbor (diameter = 1). This is valid and should work.
- `ndim = 2, grid_size = 2`: only 4 blocks, each with 4 neighbors, high overlap. Minimal viable torus.

### test_config.py

- Default `SimConfig()` instantiates without error and has expected default values
- Hydra `compose` API loads each config group (requires `initialize_config_dir` in test)
- Validation catches: `ndim=0`, `ndim=5`, `grid_size=0`, `grid_size=-1`, `excitatory_ratio=1.5`
- Overrides work: loading `torus=small_2d` gives `ndim=2, grid_size=4`
- `SimConfig` can be serialized to/from OmegaConf DictConfig (round-trip)
