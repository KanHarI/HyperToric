# WC-5: Spike Propagation

**Branch:** `wc-5-propagation`
**Dependencies:** WC-3 (fields), WC-4 (neuron kernel)

## Goal

Implement the kernel that converts spikes into synaptic currents. When a neuron fires, its spike travels through intra-block and inter-block weight matrices to produce current in post-synaptic neurons. This is the most compute-intensive kernel in the simulation — it touches every weight in the network every timestep.

## Files

| File | Action | Notes |
|------|--------|-------|
| `src/hypertoric/kernels/propagate.py` | Create | make_spike_propagate factory |
| `tests/test_propagate.py` | Create | Connectivity, wrapping, decay tests |

## The Computation

For each neuron `(block_idx, i)`, accumulate incoming synaptic current:

```
acc = 0
# Intra-block: spikes from all neurons in same block
for j in range(K):
    acc += W_intra[block_idx, j, i] * spikes[block_idx, j]

# Inter-block: spikes from all neurons in all neighbor blocks
for d in range(num_neighbors):
    nb = neighbor(block_idx, d)
    for j in range(K):
        acc += W_inter[block_idx, d, j, i] * spikes[nb, j]

# Exponential decay + new input
I_syn[block_idx, i] = I_syn[block_idx, i] * exp(-dt / tau_syn) + acc
```

### Why Exponential Decay, Not Instantaneous

Biological synaptic currents don't appear and disappear in a single timestep. They rise on spike arrival and decay over a few milliseconds (tau_syn ≈ 2-5ms for excitatory, 5-10ms for inhibitory). The exponential filter `I_syn *= exp(-dt/tau_syn)` models this. Without it, a spike produces current for exactly one timestep — unrealistically brief and numerically unstable (all energy in one step).

The `exp(-dt/tau_syn)` term is computed once per kernel call and passed as a parameter, not recomputed per neuron. At `dt=0.5ms, tau_syn=5ms`, the decay factor is `exp(-0.1) ≈ 0.905`.

### Weight Matrix Indexing Convention

`W_intra[block_idx, j, i]` — weight FROM neuron j TO neuron i within block. Row j is the pre-synaptic neuron, column i is the post-synaptic neuron. This means "outgoing weights from j" is `W_intra[b, j, :]` and "incoming weights to i" is `W_intra[b, :, i]`.

The loop structure reflects this: for each post-synaptic neuron i, iterate over all pre-synaptic neurons j and check if j spiked.

`W_inter[block_idx, d, j, i]` — weight from neuron j in neighbor block (direction d) to neuron i in this block. Note: `block_idx` here is the POST-synaptic block. The weight lives with the post-synaptic block, not the pre-synaptic one. This matters for STDP (WC-6), where weight updates need to be local to one block's memory.

## Inline Neighbor Computation

The kernel computes neighbors via modular arithmetic on the flat index, not from a precomputed table.

### The Math

For flat index `block_idx` in a grid of size `G` with `D` dimensions, strides are `[1, G, G², G³, ...]`:

```python
stride = G ** axis
coord_along_axis = (block_idx // stride) % G
new_coord = (coord_along_axis + offset) % G   # offset is +1 or -1
neighbor = block_idx + (new_coord - coord_along_axis) * stride
```

This is 5 integer operations: one division, two modulos, one subtraction, one multiply-add. On GPU, integer arithmetic is essentially free compared to the memory access latency of reading from a neighbor lookup table (which would be a random global memory read).

### Implementation as ti.func

```python
@ti.func
def _neighbor(block_idx: ti.i32, d: ti.i32) -> ti.i32:
    axis = d // 2
    sign = 1 - 2 * (d % 2)  # +1 if d even, -1 if d odd
    stride = strides[axis]   # captured from closure, compile-time constant
    coord = (block_idx // stride) % grid_size
    new_coord = (coord + sign) % grid_size
    return block_idx + (new_coord - coord) * stride
```

`strides` is a tuple captured from the factory closure. `ti.static` makes the direction loop unrolling work because `num_neighbors` is a compile-time constant.

### Why Not a Lookup Table?

See `docs/experiments/perf/neighbor-lookup-vs-inline.md`. Summary:
- A lookup table of shape `(B, 2*ndim)` is small but requires a global memory read per direction per neuron.
- The arithmetic approach uses only registers (no memory access).
- For small ndim (2-4), the arithmetic is 5-10 integer ops per direction — well within the compute budget of a memory-bound kernel.
- The lookup table wins only if `ndim` is very large (>10) where the unrolled loop becomes unwieldy. Not relevant here.

## Performance Characteristics

This kernel dominates runtime. For `B=64, K=256, N=6`:
- Each neuron reads `K` intra-block spikes + `N*K` inter-block spikes = `256 + 1536 = 1792` spike values.
- Each neuron reads `K` intra-block weights + `N*K` inter-block weights = `1792` weight values.
- Total reads per neuron: 3584 × 4 bytes = ~14 KB.
- Total neurons: 64 × 256 = 16,384.
- Total memory traffic: ~230 MB per timestep.

This is well within GPU memory bandwidth (modern GPUs: 500-2000 GB/s), so the kernel should complete in <1ms. The bottleneck is memory latency, not compute.

### Optimization Opportunities (not for initial implementation)

1. **Sparse weights**: if inter-block connections are <10% non-zero, sparse storage (CSR) could reduce memory traffic 10x. But sparse formats hurt GPU occupancy. Only pursue if profiling shows W_inter reads dominate.
2. **Kernel fusion**: fuse `neuron_update` + `spike_propagate` into a single kernel to avoid an extra global memory round-trip for `spikes` and `I_syn`. Only if kernel launch overhead is significant.
3. **Tiling**: load W_intra into shared memory per block, since all K neurons in a block read the same weight matrix. Taichi supports `ti.block_local` for this.

## Tests

Use a small torus: `ndim=2, grid_size=2, K=4` (4 blocks, 4 neighbors each, 4 neurons per block).

### Intra-Block Propagation

- Set `W_intra[0, 0, 1] = 0.5`. Make neuron 0 spike. Assert `I_syn[0, 1]` increases by 0.5.
- Set `W_intra[0, 0, 1] = -0.3` (inhibitory). Make neuron 0 spike. Assert `I_syn[0, 1]` decreases by 0.3.
- No spike → `I_syn` decays but doesn't gain new current.

### Inter-Block Propagation

- Find the neighbor of block 0 in direction 0 (call it block B).
- Set `W_inter[0, 0, 0, 1] = 0.7`. Make neuron 0 in block B spike.
- Assert `I_syn[0, 1]` increases by 0.7.

### Periodic Boundary Wrapping

- In a 2×2 2D torus, block at (0,0) has neighbor at (1,0) in the +x direction. The +x neighbor of (1,0) is (0,0) — wraps around.
- Set up a weight from the "wrapped" neighbor back to block (0,0). Spike the source neuron. Verify current arrives.

### Exponential Decay

- Set `I_syn[0, 0] = 10.0`. Run propagate with no spikes for several steps.
- After each step, `I_syn[0, 0]` should be `prev * exp(-dt/tau_syn)`.
- After 20 steps (10ms at dt=0.5), `I_syn ≈ 10 * exp(-10/5) ≈ 1.35` for tau_syn=5.

### Superposition

- Two neurons spike simultaneously into the same target. Target's `I_syn` should be the sum of both weights (plus decayed previous value).

### Zero Weights

- All weights zero. Spikes occur but `I_syn` only decays, never increases. Verifies no accidental current injection.
