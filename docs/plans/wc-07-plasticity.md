# WC-7: Structural Plasticity

**Branch:** `wc-7-plasticity`
**Dependencies:** WC-5 (propagation), WC-6 (STDP)

## Goal

Implement activity-dependent synapse creation and pruning — the slow structural changes that reshape the network's wiring diagram over seconds to minutes. While STDP adjusts the strength of existing connections, structural plasticity creates and destroys connections themselves. This is how the network adapts its topology to the task.

## Files

| File | Action | Notes |
|------|--------|-------|
| `src/hypertoric/kernels/plasticity.py` | Create | Calcium update, intra/inter structural plasticity |
| `tests/test_plasticity.py` | Create | Synapse creation/pruning, calcium dynamics |

## Mechanism: Calcium-Based Homeostasis

### The Biological Inspiration

Real neurons maintain a homeostatic target firing rate. Too little activity → the neuron grows new dendritic spines (incoming synapses). Too much activity → it retracts spines. The proxy for "activity level" is intracellular calcium concentration, which rises with each spike and decays exponentially.

### Calcium Trace

Each neuron has a calcium field `calcium[b, i]` that acts as a low-pass filtered spike count:

```python
# Every timestep
calcium[b, i] *= exp(-dt / calcium_tau)
if spikes[b, i] == 1:
    calcium[b, i] += 1.0
```

With `calcium_tau = 1000ms` (1 second), this smooths over ~1000 timesteps. A neuron firing at 10Hz will have an equilibrium calcium of approximately `10 * calcium_tau / dt = 20,000` (in arbitrary units). The actual target value is tuned relative to this — `calcium_target` in config.

**Practical note:** The actual numerical value of `calcium_target` depends on `dt` and `calcium_tau`. The config default (`calcium_target=0.1`) is a placeholder. It needs to be tuned so that a "healthy" neuron receiving moderate input has calcium near the target. During WC-11 (integration tests), this will be one of the first parameters to adjust.

### Decision Logic

Run periodically (not every timestep — too expensive and unnecessary given the slow timescale).

**Intra-block structural plasticity** — every `intra_interval` timesteps (~1000, i.e., ~1 second):

For each neuron `i` in each block `b`:
1. Compute `calcium[b, i]` — current activity level.
2. Compare to `calcium_target`.

**If calcium < target (underactive neuron):**
- The neuron needs more input → grow a new incoming synapse.
- Find a pre-synaptic neuron `j` in the same block where `W_intra[b, j, i] == 0` (no existing connection).
- If one exists, set `W_intra[b, j, i]` to a small initial weight (e.g., 0.01 for excitatory `j`).
- Choose `j` randomly from available neurons. Don't bias toward nearby indices — within a block, there's no spatial structure.

**If calcium > target (overactive neuron):**
- The neuron has too much input → prune the weakest incoming synapse.
- Find `j` that minimizes `|W_intra[b, j, i]|` among non-zero connections.
- Set `W_intra[b, j, i] = 0.0`.

**Weight threshold pruning (always):**
- For all connections to neuron `i`: if `|W_intra[b, j, i]| < weight_threshold` (e.g., 0.001), set to 0.
- This catches synapses that STDP has weakened nearly to zero — they're not contributing and should free their slot.

### Inter-Block Structural Plasticity

Every `inter_interval` timesteps (~10,000, i.e., ~10 seconds). Same logic but operates on `W_inter[b, d, j, i]` for each direction `d`.

Inter-block is slower because:
1. It's more expensive (N times more weight matrices to scan).
2. Inter-block connectivity should be more stable — long-range rewiring is a coarser, rarer adjustment.
3. The rotating STDP schedule means inter-block weights change more slowly than intra-block weights, so the plasticity interval should match.

### Synapse Budget

Without limits, structural plasticity could densify the weight matrix entirely (every possible connection filled). This is undesirable — dense connectivity removes the network's ability to specialize.

**Approach:** Don't enforce a hard budget. Instead, rely on the homeostatic balance:
- Growing a synapse increases input → raises calcium → triggers pruning of the weakest → net connection count stays roughly stable.
- The `weight_threshold` prunes STDP-weakened connections continuously.

If testing shows runaway densification, add a soft cap: skip synapse creation if the target neuron already has more than `max_incoming` non-zero connections (e.g., `0.5 * K`). Start without this cap and add it if needed.

## Implementation: Taichi Kernels

### make_calcium_update(B, K)

Simple — runs every timestep, same structure as trace_update.

```python
@ti.kernel
def calcium_update(
    calcium: ti.template(), spikes: ti.template(),
    calcium_tau: ti.f32, dt: ti.f32,
) -> None:
    for block_idx, i in ti.ndrange(B, K):
        calcium[block_idx, i] *= ti.exp(-dt / calcium_tau)
        if spikes[block_idx, i] == 1:
            calcium[block_idx, i] += 1.0
```

### make_structural_intra(B, K)

More complex — involves conditional logic and random selection.

**Random selection on GPU:** Taichi doesn't have built-in random number generators suitable for per-neuron decisions. Options:
1. **`ti.random()`** — available in Taichi, returns uniform [0, 1). Sufficient for "pick a random available slot."
2. **Pre-generate random indices on CPU, upload as a field.** More deterministic but adds CPU↔GPU transfer. Use only if reproducibility is critical for a specific test.

**Finding an available slot:** Iterating over all `K` potential pre-synaptic neurons to find one with `W == 0` is O(K) per neuron. For `K = 256`, this is fine. For larger K, consider maintaining a "free slot" list — but K > 512 is unlikely given memory constraints.

**Pruning the weakest:** Requires finding the argmin of |W| among non-zero entries. Again O(K) — fine for K ≤ 512.

```python
@ti.kernel
def structural_intra(
    W_intra: ti.template(), calcium: ti.template(),
    calcium_target: ti.f32, weight_threshold: ti.f32,
    init_weight: ti.f32,
) -> None:
    for block_idx, i in ti.ndrange(B, K):
        # Threshold pruning first
        for j in range(K):
            if W_intra[block_idx, j, i] != 0.0:
                if ti.abs(W_intra[block_idx, j, i]) < weight_threshold:
                    W_intra[block_idx, j, i] = 0.0

        # Homeostatic growth/pruning
        if calcium[block_idx, i] < calcium_target:
            # Grow: find a random empty incoming slot
            # Iterate from a random offset to avoid bias
            start = ti.cast(ti.random() * K, ti.i32) % K
            for offset in range(K):
                j = (start + offset) % K
                if j != i and W_intra[block_idx, j, i] == 0.0:
                    W_intra[block_idx, j, i] = init_weight
                    break  # grow one synapse per interval

        elif calcium[block_idx, i] > calcium_target:
            # Prune: find weakest incoming synapse
            min_w = w_max + 1.0
            min_j = -1
            for j in range(K):
                w = ti.abs(W_intra[block_idx, j, i])
                if w > 0.0 and w < min_w:
                    min_w = w
                    min_j = j
            if min_j >= 0:
                W_intra[block_idx, min_j, i] = 0.0
```

### make_structural_inter(B, K, num_neighbors)

Same logic but for `W_inter[b, d, j, i]`. Iterate over all directions. Since this runs every ~10s, the cost of scanning all `N` direction matrices is amortized.

## Interaction with Other Components

### STDP ↔ Structural Plasticity

- STDP weakens useless synapses toward zero → structural plasticity prunes them.
- Structural plasticity creates new synapses at small initial weight → STDP strengthens useful ones.
- This creates a "marketplace" for connections: new connections are cheap to create, but only those that carry causal information survive STDP's net-depression bias.

### Timing

The simulator (WC-8) calls structural plasticity at the configured intervals:

```python
if self._step_count % config.plasticity.intra_interval == 0:
    self.kernels.structural_intra(...)
if self._step_count % config.plasticity.inter_interval == 0:
    self.kernels.structural_inter(...)
```

These intervals are in timesteps, not wall-clock time. At `dt=0.5ms`:
- `intra_interval=1000` → every 500ms of simulated time
- `inter_interval=10000` → every 5 seconds of simulated time

## Tests

Use small configs: `B=2, K=8, ndim=2, grid_size=2`.

### Calcium Dynamics

- Spike a neuron 10 times over 100 timesteps. Verify calcium is positive and roughly matches expected filtered value.
- Stop spiking. Verify calcium decays to near-zero over `5 * calcium_tau` timesteps.

### Synapse Growth (Low Activity)

- Set `calcium[0, 0] = 0` (far below target). Run `structural_intra`.
- Count non-zero incoming weights for neuron 0 before and after. Should increase by exactly 1.
- The new synapse should have weight approximately `init_weight`.

### Synapse Pruning (High Activity)

- Set `calcium[0, 0] = 10 * calcium_target` (far above target). Set up several incoming synapses with varying weights.
- Run `structural_intra`. Count non-zero incoming weights. Should decrease by 1.
- The REMOVED synapse should be the one with the smallest |weight|.

### Threshold Pruning

- Set `W_intra[0, 1, 0] = 0.0005` (below `weight_threshold=0.001`).
- Run `structural_intra`. Verify `W_intra[0, 1, 0] == 0.0`.

### No Self-Connections

- Run structural plasticity with an underactive neuron. Verify no self-connection is created: `W_intra[b, i, i]` remains 0 for all neurons.

### Synapse Count Stability (Smoke Test)

- Initialize a small network with random calcium levels. Run structural plasticity 100 times.
- Verify total synapse count stays bounded (doesn't grow to K² or shrink to 0).
- This doesn't test convergence to a specific count — just that the mechanism is self-limiting.

### Inter-Block Plasticity

- Same tests as intra-block but for `W_inter`. Verify new inter-block synapses have positive initial weight (inter-block connections are excitatory only).
