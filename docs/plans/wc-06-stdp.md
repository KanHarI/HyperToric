# WC-6: STDP Kernels

**Branch:** `wc-6-stdp`
**Dependencies:** WC-3 (fields), WC-4 (neuron kernel)

## Goal

Implement Spike-Timing-Dependent Plasticity — the unsupervised learning rule that strengthens causal connections and weakens non-causal ones. This is how the network learns without backpropagation. STDP is trace-based (no spike history search), operates as rank-1 weight updates, and splits into intra-block (every timestep) and inter-block (rotating or all directions).

## Files

| File | Action | Notes |
|------|--------|-------|
| `src/hypertoric/kernels/stdp.py` | Create | Trace update, intra/inter STDP factories |
| `tests/test_stdp.py` | Create | LTP/LTD, trace decay, rotating mode, clamping |

## Background: How Trace-Based STDP Works

### The Biological Rule

If neuron A fires shortly before neuron B (A→B is causal), strengthen the A→B connection (Long-Term Potentiation, LTP). If A fires shortly after B (A→B is anti-causal), weaken it (Long-Term Depression, LTD).

The timing window is ~20ms: spikes separated by more than ~40ms produce negligible plasticity.

### Traces Instead of Spike History

A naive implementation would search each neuron's spike history to find pairs within the timing window — O(spikes²) and cache-hostile. Traces avoid this entirely.

Each neuron maintains two exponentially decaying traces:
- `trace_pre[b, i]`: incremented by 1 on each spike, decays with time constant `tau_pre`
- `trace_post[b, i]`: incremented by 1 on each spike, decays with time constant `tau_post`

The trace value at any moment represents the "recency" of that neuron's last spike(s). A high trace means the neuron fired recently; a near-zero trace means it hasn't fired in a while.

### Weight Update Rule

When neuron `j` spikes (pre-synaptic):
```
W[j, i] -= A_minus * trace_post[i]   for all post-synaptic neurons i
```
"If post fired recently (high trace_post) and pre fires now → anti-causal → depress."

When neuron `i` spikes (post-synaptic):
```
W[j, i] += A_plus * trace_pre[j]     for all pre-synaptic neurons j
```
"If pre fired recently (high trace_pre) and post fires now → causal → potentiate."

### Why A_minus > A_plus

Under random (uncorrelated) spiking, pre-before-post and post-before-pre are equally likely. If `A_plus == A_minus`, weights would random-walk. By making `A_minus > A_plus` (e.g., 0.012 vs 0.01), there's a net depression bias. Only genuinely correlated causal pathways — where pre fires before post more often than chance — survive. This is the competition mechanism: useful synapses strengthen, useless ones weaken and get pruned by structural plasticity.

### Rank-1 Updates

For dense weight matrices, the STDP update is:

```
ΔW = A_plus * spike_vector × trace_pre_vector.T   (LTP, outer product)
   - A_minus * trace_post_vector × spike_vector.T   (LTD, outer product)
```

Each is a rank-1 matrix update — an outer product of two K-length vectors, producing a K×K update. Extremely GPU-friendly: one GEMM-like operation per block.

## Three Kernels

### 1. make_trace_update(B, K)

Runs every timestep. Two operations:
1. Decay both traces: `trace *= exp(-dt / tau)`
2. On spike: increment trace by 1.0

```python
@ti.kernel
def trace_update(
    trace_pre: ti.template(), trace_post: ti.template(),
    spikes: ti.template(),
    tau_pre: ti.f32, tau_post: ti.f32, dt: ti.f32,
) -> None:
    for block_idx, i in ti.ndrange(B, K):
        trace_pre[block_idx, i] *= ti.exp(-dt / tau_pre)
        trace_post[block_idx, i] *= ti.exp(-dt / tau_post)
        if spikes[block_idx, i] == 1:
            trace_pre[block_idx, i] += 1.0
            trace_post[block_idx, i] += 1.0
```

Note: both pre and post traces are updated for every neuron. The pre/post distinction is relative — every neuron is pre-synaptic to some and post-synaptic to others.

### 2. make_stdp_intra(B, K)

Runs every timestep. Updates `W_intra` based on traces and spikes within each block.

```python
@ti.kernel
def stdp_intra(
    W_intra: ti.template(), spikes: ti.template(),
    trace_pre: ti.template(), trace_post: ti.template(),
    a_plus: ti.f32, a_minus: ti.f32, w_max: ti.f32,
) -> None:
    for block_idx, i in ti.ndrange(B, K):
        if spikes[block_idx, i] == 1:
            # Post-synaptic neuron i spiked → potentiate incoming synapses
            for j in range(K):
                if W_intra[block_idx, j, i] != 0.0:  # only existing synapses
                    dw = a_plus * trace_pre[block_idx, j]
                    W_intra[block_idx, j, i] += dw
                    # Clamp
                    W_intra[block_idx, j, i] = ti.min(W_intra[block_idx, j, i], w_max)
    # Second pass for LTD
    for block_idx, j in ti.ndrange(B, K):
        if spikes[block_idx, j] == 1:
            # Pre-synaptic neuron j spiked → depress outgoing synapses
            for i in range(K):
                if W_intra[block_idx, j, i] != 0.0:
                    dw = a_minus * trace_post[block_idx, i]
                    W_intra[block_idx, j, i] -= dw
                    # Clamp (respect sign: exc ≥ 0, inh ≤ 0)
                    if W_intra[block_idx, j, i] > 0.0:
                        W_intra[block_idx, j, i] = ti.max(W_intra[block_idx, j, i], 0.0)
                    else:
                        W_intra[block_idx, j, i] = ti.min(W_intra[block_idx, j, i], 0.0)
```

**Important detail — the `!= 0.0` guard:** STDP only modifies existing synapses (non-zero weights). Creating new synapses is structural plasticity's job (WC-7). Without this guard, STDP would densify the weight matrix in a few timesteps, defeating the sparse initialization.

**Weight clamping:** Excitatory weights clamp to `[0, w_max]`, inhibitory to `[-w_max, 0]`. The sign of a weight is determined at init by neuron type and never changes. STDP moves the magnitude, not the sign.

**Two-pass structure:** LTP and LTD are separate loops because they iterate over different indices (post-synaptic for LTP, pre-synaptic for LTD). Fusing them into one loop is possible but makes the code harder to reason about. Profile before optimizing.

### 3. make_stdp_inter(B, K, num_neighbors)

Takes a `direction: ti.i32` argument. Updates only `W_inter[:, direction, :, :]`.

```python
@ti.kernel
def stdp_inter(
    W_inter: ti.template(), spikes: ti.template(),
    trace_pre: ti.template(), trace_post: ti.template(),
    direction: ti.i32,
    a_plus: ti.f32, a_minus: ti.f32, w_max: ti.f32,
) -> None:
    for block_idx, i in ti.ndrange(B, K):
        nb = _neighbor(block_idx, direction)
        if spikes[block_idx, i] == 1:
            # Post i in this block spiked, potentiate from pre j in neighbor
            for j in range(K):
                if W_inter[block_idx, direction, j, i] != 0.0:
                    dw = a_plus * trace_pre[nb, j]
                    W_inter[block_idx, direction, j, i] += dw
                    W_inter[block_idx, direction, j, i] = ti.min(
                        W_inter[block_idx, direction, j, i], w_max
                    )
        # LTD: pre j in neighbor spiked
        if spikes[nb, ???] ...  # see note below
```

**Complexity note:** Inter-block STDP is trickier because the pre-synaptic neuron is in a different block. The LTD pass needs to check if neurons in the NEIGHBOR block spiked and depress their connections INTO this block. This reversal of iteration direction must be handled carefully to avoid race conditions (two blocks writing to the same weight simultaneously).

**Solution:** Each block owns its own `W_inter[block_idx, direction, :, :]`. LTP iterates over post-synaptic spikes in this block (reads neighbor traces — read-only, safe). LTD iterates over pre-synaptic spikes in the neighbor (reads this block's traces — read-only, safe) and writes to this block's weights (local write, safe). No cross-block writes.

## Rotating vs All Mode

Controlled by `config.stdp.inter_mode`:

**Rotating** (default): each timestep, only one direction gets STDP updates.
```python
direction = timestep % num_neighbors
stdp_inter(..., direction=direction)
```

**All**: every direction gets updated every timestep.
```python
for d in range(num_neighbors):
    stdp_inter(..., direction=d)
```

Rotating reduces inter-block STDP write bandwidth by `2*ndim` (6x for 3D). The traces carry enough temporal information that sampling each direction every `2*ndim` timesteps (3ms for 3D at dt=0.5) is well within the ~20ms STDP window.

The "all" mode is provided for experiments comparing learning speed vs computational cost. Hypothesis: "all" learns faster initially but costs `2*ndim` more compute. For large networks, rotating is likely necessary.

## Tests

### LTP: Pre-Before-Post

1. Force neuron A to spike at t=5 (set `I_ext` high for one step).
2. Force neuron B to spike at t=10 (5ms later at dt=0.5).
3. Run STDP. Check `W[A, B]` increased.
4. The increase should be proportional to `A_plus * trace_pre[A]` at the time B spikes, which is `A_plus * exp(-5ms / tau_pre)`.

### LTD: Post-Before-Pre

1. Force neuron B to spike at t=5.
2. Force neuron A to spike at t=10.
3. Run STDP. Check `W[A, B]` decreased.
4. Decrease proportional to `A_minus * trace_post[B]` at the time A spikes.

### Asymmetry: |LTD| > |LTP| for Same |Δt|

With default params (`A_minus=0.012 > A_plus=0.01`), the LTD change should be larger in magnitude than the LTP change for the same time difference. This ensures net depression under random activity.

### Trace Decay

- Set `trace_pre[0, 0] = 1.0`, run trace_update for N steps with no spikes.
- After N steps: `trace ≈ exp(-N * dt / tau_pre)`.
- Verify within 1% tolerance.

### Trace Accumulation

- Spike the same neuron twice, 10 steps apart. After the second spike, `trace = exp(-10*dt/tau) + 1.0`, not just 1.0. Traces accumulate — they don't reset on spike.

### Rotating Mode

- Run `stdp_inter` with `direction=0`. Verify only `W_inter[:, 0, :, :]` changed.
- All other directions' weights remain identical to their initial values.

### Weight Clamping

- Set an excitatory weight to `w_max - 0.001`. Apply LTP. Weight should clamp to `w_max`, not exceed it.
- Set an excitatory weight to `0.001`. Apply LTD. Weight should clamp to `0.0`, not go negative.
- Set an inhibitory weight to `-w_max + 0.001`. Apply LTD. Should clamp to `-w_max`.

### Synapse Gate (Non-Zero Guard)

- Set `W_intra[b, j, i] = 0.0` (no synapse). Spike both j and i. After STDP, weight should remain exactly 0.0 — STDP doesn't create synapses.
