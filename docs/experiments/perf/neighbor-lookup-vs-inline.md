# Experiment: Precomputed Neighbor Lookup Table vs Inline Modular Arithmetic

## Question

Should inter-block neighbor computation use a precomputed lookup table stored in a Taichi field, or inline modular arithmetic computed per-thread?

## Hypothesis

Inline arithmetic is faster for small ndim (2-4) because integer arithmetic uses only registers, while a lookup table requires a global memory read that may cache-miss.

## Current Decision

**Inline arithmetic** — used in all kernels (`propagate.py`, `stdp.py`, `plasticity.py`).

## Rationale

The neighbor computation for flat index `block_idx` in direction `d`:

```python
stride = G ** axis          # compile-time constant
coord = (block_idx // stride) % G
new_coord = (coord + offset) % G
neighbor = block_idx + (new_coord - coord) * stride
```

This is 5 integer operations per direction. On modern GPUs, integer ALU throughput is high (hundreds of GOPS). The total cost per neuron is `5 * 2 * ndim` integer ops — at most 40 ops for 4D.

The lookup table alternative would be a field of shape `(B, 2*ndim)` storing precomputed neighbor indices. Reading it costs one global memory access per direction. On GPU, global memory latency is 200-800 cycles, partially hidden by warp scheduling. But since the propagation kernel is already memory-bandwidth-limited (reading weight matrices), adding more memory traffic doesn't help — it competes for the same bandwidth.

## Benchmark Plan

Once the codebase is functional:

1. Implement a `_neighbor_lut` version that reads from a precomputed field.
2. Run both versions on the propagation kernel with:
   - ndim=2, grid_size=8, K=256
   - ndim=3, grid_size=4, K=256
   - ndim=4, grid_size=3, K=128
3. Measure kernel execution time (Taichi profiler or `ti.sync()` + wall clock).
4. Compare on both CUDA and Metal backends.

## Expected Result

Inline wins or ties for ndim ≤ 4. The lookup table might win for ndim > 6 (unlikely use case) where the unrolled loop becomes long enough that instruction cache pressure matters.

## Status

Not yet benchmarked. Decision made based on reasoning. Revisit if profiling shows neighbor computation as a bottleneck (unlikely — weight matrix reads dominate).
