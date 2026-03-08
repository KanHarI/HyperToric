# Synthetic Biological Intelligence Simulator

A GPU-accelerated spiking neural network with dynamic topology, inspired by Cortical Labs' DishBrain/CL1 biological computer. Not intended to compete with conventional neural networks — the goal is to build a fast, minimal simulation of the computational principles underlying biological neural cultures on silicon, suitable for algorithm development and experimentation.

## Motivation

Cortical Labs grows ~800,000 human neurons on a multi-electrode array. These neurons self-organize, form and dissolve synapses, and learn tasks (Pong, Doom) within minutes — guided not by backpropagation but by the Free Energy Principle: neurons inherently reorganize to minimize the unpredictability of their sensory inputs. The "training signal" is simply the difference between ordered and chaotic electrical feedback.

No existing simulator packages this together cleanly. This project builds the minimal computational substrate that captures the essential dynamics: spiking neurons, spatial layout, STDP, activity-dependent structural plasticity, and the DishBrain feedback protocol — all on GPU.

## Architecture

### Topology: 3D Hypertorus of Blocks

64 blocks arranged in a 4×4×4 grid with **periodic boundary conditions** (hypertorus). Every block has exactly 6 face-neighbors — no corners, no edges, no boundary effects.

This is a deliberate departure from the physical constraints of the CL1's 2D MEA surface. The hypertorus gives us:

- **Uniform topology.** Every block is structurally identical. No neuron is disadvantaged by position.
- **Short path lengths.** Diameter of 6 hops (vs 9 on flat grid). Information flows efficiently.
- **No privileged directions.** Sensory and motor regions can be placed anywhere. Routing emerges.
- **Clean GPU mapping.** No boundary conditionals. Every block has the same shape, same neighbor count, same kernel launch profile.

Neighbor lookup is trivial modular arithmetic:

```python
def get_neighbors(x, y, z, size=4):
    return [((x+d[0]) % size, (y+d[1]) % size, (z+d[2]) % size)
            for d in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]
```

### Neuron Model: Izhikevich

Two coupled ODEs per neuron:

```
dv/dt = 0.04v² + 5v + 140 - u + I
du/dt = a(bv - u)
if v >= 30: spike, v = c, u += d
```

- **v**: membrane potential (the thing that spikes)
- **u**: recovery variable (slow ionic currents that pull v back down)
- **I**: total synaptic input + external current injection
- **(a, b, c, d)**: parameters that select neuron type

Neuron types by parameter choice:

| Type | a | b | c | d | Role |
|------|------|------|------|------|------|
| Regular spiking | 0.02 | 0.2 | -65 | 8 | Excitatory |
| Fast spiking | 0.1 | 0.2 | -65 | 2 | Inhibitory |
| Chattering | 0.02 | 0.2 | -50 | 2 | Excitatory |
| Intrinsically bursting | 0.02 | 0.2 | -55 | 4 | Excitatory |

~10 FLOPs per neuron per timestep. The bottleneck is never the neuron update.

Discretized with half-step Euler for numerical stability at dt = 0.5–1ms.

### Synaptic Input

For each neuron i:

```
I_i = Σ_j W[j,i] * s_j(t - delay) + I_external(t)
```

Post-synaptic currents modeled as exponentially decaying traces for biological realism:

```
I_syn[i] += W[j,i]              // on incoming spike from j
I_syn[i] *= exp(-dt / tau_syn)  // decay every timestep
```

tau_syn ≈ 2–5ms (excitatory/AMPA), 5–10ms (inhibitory/GABA).

### Connectivity

**Intra-block: dense matrix.** K×K weight matrix per block (K = 256–512 neurons per block). Fits in GPU shared memory. Spike propagation is a small dense matrix-vector multiply.

**Inter-block: dense but sparse-initialized.** K×K matrix per neighbor pair, initialized at low connection probability (~5–10%) or zero, with structural plasticity growing connections over time.

Memory budget at K=256, 6 neighbors: ~1.5MB per block, ~100MB total. Trivial for any modern GPU.

### STDP (Spike-Timing-Dependent Plasticity)

Trace-based implementation — no spike history search:

```
// Every timestep: decay traces
x_pre[i]  *= exp(-dt / tau_pre)
x_post[i] *= exp(-dt / tau_post)

// When neuron i spikes:
x_pre[i] += 1.0
W[i, :] -= A_minus * x_post[:]   // depress outgoing synapses

// When neuron j spikes:
x_post[j] += 1.0
W[:, j] += A_plus * x_pre[:]     // potentiate incoming synapses
```

A_minus > A_plus ensures net depression under random activity — only genuinely correlated causal pathways survive.

With dense matrices this reduces to rank-1 updates (outer products of spike vectors with trace vectors). Extremely GPU-friendly.

#### Rotating Inter-Block STDP

Key optimization: **spike propagation** happens to all 6 neighbors every timestep (required for correct dynamics), but **STDP weight updates** for inter-block connections rotate through one direction per timestep:

```
t % 6 == 0: all blocks update STDP with +x neighbor
t % 6 == 1: all blocks update STDP with -x neighbor
t % 6 == 2: all blocks update STDP with +y neighbor
t % 6 == 3: all blocks update STDP with -y neighbor
t % 6 == 4: all blocks update STDP with +z neighbor
t % 6 == 5: all blocks update STDP with -z neighbor
```

This is valid because STDP traces have time constants of ~20ms. Cycling through all 6 directions every 6 timesteps (~3–6ms) is well within the STDP window. The traces carry the temporal information — they decay continuously regardless of when they're sampled for a weight update.

Why this works well:

- **6x reduction in inter-block write bandwidth.** The expensive read-modify-write of weight matrices happens for 1 direction instead of 6. Spike propagation (read-only, cheap) still covers all 6.
- **Uniform GPU workload.** Every block does the same direction on the same timestep. Zero divergence, predictable memory access.
- **Bidirectional consistency.** When block A updates its +x connection to block B, block B simultaneously updates its -x connection back to A. Both sides of a link update on consecutive timesteps.
- **Matches the biological intuition.** Intra-block connections get full-rate STDP — high-frequency local learning. Inter-block connections learn at 1/6th sample rate — slower, coarser long-range adaptation. This is analogous to the biological reality where local synaptic modification is fast and long-range axonal plasticity is slower.

Structural plasticity (synapse creation/pruning) follows the same asymmetry: intra-block rewiring uses all spike data, inter-block rewiring operates on the sparser sample.

### Structural Plasticity

Three timescales of reorganization, with an intra/inter asymmetry matching the STDP schedule:

1. **Synaptic weights (fast, every timestep).** STDP strengthens causal pathways, weakens non-causal ones. Intra-block: every timestep. Inter-block: 1/6th rate via rotating direction.

2. **Intra-block topology (medium, every ~1s).** Activity-dependent synapse creation and pruning using full spike data:
   - Maintain a calcium trace per neuron (exponentially filtered spike count)
   - Below homeostatic target → neuron grows new incoming synapses (sampled with distance bias within block)
   - Above target → prune weakest incoming synapses
   - Synapses below weight threshold → pruned; freed slot available for new connections

3. **Inter-block topology (slow, every ~10s).** Coarser restructuring using the sparser inter-block activity sample:
   - If boundary neurons consistently receive useful drive from a neighbor → grow more inter-block connections
   - If an inter-block pathway carries no correlated activity → prune it
   - Optionally: dynamic neighbor selection (drop an inactive inter-block connection, open one to a different neighbor)

## I/O Topology: Planar Input and Output with Torus Processing

Sensory and motor neurons occupy opposite planes of the hypertorus, giving the network an initial directional bias for information flow while preserving torus advantages in all other dimensions.

### Layout

**Sensory plane: x=0.** All 16 blocks at `(0, a, b)` for `a, b ∈ {0,1,2,3}` contain sensory neuron clusters. Each block gets a subset of sensory channels — small clusters (16–32 out of K=256), not the entire block. Ball position is distributed across the plane using place coding: which block in the y-z plane is stimulated encodes coarse position.

**Motor plane: x=2.** Blocks at `(2, a, b)` contain motor neuron clusters. A subset of these blocks host "up" populations, another subset host "down" populations.

**Processing planes: x=1 and x=3.** Pure processing — no I/O neurons. These 32 blocks self-organize entirely through activity-dependent plasticity.

```python
# Sensory clusters distributed across the x=0 plane
sensory_blocks = [(0, a, b) for a in range(GRID) for b in range(GRID)]

# Motor clusters in the x=2 plane
motor_up_blocks   = [(2, a, b) for a in range(2) for b in range(GRID)]  # half the plane
motor_down_blocks = [(2, a, b) for a in range(2,4) for b in range(GRID)]  # other half
```

### Why this works on the hypertorus

On a 4-torus, x=0 to x=2 is exactly 2 hops in *both* directions: through x=1 (the "front" path) or through x=3 (the "back" path). The network gets **two parallel processing pathways for free**. Both intermediate planes can develop different processing strategies. The network discovers which route to use, or uses both.

Within each plane, the y and z dimensions still wrap around — no edges, no corners. Every block in the sensory plane has the same local topology. Every block in the motor plane has the same local topology.

### Why not co-located

If sensory and motor neurons share a block, the intra-block dense matrix provides a direct input→output pathway that STDP finds almost immediately. The learning is trivially fast — just training a single K×K weight matrix. The other 63 blocks are dead weight.

### What the network has to solve

1. **Multi-hop relay.** Sensory signals must traverse at least 2 inter-block connections to reach motor output. The intermediate blocks must develop useful relay or transformation pathways.
2. **Convergence within planes.** The sensory plane distributes ball position across 16 blocks. Motor blocks need integrated information, not raw single-channel signals. Lateral connections within and between planes must aggregate.
3. **Dual pathway selection.** Signals can flow x=0→1→2 or x=0→3→2. The network may learn to use one path, both in parallel, or different paths for different aspects of the task.
4. **Specialization.** Processing blocks at x=1 and x=3 self-organize into different roles — some relay, some integrate, some may go dormant. Observable in the connectivity patterns that emerge.

## Training Protocol

Following the core principles of Kagan et al. (2022), adapted with graded feedback.

**Sensory encoding.** Inject current into sensory neuron clusters across the x=0 plane. Encode target/ball position using combined rate + place coding: which block in the y-z plane is stimulated encodes coarse position, firing rate encodes fine position.

**Motor decoding.** Read spike rates from motor neuron clusters in the x=2 plane. Differential spike rate between "up" and "down" motor populations controls cursor/paddle movement.

## Feedback Protocol

### Feedback Channel = Sensory Channel

Critical design constraint: feedback is delivered to the **same neurons** that carry sensory input. This is not incidental — it is why the protocol works.

If feedback went to separate neurons, the network would learn to route around them. STDP would build pathways from sensory→processing→motor that bypass the feedback population entirely. The chaos signal becomes ignorable background noise. The punishment is decoupled from the information source, so there's no pressure to change the sensory-motor mapping.

When feedback hits the same neurons as input, the network faces an inescapable dilemma: it *must* listen to the sensory neurons to get input, but those same neurons deliver chaos on failure. The only way to reduce unpredictability is to act on the sensory information correctly — because correct action makes those neurons predictable again. The pathways built to process sensory input are exactly the pathways disrupted by chaos feedback.

In this implementation: all feedback goes to sensory neuron clusters in the x=0 plane.

### Graded Feedback: Entropy as a Continuous Signal

Rather than binary hit/miss, feedback is graded by performance quality. The signal mixes order and chaos in proportion to how well the network is performing, creating an energy landscape the network can roll down.

**Mixing ratio** based on distance between cursor and target:

```python
# p: probability of ordered pulse. 1.0 = pure order, 0.0 = pure chaos
if distance == 0:
    p = 1.0    # perfectly predictable
elif distance == 1:
    p = 0.7    # mostly predictable, occasional noise
elif distance == 2:
    p = 0.3    # mostly chaotic, some structure
else:
    p = 0.0    # pure chaos
```

**Stochastic mixing, not deterministic blending.** On each feedback pulse, flip a biased coin: with probability p deliver the ordered pattern, with probability (1-p) deliver a random pattern.

```python
# Per feedback pulse
if random() < p:
    deliver_ordered_pulse(sensory_neurons)   # 100Hz synchronized, 100ms
else:
    deliver_random_noise(sensory_neurons)    # random spatiotemporal patterns
```

Why stochastic: a deterministic blend of 70% order + 30% noise is itself a consistent pattern the neuron can learn to model. It would plateau at "I've learned to predict this murky signal" without pressure to improve. Stochastic mixing means the noise component is irreducible — the only way to reduce free energy is to increase p by improving performance.

**Three distinct regimes the network experiences:**

- **distance 0:** Inputs are always predictable. Free energy at floor. No pressure to change. Reward.
- **distance 1:** Inputs are usually predictable with occasional surprises. Free energy is low but nonzero. Gentle gradient toward improvement — existing pathways are preserved but there's pressure to refine.
- **distance 2+:** Inputs are mostly or fully chaotic. Free energy is high. Strong pressure to reorganize. Existing pathways actively disrupted.

This creates a smooth optimization landscape rather than a cliff. The network can discover "I got warmer" before it discovers "I got there."

No explicit reward signal. No loss function. No gradient computation. Just the statistical structure of the sensory input, shaped by performance, exploiting the neurons' intrinsic drive to minimize unpredictability (Free Energy Principle).

## First Task: 1D Target Tracking

Pong is too complex for initial testing — ball physics, timing, trajectory prediction all confound the core question: *can the network learn any sensory-motor mapping through order-vs-chaos feedback?*

### Setup

A target position occupies one of 8 discrete locations along a 1D axis (mapped to sensory blocks in the y-z plane of x=0). The network controls a cursor position via motor output.

```
Target:  ·  ·  ·  ·  ■  ·  ·  ·     (position 4)
Cursor:  ·  ·  ■  ·  ·  ·  ·  ·     (position 2 — distance 2)
```

**Sensory encoding.** Inject current into sensory block(s) at x=0 corresponding to target position. Gaussian place coding: primary block gets strong stimulation, neighboring blocks get weaker stimulation. Gives the network a spatial gradient, not just a single active channel.

**Motor decoding.** Differential spike rate between "up" and "down" motor populations in the x=2 plane. Cursor moves one position per game tick in the dominant direction.

**Feedback.** Graded per the protocol above. Applied to the same sensory neurons carrying target position.

### Difficulty Progression

Each level isolates a different mechanism. If a level fails, it diagnoses where the problem is.

1. **Static target.** Target stays at one position indefinitely. Network must learn "this sensory pattern → move cursor here and stop." Simplest possible test. Tests basic sensory-motor pathway formation. *If this doesn't work, there's a bug in the pipeline.*

2. **Slow step.** Target jumps to a new random position every ~5 seconds. Network must re-adapt cursor position after each jump. Tests whether learning generalizes across positions or memorizes one. *If this fails, structural plasticity isn't generalizing.*

3. **Slow ramp.** Target moves one position per second, bouncing at boundaries. Closest to Pong without the ball. Tests continuous tracking. *If this fails, temporal dynamics need tuning.*

4. **Sine wave.** Target follows a smooth periodic trajectory. Tests predictive tracking — can the network learn to move the cursor *before* the target arrives? *If this works, the network has learned temporal structure, not just reactive mapping.*

### Measurement

Primary metric: absolute distance between cursor and target, averaged over a sliding window. Plot across training time. Should decrease if learning is happening.

Secondary: per-position accuracy (does the network learn some target locations faster than others — reveals spatial organization), and pathway analysis (which processing blocks at x=1 and x=3 develop the strongest inter-block connections — reveals emergent routing).

### Graduation to Pong

Once ramp tracking works, Pong is a small step: ball position replaces target position, timing pressure increases (ball moves faster), and discrete hit/miss events replace continuous proximity feedback. Sensory encoding and motor decoding stay identical.

## Execution Model

```
Per timestep (~1ms):
  neuron_update()                — Izhikevich step, parallel over all neurons
  spike_propagate()              — intra-block + ALL 6 inter-block (read-only on W)
  stdp_intra()                   — full intra-block trace decay + rank-1 weight updates
  stdp_inter(t % 6)             — ONE inter-block direction, rotating

Per structural plasticity interval (~1s):
  structural_intra()             — prune/grow intra-block synapses based on calcium
  structural_inter(t_slow % 6)  — prune/grow inter-block synapses (sparser sample)

Per block rewiring interval (~10s):
  block_rewire()                 — adjust inter-block connectivity at block level

On CPU (per game tick):
  Environment step (target tracking / Pong) → sensory currents / feedback signals
```

The critical split: **propagation is global, learning is local-first.** Every timestep, all spikes reach all neighbors (6 read-only matmuls per block). But weight modification is concentrated intra-block (full rate) with inter-block learning amortized at 1/6th rate across the rotating direction schedule.

Taichi auto-parallelizes the `for bx, by, bz, i in field` loops across GPU threads. Each neural block's data is contiguous in memory. Inter-block reads access neighbor data at known offsets. Inter-block writes hit only one neighbor's weight matrix per timestep.

## Implementation Stack: Taichi

### Why Taichi

The core constraint: must run with GPU acceleration on both NVIDIA (CUDA) and Apple Silicon (Metal), and fall back to CPU gracefully. Must be `pip install`-able with no build toolchain.

Alternatives considered:

- **PyTorch custom extensions.** The workload (sparse event-driven updates, conditional resets, irregular structural plasticity) doesn't map to PyTorch's tensor abstraction. Most time would be spent fighting the framework. MPS support for custom ops is incomplete — Mac users would get CPU fallback anyway.
- **CuPy / raw CUDA.** Good NVIDIA performance, but CUDA doesn't exist on macOS (Apple killed it in 2019). No Metal backend. CuPy's Python-level kernel dispatch adds overhead for the many small operations in this workload.
- **C++ core with pybind11 + multiple backends.** Maximum performance but requires maintaining CUDA, Metal, and CPU codepaths plus a complex build matrix (Linux+CUDA, macOS+Metal/CPU, Windows+CUDA). Too much plumbing before reaching the interesting work.
- **NVIDIA Warp.** Similar to Taichi but NVIDIA-only. Eliminates macOS GPU story entirely.

**Taichi** compiles a single Python-embedded DSL to CUDA, Metal, Vulkan, and CPU. `pip install taichi` just works. It was designed for exactly this class of workload — spatially-structured parallel computation over grid-based data.

### Data Layout

Taichi fields map directly to the hypertorus block structure:

```python
import taichi as ti
ti.init(arch=ti.gpu)  # auto-selects CUDA, Metal, or CPU

K = 256
GRID = 4

v          = ti.field(dtype=ti.f32, shape=(GRID, GRID, GRID, K))
u          = ti.field(dtype=ti.f32, shape=(GRID, GRID, GRID, K))
spikes     = ti.field(dtype=ti.i32, shape=(GRID, GRID, GRID, K))
I_syn      = ti.field(dtype=ti.f32, shape=(GRID, GRID, GRID, K))
calcium    = ti.field(dtype=ti.f32, shape=(GRID, GRID, GRID, K))
trace_pre  = ti.field(dtype=ti.f32, shape=(GRID, GRID, GRID, K))
trace_post = ti.field(dtype=ti.f32, shape=(GRID, GRID, GRID, K))

W_intra    = ti.field(dtype=ti.f32, shape=(GRID, GRID, GRID, K, K))
W_inter    = ti.field(dtype=ti.f32, shape=(GRID, GRID, GRID, 6, K, K))
```

Memory at K=256: ~100MB total. Comfortable on any modern GPU.

### Kernel Structure

```python
@ti.kernel
def neuron_update():
    for bx, by, bz, i in v:
        vv = v[bx, by, bz, i]
        uu = u[bx, by, bz, i]
        I = I_syn[bx, by, bz, i]
        vv += 0.5 * (0.04*vv*vv + 5.0*vv + 140.0 - uu + I)
        vv += 0.5 * (0.04*vv*vv + 5.0*vv + 140.0 - uu + I)
        uu += a * (b * vv - uu)
        if vv >= 30.0:
            spikes[bx, by, bz, i] = 1
            vv = c
            uu += d
        else:
            spikes[bx, by, bz, i] = 0
        v[bx, by, bz, i] = vv
        u[bx, by, bz, i] = uu

@ti.func
def neighbor(bx: ti.i32, by: ti.i32, bz: ti.i32, d: ti.i32) -> ti.math.ivec3:
    offsets = ti.Matrix([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ])
    return ti.math.ivec3(
        (bx + offsets[d,0]) % GRID,
        (by + offsets[d,1]) % GRID,
        (bz + offsets[d,2]) % GRID
    )

@ti.kernel
def spike_propagate():
    # Intra-block + all 6 inter-block directions (read-only on W)
    for bx, by, bz, i in I_syn:
        acc = 0.0
        for j in range(K):
            acc += W_intra[bx, by, bz, j, i] * spikes[bx, by, bz, j]
        for d in range(6):
            n = neighbor(bx, by, bz, d)
            for j in range(K):
                acc += W_inter[bx, by, bz, d, j, i] * spikes[n.x, n.y, n.z, j]
        I_syn[bx, by, bz, i] = acc

@ti.kernel
def stdp_inter(direction: ti.i32):
    # One direction per call — rotated by caller
    for bx, by, bz, i in trace_pre:
        n = neighbor(bx, by, bz, direction)
        for j in range(K):
            W_inter[bx, by, bz, direction, i, j] += (
                A_plus * trace_pre[bx, by, bz, i] * spikes[n.x, n.y, n.z, j]
              - A_minus * trace_post[n.x, n.y, n.z, j] * spikes[bx, by, bz, i]
            )
```

Taichi's parallel `for` over fields auto-parallelizes across GPU threads. The `(GRID, GRID, GRID, K)` iteration maps naturally to the 64-block × K-neuron structure. No manual thread/block management, no boundary conditionals on the hypertorus.

### Performance Expectations

Taichi compiles to optimized CUDA PTX or Metal shaders. For this workload (structured grid, parallel field access, basic arithmetic, no complex control flow), expected performance is within 2–3x of hand-written CUDA. If a specific kernel becomes a bottleneck, it can be replaced with a raw CUDA kernel via Taichi's AOT compilation or CuPy RawKernel while keeping the rest in Taichi.

### Installation

```bash
pip install taichi
```

No C++ compiler, no CUDA toolkit, no build toolchain. Works on Linux (CUDA/Vulkan), macOS (Metal/CPU), Windows (CUDA/Vulkan).

## What This Captures vs. What It Doesn't

**Captured:**
- Spiking dynamics with biologically realistic neuron types
- Spatial topology with local connectivity
- STDP (unsupervised causal pathway reinforcement)
- Activity-dependent structural plasticity (synapse creation/pruning)
- The DishBrain feedback protocol (order vs. chaos)
- Free Energy Principle dynamics (emergent, not explicitly programmed)

**Not captured:**
- Dendritic computation (point neurons only)
- Continuous neurite outgrowth (discrete synapse events)
- Ion channel diversity
- Glial cell interactions
- Neuromodulatory systems (dopamine, serotonin, etc.)
- Actual biological timescales of growth (compressed)

## References

- Kagan, B.J. et al. (2022). "In vitro neurons learn and exhibit sentience when embodied in a simulated game-world." *Neuron*, 110(23), 3952–3969.
- Izhikevich, E.M. (2003). "Simple model of spiking neurons." *IEEE Transactions on Neural Networks*, 14(6), 1569–1572.
- Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127–138.
- Pfister, J.P. & Gerstner, W. (2006). "Triplets of spikes in a model of spike timing-dependent plasticity." *Journal of Neuroscience*, 26(38), 9673–9682.
- Butz, M. & van Ooyen, A. (2013). "A simple rule for dendritic spine and axonal bouton formation can account for cortical reorganization after focal retinal lesions." *PLoS Comput. Biol.*, 9(10), e1003259.

