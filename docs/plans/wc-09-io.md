# WC-9: I/O Manager

**Branch:** `wc-9-io`
**Dependencies:** WC-7 (structural plasticity), WC-8 (simulator)

## Goal

Map between the external world (task state) and the network's sensory/motor planes. The I/O manager encodes target positions as spatiotemporal current patterns on the sensory plane, decodes motor plane spike rates into cursor movements, and delivers feedback (order/chaos) through the sensory channel. All plane selection is ndim-agnostic via the topology module.

## Files

| File | Action | Notes |
|------|--------|-------|
| `src/hypertoric/io.py` | Create | IOManager class |
| `tests/test_io.py` | Create | Plane selection, encoding, decoding, feedback |

## Plane Selection: ndim-Agnostic

The sensory and motor planes are hyperplanes through the torus. In 3D with `sensory_axis=0, sensory_position=0`, the sensory plane is all blocks at `x=0` — a 4×4 grid of 16 blocks. In 2D with the same settings, it's all blocks at `x=0` — a 1D row of 4 blocks.

The IOManager asks the topology for these at construction time:

```python
self.sensory_blocks = topology.get_plane(config.io.sensory_axis, config.io.sensory_position)
self.motor_blocks = topology.get_plane(config.io.sensory_axis, config.io.motor_position)
```

Both planes are slices along the SAME axis (`sensory_axis`) at different positions. This ensures the shortest path between sensory and motor goes through the processing blocks in between.

### Motor Sub-Populations

Motor output requires two populations: "up" and "down" (or more generally, positive and negative direction). These are divided by neuron index within each motor block:

- **Up population:** neurons `0` to `K // 2 - 1` in each motor block
- **Down population:** neurons `K // 2` to `K - 1` in each motor block

Alternative: split by block (half the motor blocks are "up", half are "down"). This was the original plan, but splitting by neuron index is simpler and works regardless of how many motor blocks exist. The block-based split requires at least 2 motor blocks, which might not hold for small 2D grids.

## Sensory Encoding: Gaussian Place Coding

### The Problem

A target position (e.g., position 4 out of 8) must be converted into current injection patterns across the sensory plane. The encoding must give the network enough spatial information to distinguish positions AND provide a gradient signal (nearby positions should produce similar patterns).

### Place Coding

Each sensory block maps to a position along the target axis. For a sensory plane of `P` blocks (e.g., 16 blocks in the 3D case), we linearize them and assign each a "preferred position" along the target axis.

For target position `t` (float in [0, 1] normalized):

```python
for idx, block_id in enumerate(self.sensory_blocks):
    preferred = idx / len(self.sensory_blocks)  # position this block "likes"
    # Torus-aware distance (positions wrap)
    dist = min(abs(t - preferred), 1.0 - abs(t - preferred))
    amplitude = base_current * exp(-dist**2 / (2 * sigma**2))
    # Inject into sensory cluster within this block
    for neuron in range(config.io.sensory_cluster_size):
        fields.I_ext[block_id, neuron] = amplitude
```

**Gaussian width (`sigma`):** Controls how many blocks respond to each target. Too narrow (sigma=0.01) → only one block fires, no spatial gradient. Too wide (sigma=0.5) → all blocks fire equally, no position discrimination. Start with `sigma = 1.0 / len(sensory_blocks)` (one block width).

**Sensory cluster size:** Not all K neurons in a sensory block receive input — only `sensory_cluster_size` (default 32 out of 256). The remaining neurons in the block are "processing" neurons that can form associations between sensory input and internal dynamics. This gives the block-local circuitry room to transform the input before relaying it.

### Torus-Aware Distance

The sensory plane's blocks have torus topology in the non-sensory axes. Positions wrap around. The distance function must account for this: the distance from position 0 to position 7 on an 8-position axis is 1, not 7.

## Motor Decoding: Adaptive Threshold

### The Problem

Motor populations fire irregularly. We need to convert noisy spike trains into discrete cursor movements at a rate the game can use (~50ms per game tick). The challenge: the network's baseline firing rates change as it learns, so a fixed threshold drifts.

### Rate Estimation: Exponential Trace

Maintain a running rate estimate for each motor population:

```python
# Every simulation timestep
self.rate_up *= self.decay      # decay = exp(-dt / tau_motor)
self.rate_down *= self.decay
self.rate_up += spike_count_up_this_step    # count of spikes in up-population this step
self.rate_down += spike_count_down_this_step
```

`tau_motor` (~20ms) controls the smoothing window. Shorter = more responsive but noisier. Longer = smoother but adds latency.

### Adaptive Threshold with Dead Zone

```python
# Every game tick (every ~50 timesteps)
diff = self.rate_up - self.rate_down

# Running statistics
self.diff_mean = self.diff_mean * self.momentum + (1 - self.momentum) * diff
self.diff_var = self.diff_var * self.momentum + (1 - self.momentum) * (diff - self.diff_mean) ** 2

# Threshold
threshold = self.k * math.sqrt(self.diff_var + 1e-8)  # epsilon for stability
centered = diff - self.diff_mean

if centered > threshold:
    return +1  # move cursor up
elif centered < -threshold:
    return -1  # move cursor down
else:
    return 0   # no movement (dead zone)
```

**Why adaptive:**
- **Mean centering** cancels tonic asymmetry — if the network has a permanent bias toward "up" neurons, it doesn't cause constant drift.
- **Variance tracking** scales the dead zone to current noise level. Early in training (high noise), the dead zone is wide → fewer false movements. As the network organizes (lower noise), the dead zone narrows → more responsive.
- **Momentum 0.999** averages over ~1000 timesteps (~500ms). Matches the structural plasticity timescale, so the threshold adapts as fast as the network rewires.

### Parameters to Tune (see `docs/experiments/perf/motor-sampling-params.md`)

| Parameter | Default | Range to explore | Effect |
|-----------|---------|-------------------|--------|
| `tau_motor` | 20ms | 5-50ms | Rate smoothing window |
| `k` | 1.5 | 0.5-3.0 | Dead zone width (std devs) |
| `momentum` | 0.999 | 0.99-0.9999 | Statistics adaptation speed |
| Game tick | 50 timesteps (25ms) | 20-200 timesteps | Decision rate |

## Feedback Protocol

### Order vs Chaos Mixing

Feedback goes to the SAME sensory neurons that carry input (see architecture doc for why this is essential). The mixing ratio `p` depends on cursor-target distance:

```python
def _mixing_ratio(self, distance: int) -> float:
    if distance == 0:
        return 1.0    # pure order
    elif distance == 1:
        return 0.7
    elif distance == 2:
        return 0.3
    else:
        return 0.0    # pure chaos
```

### Ordered Pulse

Synchronized, predictable input — all sensory neurons in the cluster fire together at ~100Hz for ~100ms. This is a pattern the network can predict, so its free energy is low.

```python
def _deliver_ordered_pulse(self, fields: SimFields) -> None:
    amplitude = 15.0  # strong enough to reliably trigger spikes
    for block_id in self.sensory_blocks:
        for neuron in range(self.sensory_cluster_size):
            fields.I_ext[block_id, neuron] = amplitude
```

### Chaotic Pulse

Random spatiotemporal pattern — each sensory neuron gets independent random current. The network can't predict this, so its free energy is high.

```python
def _deliver_chaos(self, fields: SimFields, rng: np.random.Generator) -> None:
    for block_id in self.sensory_blocks:
        currents = rng.uniform(0, 20.0, size=self.sensory_cluster_size)
        for neuron in range(self.sensory_cluster_size):
            fields.I_ext[block_id, neuron] = currents[neuron]
```

### Stochastic Mixing

Each feedback pulse is a coin flip — not a deterministic blend:

```python
def deliver_feedback(self, distance: int, fields: SimFields) -> None:
    p = self._mixing_ratio(distance)
    if self._rng.random() < p:
        self._deliver_ordered_pulse(fields)
    else:
        self._deliver_chaos(fields, self._rng)
```

Why stochastic: a deterministic 70/30 mix is itself a learnable pattern. The network would plateau at "I can predict this murky signal" without pressure to improve. Stochastic mixing makes the chaos component irreducible.

## IOManager Class

```python
class IOManager:
    def __init__(self, topology: Topology, io_config: IOConfig, seed: int) -> None:
        self.sensory_blocks = topology.get_plane(io_config.sensory_axis, io_config.sensory_position)
        self.motor_blocks = topology.get_plane(io_config.sensory_axis, io_config.motor_position)
        self._rng = np.random.default_rng(seed)
        # Motor state
        self.rate_up = 0.0
        self.rate_down = 0.0
        self.diff_mean = 0.0
        self.diff_var = 1.0  # init with some variance to avoid zero-division
        # ... config params

    def encode_sensory(self, target_pos: float, fields: SimFields) -> None: ...
    def update_motor_rates(self, fields: SimFields, dt: float) -> None: ...
    def decode_motor(self) -> int: ...  # returns -1, 0, or +1
    def deliver_feedback(self, distance: int, fields: SimFields) -> None: ...
```

The training loop (WC-10) calls `update_motor_rates()` every simulation step (fast — just reads spikes from motor blocks) and `decode_motor()` every game tick (uses the accumulated rates to make a discrete decision).

## Tests

### Plane Selection

- For `ndim=3, grid_size=4, sensory_axis=0, sensory_position=0`: sensory plane has 16 blocks, all with `coord[0] == 0`.
- For `ndim=2, grid_size=3, sensory_axis=0, sensory_position=0`: sensory plane has 3 blocks.
- Motor plane at position 2: all blocks have `coord[0] == 2`.
- Sensory and motor planes are disjoint (no block appears in both).

### Sensory Encoding

- Target at position 0.5: the block closest to center gets maximum current, edge blocks get less.
- Two different target positions produce different current patterns.
- Total injected current is non-zero for any valid target position.
- Only `sensory_cluster_size` neurons per block receive current (rest are 0).

### Motor Decoding

- Rate trace decays correctly when no spikes arrive: `rate *= exp(-dt/tau_motor)` per step.
- Inject spikes into "up" population only → `decode_motor()` returns +1 after enough steps.
- Inject equal spikes into both → `decode_motor()` returns 0 (dead zone).
- Adaptive threshold widens when variance increases, narrows when it decreases.

### Feedback

- `distance=0` → `deliver_feedback` always delivers ordered pulse (p=1.0).
- `distance=100` → always delivers chaos (p=0.0).
- `distance=1` → over many calls, ~70% ordered, ~30% chaotic (statistical test, allow tolerance).
- Ordered pulse: all sensory neurons get the same amplitude.
- Chaotic pulse: sensory neurons get varied amplitudes (check std dev > 0).
