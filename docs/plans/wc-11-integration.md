# WC-11: Integration Tests + Inversion

**Branch:** `wc-11-integration`
**Dependencies:** WC-9 (I/O manager), WC-10 (task + training loop)

## Goal

End-to-end tests that verify the full system works together — from config to simulation to learning. The crown jewel is the "inversion test": a tiny network learns a NOT gate through STDP and order/chaos feedback alone. If this passes, the core hypothesis works.

## Files

| File | Action | Notes |
|------|--------|-------|
| `tests/test_integration.py` | Create | Smoke tests, NaN checks |
| `tests/test_inversion.py` | Create | NOT gate convergence test |

## Smoke Test

### Purpose

Verify the full pipeline runs without crashing on a small config. Not a learning test — just mechanical correctness.

### Setup

```python
@pytest.mark.integration
def test_smoke_small_torus():
    config = SimConfig(
        torus=TorusConfig(ndim=2, grid_size=2, neurons_per_block=16),
        neuron=NeuronConfig(dt=0.5),
        backend="cpu",
    )
    sim = Simulator(config)
    io_mgr = IOManager(sim.topology, config.io, config.seed)
    task = TargetTracking1D(config.training)

    for tick in range(20):
        io_mgr.encode_sensory(task.get_target(), sim.fields)
        for _ in range(50):
            io_mgr.update_motor_rates(sim.fields, config.neuron.dt)
            sim.step()
        action = io_mgr.decode_motor()
        distance = task.step(action)
        io_mgr.deliver_feedback(int(distance), sim.fields)
```

### Assertions

- No exceptions raised.
- After 1000 simulation steps, check all field values are finite (no NaN, no Inf):
  ```python
  v_arr = sim.fields.v.to_numpy()
  assert np.all(np.isfinite(v_arr))
  ```
- Membrane potentials stay bounded: all `v <= 30.0` (anything above 30 should have triggered a spike reset).
- At least some spikes occurred (the sensory input should trigger activity).

### Parametrize Over Configs

Run the smoke test with several configs to catch ndim-specific bugs:

| Config | ndim | grid_size | K | Why |
|--------|------|-----------|---|-----|
| Minimal 2D | 2 | 2 | 8 | Smallest possible torus (4 blocks, 4 neurons per block) |
| Small 3D | 3 | 2 | 16 | Adds the third axis — catches ndim=3 specific issues |
| Tiny 4D | 4 | 2 | 8 | 16 blocks, 8 directions — tests 4D neighbor computation |

Each should complete in <5 seconds on CPU.

## NaN Provenance Test

### Purpose

If a NaN appears in any field, identify WHERE it originates. NaN is infectious — once one field has NaN, it spreads everywhere in the next step, making the source impossible to find.

### Approach

Run the simulation step-by-step, checking for NaN after each kernel call:

```python
@pytest.mark.integration
def test_nan_provenance():
    # Setup with a config known to be tricky (e.g., high excitatory ratio)
    sim = Simulator(config)
    io_mgr.encode_sensory(0.5, sim.fields)

    for step in range(500):
        sim.kernels.neuron_update(...)
        assert_no_nan(sim.fields.v, f"v after neuron_update, step {step}")
        assert_no_nan(sim.fields.u, f"u after neuron_update, step {step}")

        sim.kernels.spike_propagate(...)
        assert_no_nan(sim.fields.I_syn, f"I_syn after propagate, step {step}")

        # ... etc for each kernel
```

This is slow (GPU→CPU transfer after every kernel), so it's marked `@pytest.mark.slow`. It's a diagnostic tool, not a regular CI test.

## The Inversion Test

### What It Tests

The fundamental question: can the network learn an input-output mapping through STDP + order/chaos feedback alone, with no backpropagation, no reward signal, no gradient?

The simplest possible mapping is inversion (NOT gate):
- Input pattern A → expect output pattern B
- Input pattern B → expect output pattern A

### Why NOT Gate Specifically

1. **Non-trivial routing.** The sensory and motor populations are separated by at least one hop. The network must build a relay pathway, then INVERT the signal along the way.
2. **Can't be solved by default wiring.** Random initial weights produce random output. The network must reorganize.
3. **Verifiable.** Binary classification with 60% accuracy is clearly above chance (50%).
4. **Fast.** Only 2 input patterns, small network, should converge in <30s on CPU.

### Setup

```python
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.timeout(30)
def test_inversion_convergence():
    config = SimConfig(
        torus=TorusConfig(ndim=2, grid_size=2, neurons_per_block=16),
        neuron=NeuronConfig(dt=0.5),
        stdp=STDPConfig(inter_mode="all"),  # all directions for faster learning
        backend="cpu",
        seed=42,
    )
```

**Network structure (2×2 2D torus, 4 blocks):**

```
(0,1) ---- (1,1)
  |          |
(0,0) ---- (1,0)
```

- **Sensory block:** `(0,0)` — flat index 0
- **Motor block:** `(1,0)` — flat index 1 (one hop from sensory along axis 0)
- **Processing blocks:** `(0,1)` and `(1,1)` — provide indirect pathways

### Input Encoding

Two input patterns, each targeting half the sensory block's neurons:

```python
PATTERN_A = slice(0, 8)    # neurons 0-7 in sensory block
PATTERN_B = slice(8, 16)   # neurons 8-15 in sensory block
```

**Input 0:** inject strong current (15.0) into neurons 0-7 of the sensory block.
**Input 1:** inject strong current (15.0) into neurons 8-15.

### Expected Output

**Output:** differential spike rate in motor block.
- "Output 0" → neurons 0-7 fire more than neurons 8-15
- "Output 1" → neurons 8-15 fire more than neurons 0-7

**NOT gate:**
- Input 0 → expect Output 1
- Input 1 → expect Output 0

### Training Protocol

```python
correct_count = 0
total_count = 0

for epoch in range(max_epochs):
    for input_val in [0, 1]:
        # Present input for presentation_steps
        for t in range(presentation_steps):
            # Set I_ext for sensory block
            if input_val == 0:
                set_current(sim.fields, sensory_block, PATTERN_A, 15.0)
            else:
                set_current(sim.fields, sensory_block, PATTERN_B, 15.0)

            io_mgr.update_motor_rates(sim.fields, config.neuron.dt)
            sim.step()

        # Read motor output
        output = io_mgr.decode_motor()  # simplified: just check rate_up vs rate_down
        expected = 1 - input_val  # NOT gate

        # Score
        motor_spikes_0_7 = count_spikes(sim.fields, motor_block, slice(0, 8))
        motor_spikes_8_15 = count_spikes(sim.fields, motor_block, slice(8, 16))
        actual_output = 0 if motor_spikes_0_7 > motor_spikes_8_15 else 1
        is_correct = (actual_output == expected)

        correct_count += int(is_correct)
        total_count += 1

        # Deliver feedback
        distance = 0 if is_correct else 2
        io_mgr.deliver_feedback(distance, sim.fields)

    # Check convergence
    if total_count >= 20:
        accuracy = correct_count / total_count
        if accuracy >= 0.6:
            break

assert correct_count / total_count >= 0.6, (
    f"Inversion test failed: accuracy {correct_count/total_count:.1%} < 60%"
)
```

### Why 60% and Not Higher

- **50% is chance** for a binary classification.
- **60% shows learning** — the network is doing better than random, meaning STDP + feedback created some useful pathway structure.
- **Higher thresholds are risky** on CPU with a tiny network. The 16-neuron blocks have limited capacity, and CPU backend has no parallelism to explore multiple configurations simultaneously.
- If the test consistently hits >80%, raise the bar. Start conservative.

### Presentation Duration

`presentation_steps = 500` (250ms at dt=0.5ms). This is enough for:
1. Sensory current to drive spikes in the sensory block (~5ms latency).
2. Spikes to propagate through processing blocks to motor (~10-20ms, 1-2 hops).
3. Motor population to accumulate enough spikes for rate estimation (~50-100ms).
4. Several STDP updates to modify weights (~100+ updates).

If convergence is too slow, increase `presentation_steps`. If the test is too slow, decrease it (but not below ~200).

### Seeds and Flakiness

The test depends on random initialization. It should pass for MOST seeds but might fail for some (unlucky weight initialization that traps the network in a local minimum).

**Mitigation:**
1. Fix the seed in the test (seed=42). If this seed fails, try others and pick one that works.
2. If no single seed works reliably, the implementation has a bug — don't paper over it with retry logic.
3. Mark as `@pytest.mark.slow` so it doesn't run on every commit, only in CI's integration job.

### Timeout

30 seconds on CPU. If it hasn't converged by then, it won't. The timeout prevents CI from hanging on a divergent network.

### Debugging a Failing Inversion Test

If the test fails, check in this order:

1. **Are there spikes at all?** If total spike count across the network is 0, the input current is too weak or the neuron parameters are wrong.
2. **Do spikes reach the motor block?** If sensory block spikes but motor block doesn't, inter-block propagation or weight initialization is broken.
3. **Is STDP changing weights?** Compare W_intra/W_inter before and after training. If unchanged, traces aren't updating or the STDP kernel has a bug.
4. **Is feedback working?** Compare network activity under ordered vs chaotic feedback. Ordered should produce more coherent spike patterns.
5. **Is the motor decoder working?** Check raw spike counts in motor populations. If one population always dominates regardless of input, the decoder has a bias bug.

## Experiment Docs

The integration tests will likely reveal tuning needs. Create stub experiment docs:

### docs/experiments/perf/neighbor-lookup-vs-inline.md
Document the decision to use inline modular arithmetic vs a precomputed lookup table, with benchmarks once both are implemented.

### docs/experiments/motor-proportional-control.md
Experiment: instead of discrete ±1 movement, use proportional control where cursor movement magnitude depends on signal strength. Hypothesis: faster convergence for large distances but potential instability near target.

### docs/experiments/perf/motor-sampling-params.md
Systematic sweep of `tau_motor`, `k`, `momentum`, and game tick rate. Record convergence speed and final accuracy for each combination.
