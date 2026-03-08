# WC-10: Task + Training Loop

**Branch:** `wc-10-task`
**Dependencies:** WC-7 (structural plasticity), WC-8 (simulator)

## Goal

Define the task abstraction, implement the 1D target tracking task with difficulty progression, and build the Hydra-powered entry point that wires everything together. After this chapter, `uv run python -m hypertoric` runs a complete training session.

## Files

| File | Action | Notes |
|------|--------|-------|
| `src/hypertoric/task.py` | Create | Task protocol + TargetTracking1D |
| `src/hypertoric/__main__.py` | Create | Hydra entry point |
| `tests/test_task.py` | Create | Task behavior, training loop smoke test |

## Task Protocol

```python
from typing import Protocol

class Task(Protocol):
    def reset(self) -> None:
        """Reset task to initial state."""
        ...

    def get_target(self) -> float:
        """Return current target position, normalized to [0, 1]."""
        ...

    def step(self, action: float) -> float:
        """Advance task by one game tick. Returns distance between cursor and target."""
        ...
```

The `action` argument is the cursor movement: -1, 0, or +1 (from IOManager.decode_motor). The returned distance is used by IOManager.deliver_feedback to determine the order/chaos mixing ratio.

Why a protocol, not an abstract base class: protocols are structural (duck typing) — any class with the right methods satisfies it, no inheritance needed. This matters when users add custom tasks.

## TargetTracking1D

### State

```python
class TargetTracking1D:
    def __init__(self, training_config: TrainingConfig, num_positions: int = 8) -> None:
        self._num_positions = num_positions
        self._target: int = 0         # target position (discrete)
        self._cursor: int = 0         # cursor position (discrete)
        self._tick: int = 0           # game ticks elapsed
        self._level: int = 0          # difficulty level
        self._rng = np.random.default_rng(42)
```

### Difficulty Levels

Each level isolates a specific learning mechanism. The task auto-advances when performance criteria are met, but can also be pinned to a specific level via config.

#### Level 0: Static Target

Target fixed at one position (e.g., position 4). Never moves. The network must learn: "this sensory pattern → move cursor to position 4 and stop."

**What this tests:** Basic sensory-motor pathway formation through STDP. If this fails, something is wrong with the pipeline (wrong signs, broken propagation, etc.).

**Advancement criterion:** Average distance < 0.5 over last 100 game ticks.

#### Level 1: Slow Step

Target jumps to a new random position every `step_interval` game ticks (default 100 ticks = 5 seconds at 50ms/tick). The network must re-adapt cursor position after each jump.

**What this tests:** Generalization — did the network learn a position-specific mapping, or did it memorize one input-output pair? If it fails here but passes level 0, the STDP learned a fixed pattern rather than a general sensory-motor map.

**Advancement criterion:** Average distance < 1.0 over last 500 game ticks.

#### Level 2: Slow Ramp

Target moves one position per second (every 20 game ticks), bouncing at boundaries. Continuous tracking.

**What this tests:** Temporal dynamics — can the network track a moving target? This requires ongoing sensory processing, not just a learned static map.

**Advancement criterion:** Average distance < 1.5 over last 1000 game ticks.

#### Level 3: Sine Wave

Target follows `pos = A * sin(2π * t / period) + offset`, discretized. Periodic, predictable trajectory.

**What this tests:** Predictive tracking — can the network learn to move the cursor BEFORE the target arrives? If it does, the network has learned temporal structure, not just reactive mapping. This would be a remarkable result.

**No advancement criterion** — this is the final level.

### Step Logic

```python
def step(self, action: float) -> float:
    # Move cursor
    movement = int(round(action))
    self._cursor = max(0, min(self._num_positions - 1, self._cursor + movement))

    # Update target based on level
    self._tick += 1
    if self._level == 0:
        pass  # target stays fixed
    elif self._level == 1:
        if self._tick % self._step_interval == 0:
            self._target = self._rng.integers(0, self._num_positions)
    elif self._level == 2:
        if self._tick % self._ramp_interval == 0:
            self._target += self._ramp_direction
            if self._target >= self._num_positions - 1 or self._target <= 0:
                self._ramp_direction *= -1
    elif self._level == 3:
        t = self._tick * self._tick_duration
        self._target = int(round(
            (self._num_positions - 1) / 2 * (1 + math.sin(2 * math.pi * t / self._sine_period))
        ))

    # Distance
    return float(abs(self._cursor - self._target))
```

**Cursor clamping:** The cursor can't go below 0 or above `num_positions - 1`. This is intentional — the boundaries provide an important learning signal. A network that always outputs "move up" will get stuck at the top, which is wrong for any target below the top. The asymmetry at boundaries breaks trivial solutions.

### Normalized Target Position for Sensory Encoding

The IOManager needs a float in [0, 1] for place coding. The task provides `get_target()`:

```python
def get_target(self) -> float:
    return self._target / (self._num_positions - 1)
```

## Entry Point: __main__.py

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    config = to_sim_config(cfg)  # Convert OmegaConf → SimConfig dataclass
    sim = Simulator(config)
    io_mgr = IOManager(sim.topology, config.io, config.seed)
    task = TargetTracking1D(config.training)

    game_tick_interval = 50  # simulation steps per game tick

    for epoch in range(config.training.max_epochs):
        task.reset()
        for tick in range(config.training.ticks_per_epoch):
            # Encode sensory input
            io_mgr.encode_sensory(task.get_target(), sim.fields)

            # Run simulation for one game tick
            for _ in range(game_tick_interval):
                io_mgr.update_motor_rates(sim.fields, config.neuron.dt)
                sim.step()

            # Decode motor output
            action = io_mgr.decode_motor()

            # Task step
            distance = task.step(action)

            # Deliver feedback
            io_mgr.deliver_feedback(int(distance), sim.fields)

            # Logging (every N ticks)
            if tick % 100 == 0:
                print(f"Epoch {epoch}, tick {tick}: target={task._target}, "
                      f"cursor={task._cursor}, distance={distance:.1f}")

if __name__ == "__main__":
    main()
```

### to_sim_config()

Converts OmegaConf's `DictConfig` (untyped, dynamic) to our typed `SimConfig` dataclass. This is where validation happens — the dataclass `__post_init__` checks constraints.

```python
def to_sim_config(cfg: DictConfig) -> SimConfig:
    return SimConfig(
        torus=TorusConfig(**cfg.torus),
        neuron=NeuronConfig(**cfg.neuron),
        stdp=STDPConfig(**cfg.stdp),
        plasticity=PlasticityConfig(**cfg.plasticity),
        io=IOConfig(**cfg.io),
        training=TrainingConfig(**cfg.training),
        seed=cfg.seed,
        backend=cfg.backend,
    )
```

### Config Path

`config_path="../../conf"` is relative to `__main__.py` at `src/hypertoric/__main__.py`. This points to `conf/` at the project root. Hydra resolves this at runtime.

### TrainingConfig Additions

```python
@dataclass
class TrainingConfig:
    task: str = "tracking_1d"
    feedback_tau: float = 100.0
    max_epochs: int = 100
    ticks_per_epoch: int = 1000
    game_tick_steps: int = 50  # sim steps per game tick
```

## Tests

### Task Behavior

**Static target:**
- Reset task. `get_target()` returns a consistent value.
- Call `step(0)` 100 times. Distance stays constant (cursor doesn't move, target doesn't move).
- Call `step(1)` enough times to reach target. Distance should reach 0.
- Continue calling `step(1)` past the target. Cursor clamps at boundary, distance increases.

**Slow step:**
- Run enough ticks for a target jump. Verify target changes.
- Verify target is within valid range after jump.

**Cursor boundaries:**
- Start at position 0. Call `step(-1)`. Cursor stays at 0 (clamped).
- Start at max position. Call `step(1)`. Cursor stays at max.

**Distance calculation:**
- Target=4, cursor=2 → distance=2.
- Target=0, cursor=7 → distance=7 (NOT 1 — the cursor axis does NOT wrap, unlike the torus).

### Training Loop Smoke Test

- Construct Simulator, IOManager, Task with a tiny config (`ndim=2, grid_size=2, K=8`).
- Run 10 game ticks (500 simulation steps total).
- No errors, no NaN, no infinite values.
- This is NOT a learning test — just verifies the wiring is correct.

### Config Override via Hydra

- Test that `python -m hypertoric torus=small_2d` launches without error (use subprocess or Hydra compose API in test).
- Test that invalid config (`torus.ndim=0`) raises a validation error.
