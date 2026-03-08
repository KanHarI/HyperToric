# Experiment: Motor Sampling Parameter Tuning

## Question

What are the optimal values for the motor decoder's parameters? The decoder converts noisy motor population spike trains into discrete cursor movements, and its parameters control the tradeoff between responsiveness and noise rejection.

## Parameters Under Test

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `tau_motor` | Exponential rate trace time constant (ms) | 20 | 5, 10, 20, 50, 100 |
| `k` | Dead zone width in standard deviations | 1.5 | 0.5, 1.0, 1.5, 2.0, 3.0 |
| `momentum` | Running statistics EMA momentum | 0.999 | 0.99, 0.995, 0.999, 0.9995 |
| `game_tick` | Simulation steps per game tick | 50 | 20, 50, 100, 200 |

## Expected Interactions

### tau_motor × game_tick

These interact strongly. If `tau_motor` is short (5ms) but the game tick is long (200 steps = 100ms), the rate trace decays almost to zero between spikes — the decoder only sees the most recent spikes, losing the accumulated signal. Conversely, if `tau_motor` is long (100ms) and the game tick is short (20 steps = 10ms), the trace barely updates between decisions — decisions are highly correlated and sluggish.

**Rule of thumb:** `tau_motor` should be roughly `0.5 * game_tick * dt`. This means the trace represents roughly one game tick's worth of spikes.

### k × momentum

`k` controls how many standard deviations from baseline constitute a "real" signal. `momentum` controls how fast the baseline adapts. If `momentum` is high (0.9995, slow adaptation) and the network's firing rates shift rapidly during learning, the baseline lags — causing spurious movements (apparent signal that's actually a shifted baseline). If `momentum` is low (0.99, fast adaptation), the baseline tracks the signal too closely — the dead zone never triggers because everything looks like baseline.

**Rule of thumb:** `momentum` should track changes on the timescale of structural plasticity (~1-10s). At 50ms game ticks, `momentum=0.999` gives a ~1000-tick window (~50s). This is conservative — the baseline adapts slowly, requiring a clearer signal. Start here and decrease if the network seems unresponsive.

### k values

- `k=0.5` — nearly any asymmetry triggers movement. Good for strong signals, bad with noise.
- `k=1.5` — requires a clear signal. Good default.
- `k=3.0` — very conservative, almost nothing triggers. Cursor will seem stuck. Only useful if false movements are very costly.

## Sweep Protocol

1. Fix network config: `ndim=2, grid_size=2, K=32`, seed=42.
2. Task: slow step (level 1), 20 target jumps.
3. For each parameter combination:
   - Run 5 seeds (42-46) to reduce variance.
   - Record: mean distance over last 500 ticks, time to first correct hit per jump (averaged), total cursor reversals.
4. Grid search: 5 × 5 × 4 × 4 = 400 combinations × 5 seeds = 2000 runs.
5. Each run: ~20s on CPU (2000 game ticks × 50 steps × 5ms/step overhead). Total: ~11 hours.
6. Reduce by pruning obviously bad corners first (tau_motor=5 + game_tick=200, etc).

## Metrics

| Metric | Good | Bad |
|--------|------|-----|
| Mean distance (last 500 ticks) | <1.0 | >2.0 |
| Time to first hit after jump | <50 ticks (2.5s) | >200 ticks (10s) |
| Cursor reversals per 100 ticks | <5 | >20 (oscillating) |

## Status

Not yet run. Depends on baseline system working end-to-end (after WC-11).
