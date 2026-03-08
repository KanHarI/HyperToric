# Experiment: Proportional Motor Control

## Question

Should cursor movement be proportional to motor signal magnitude, rather than always ±1?

## Current Design

Discrete movement: the adaptive threshold decoder outputs -1, 0, or +1 per game tick. Cursor moves at most one position per tick regardless of signal strength.

## Proposed Alternative

```python
scale = 2.0  # tunable
max_step = 3  # prevent huge jumps
centered = diff - diff_mean
magnitude = abs(centered) / (k * sqrt(diff_var) + 1e-8)
step_size = min(int(magnitude), max_step)
cursor += sign(centered) * step_size
```

When the motor signal is strong (well above threshold), the cursor moves multiple positions. When weak (barely above threshold), it moves one position.

## Hypothesis

**Faster convergence for large distances.** When the cursor is far from the target, the network receives pure chaos (strong drive to reorganize). If it develops a strong motor signal in the correct direction, proportional control lets it close the gap quickly — several positions per tick instead of one. This should reduce the time spent in the "far from target" regime, where chaos disrupts existing pathways.

**Potential risk: instability near target.** When cursor is close to target (distance 1), overshooting is costly — the network goes from p=0.7 feedback to p=0.3. If the motor signal is noisy and the proportional control moves 2-3 positions, the cursor oscillates around the target instead of settling. The discrete ±1 approach doesn't have this problem because the maximum overshoot is 1 position.

## Test Plan

1. Run TargetTracking1D (level 1: slow step) with both decoders for 50 epochs.
2. Compare:
   - Time to first correct hit after target jump (convergence speed)
   - Average distance over last 1000 ticks (steady-state accuracy)
   - Oscillation frequency near target (count direction reversals when distance ≤ 1)
3. Sweep `scale` and `max_step` to find the sweet spot.

## Possible Outcome

Proportional control with `max_step=2` may be the best compromise — allows faster approach but caps overshoot. If oscillation is still a problem, add hysteresis: only allow proportional control when distance > 2, fall back to ±1 when close.

## Status

Not implemented. Run after baseline TargetTracking1D works with discrete control.
