#!/usr/bin/env python3
"""Generate CPU reference trajectories for numerical cross-backend verification.

Run once on CPU to produce tests/reference_trajectories.npz, then use
test_numerical_reference.py to verify other backends match.

Three types of reference data:
  1. Single-step: one kernel step from known initial state (tight tolerance)
  2. Short-horizon: first 20 steps before chaotic divergence (moderate tolerance)
  3. Statistical: 500-step spike counts and mean voltages (loose tolerance)
"""

import numpy as np
import taichi as ti

from hypertoric.kernels.neuron_models.izhikevich import make_izhikevich_update

PARAMS = {
    "RS": (0.02, 0.2, -65.0, 8.0),
    "IB": (0.02, 0.2, -55.0, 4.0),
    "CH": (0.02, 0.2, -50.0, 2.0),
    "FS": (0.1, 0.2, -65.0, 2.0),
}

DT = 0.5
B, K = 1, 1

SHORT_HORIZON = 20
LONG_RUN = 500


def _alloc(v0: float, u0: float, current: float, ntype: str) -> tuple:
    a, bv, c, d = PARAMS[ntype]
    v = ti.field(dtype=ti.f32, shape=(B, K))
    u = ti.field(dtype=ti.f32, shape=(B, K))
    spikes = ti.field(dtype=ti.i32, shape=(B, K))
    i_ext = ti.field(dtype=ti.f32, shape=(B, K))
    pa = ti.field(dtype=ti.f32, shape=(B, K))
    pb = ti.field(dtype=ti.f32, shape=(B, K))
    pc = ti.field(dtype=ti.f32, shape=(B, K))
    pd = ti.field(dtype=ti.f32, shape=(B, K))
    v.from_numpy(np.full((B, K), v0, dtype=np.float32))
    u.from_numpy(np.full((B, K), u0, dtype=np.float32))
    spikes.from_numpy(np.zeros((B, K), dtype=np.int32))
    i_ext.from_numpy(np.full((B, K), current, dtype=np.float32))
    pa.from_numpy(np.full((B, K), a, dtype=np.float32))
    pb.from_numpy(np.full((B, K), bv, dtype=np.float32))
    pc.from_numpy(np.full((B, K), c, dtype=np.float32))
    pd.from_numpy(np.full((B, K), d, dtype=np.float32))
    return v, u, spikes, i_ext, pa, pb, pc, pd


def main() -> None:
    import pathlib

    ti.init(arch=ti.cpu, offline_cache=False)

    results: dict[str, np.ndarray] = {}
    kernel = make_izhikevich_update(B, K)

    for ntype in PARAMS:
        a, bv, c, d = PARAMS[ntype]
        v0, u0 = -65.0, bv * -65.0

        for current in [10.0, 15.0]:
            key = f"{ntype}_I{current:.0f}"

            # --- Single step ---
            fields = _alloc(v0, u0, current, ntype)
            v, u, spikes, i_ext, pa, pb, pc, pd = fields
            kernel(v, u, spikes, i_ext, pa, pb, pc, pd, DT)
            results[f"{key}_single_v"] = np.array(
                [v.to_numpy()[0, 0]], dtype=np.float32
            )
            results[f"{key}_single_u"] = np.array(
                [u.to_numpy()[0, 0]], dtype=np.float32
            )

            # --- Short horizon (20 steps) ---
            fields = _alloc(v0, u0, current, ntype)
            v, u, spikes, i_ext, pa, pb, pc, pd = fields
            v_trace = np.zeros(SHORT_HORIZON, dtype=np.float32)
            u_trace = np.zeros(SHORT_HORIZON, dtype=np.float32)
            for step in range(SHORT_HORIZON):
                kernel(v, u, spikes, i_ext, pa, pb, pc, pd, DT)
                v_trace[step] = v.to_numpy()[0, 0]
                u_trace[step] = u.to_numpy()[0, 0]
            results[f"{key}_short_v"] = v_trace
            results[f"{key}_short_u"] = u_trace

            # --- Long run statistics ---
            fields = _alloc(v0, u0, current, ntype)
            v, u, spikes, i_ext, pa, pb, pc, pd = fields
            all_v = np.zeros(LONG_RUN, dtype=np.float32)
            total_spikes = 0
            for step in range(LONG_RUN):
                kernel(v, u, spikes, i_ext, pa, pb, pc, pd, DT)
                all_v[step] = v.to_numpy()[0, 0]
                total_spikes += int(spikes.to_numpy()[0, 0])
            results[f"{key}_spike_count"] = np.array([total_spikes], dtype=np.int32)
            results[f"{key}_mean_v"] = np.array([all_v.mean()], dtype=np.float32)
            results[f"{key}_std_v"] = np.array([all_v.std()], dtype=np.float32)

            print(
                f"  {key}: single_v={results[f'{key}_single_v'][0]:.6f}, "
                f"spikes={total_spikes}, mean_v={all_v.mean():.2f}"
            )

    out = pathlib.Path(__file__).parent / "reference_trajectories.npz"
    np.savez(out, **results)
    print(f"\nSaved {len(results)} arrays to {out}")

    ti.reset()


if __name__ == "__main__":
    main()
