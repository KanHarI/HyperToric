"""I/O Manager: sensory encoding, motor decoding, and feedback delivery."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hypertoric.config import IOConfig
    from hypertoric.fields import SimFields
    from hypertoric.topology import Topology


class IOManager:
    """Maps between external task state and the network's sensory/motor planes.

    Encodes target positions as current patterns on sensory blocks, decodes
    motor spikes into discrete cursor movements, and delivers order/chaos
    feedback.
    """

    def __init__(
        self,
        topology: Topology,
        io_config: IOConfig,
        k: int,
        seed: int,
    ) -> None:
        self._cfg = io_config
        self._k = k
        self._rng = np.random.default_rng(seed)

        self.sensory_blocks = topology.get_plane(
            io_config.sensory_axis, io_config.sensory_position
        )
        self.motor_blocks = topology.get_plane(
            io_config.sensory_axis, io_config.motor_position
        )

        # Motor rate traces
        self.rate_up = 0.0
        self.rate_down = 0.0

        # Adaptive threshold stats
        self.diff_mean = 0.0
        self.diff_var = 1.0

        # Precompute preferred positions for sensory blocks (evenly spaced on [0,1))
        n_sensory = len(self.sensory_blocks)
        self._preferred = np.array(
            [i / n_sensory for i in range(n_sensory)], dtype=np.float64
        )

    def encode_sensory(self, target_pos: float, fields: SimFields) -> None:
        """Inject Gaussian place-code currents into sensory blocks."""
        n_sensory = len(self.sensory_blocks)
        sigma = 1.0 / n_sensory

        i_ext = fields.I_ext.to_numpy()

        for idx, block in enumerate(self.sensory_blocks):
            raw_dist = abs(target_pos - self._preferred[idx])
            dist = min(raw_dist, 1.0 - raw_dist)
            amplitude = self._cfg.base_current * math.exp(-(dist**2) / (2.0 * sigma**2))
            cluster = min(self._cfg.sensory_cluster_size, self._k)
            i_ext[block, :cluster] = amplitude

        fields.I_ext.from_numpy(i_ext)

    def update_motor_rates(self, fields: SimFields, dt: float) -> None:
        """Update exponentially-decayed motor rate traces from spike counts."""
        spikes = fields.spikes.to_numpy()
        half_k = self._k // 2

        spike_count_up = 0
        spike_count_down = 0
        for block in self.motor_blocks:
            spike_count_up += int(np.sum(spikes[block, :half_k]))
            spike_count_down += int(np.sum(spikes[block, half_k:]))

        decay = math.exp(-dt / self._cfg.tau_motor)
        self.rate_up = self.rate_up * decay + spike_count_up
        self.rate_down = self.rate_down * decay + spike_count_down

    def decode_motor(self) -> int:
        """Decode motor rates into a discrete movement: +1, -1, or 0."""
        diff = self.rate_up - self.rate_down
        m = self._cfg.momentum
        self.diff_mean = m * self.diff_mean + (1.0 - m) * diff
        self.diff_var = m * self.diff_var + (1.0 - m) * (diff - self.diff_mean) ** 2

        threshold = self._cfg.k_threshold * math.sqrt(self.diff_var + 1e-8)
        centered = diff - self.diff_mean

        if centered > threshold:
            return 1
        if centered < -threshold:
            return -1
        return 0

    def deliver_feedback(self, distance: int, fields: SimFields) -> None:
        """Deliver order/chaos feedback pulses to sensory clusters."""
        if distance == 0:
            p = 1.0
        elif distance == 1:
            p = 0.7
        elif distance == 2:
            p = 0.3
        else:
            p = 0.0

        ordered = self._rng.random() < p
        cluster = min(self._cfg.sensory_cluster_size, self._k)
        i_ext = fields.I_ext.to_numpy()

        for block in self.sensory_blocks:
            if ordered:
                i_ext[block, :cluster] = 15.0
            else:
                i_ext[block, :cluster] = self._rng.uniform(0.0, 20.0, size=cluster)

        fields.I_ext.from_numpy(i_ext)
