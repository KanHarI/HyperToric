"""Typed configuration dataclasses with Hydra structured config registration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class TorusConfig:
    """N-dimensional torus grid parameters."""

    ndim: int = 3
    grid_size: int = 4
    neurons_per_block: int = 256


@dataclass
class NeuronConfig:
    """Neuron model parameters."""

    model: str = "izhikevich"
    excitatory_ratio: float = 0.8
    dt: float = 0.5  # ms
    tau_syn: float = 5.0  # synaptic decay time constant (ms)


@dataclass
class STDPConfig:
    """Spike-timing-dependent plasticity parameters."""

    a_plus: float = 0.01
    a_minus: float = 0.012
    tau_pre: float = 20.0  # ms
    tau_post: float = 20.0  # ms
    inter_mode: str = "rotating"  # "rotating" or "all"


@dataclass
class PlasticityConfig:
    """Structural plasticity parameters."""

    interval: int = 1000  # timesteps between intra-block plasticity updates
    inter_interval: int = 10000  # inter-block structural plasticity interval
    calcium_tau: float = 50.0
    calcium_threshold_low: float = 0.2
    calcium_threshold_high: float = 0.8
    w_max: float = 1.0
    weight_threshold: float = 0.001  # prune synapses below this |weight|
    init_weight: float = 0.01  # weight for newly grown synapses


@dataclass
class IOConfig:
    """I/O layout for sensory and motor planes."""

    sensory_axis: int = 0
    sensory_position: int = 0
    motor_axis: int = 0
    motor_position: int = field(default=3)
    sensory_cluster_size: int = 16
    motor_cluster_size: int = 16
    tau_motor: float = 20.0  # motor rate smoothing (ms)
    k_threshold: float = 1.5  # dead zone width in std devs
    momentum: float = 0.999  # adaptive threshold momentum
    base_current: float = 15.0  # sensory encoding base current


@dataclass
class TrainingConfig:
    """Task and training loop parameters."""

    task: str = "static_target"
    feedback_tau: float = 100.0
    num_steps: int = 10000


@dataclass
class SimConfig:
    """Top-level simulation configuration."""

    torus: TorusConfig = field(default_factory=TorusConfig)
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    stdp: STDPConfig = field(default_factory=STDPConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    io: IOConfig = field(default_factory=IOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    backend: str = "gpu"


def validate_config(cfg: SimConfig) -> None:
    """Validate config constraints. Raises ValueError on hard errors."""
    # Torus constraints
    if cfg.torus.ndim < 2 or cfg.torus.ndim > 4:
        msg = f"ndim must be in {{2, 3, 4}}, got {cfg.torus.ndim}"
        raise ValueError(msg)
    if cfg.torus.grid_size < 2:
        msg = f"grid_size must be >= 2, got {cfg.torus.grid_size}"
        raise ValueError(msg)
    if cfg.torus.neurons_per_block < 2:
        msg = f"neurons_per_block must be >= 2, got {cfg.torus.neurons_per_block}"
        raise ValueError(msg)

    # Neuron constraints
    if cfg.neuron.excitatory_ratio <= 0.0 or cfg.neuron.excitatory_ratio >= 1.0:
        msg = f"excitatory_ratio must be in (0, 1), got {cfg.neuron.excitatory_ratio}"
        raise ValueError(msg)

    # STDP constraints
    if cfg.stdp.inter_mode not in {"rotating", "all"}:
        msg = f"inter_mode must be 'rotating' or 'all', got {cfg.stdp.inter_mode!r}"
        raise ValueError(msg)

    # STDP stability warning
    if cfg.stdp.a_minus <= cfg.stdp.a_plus:
        warnings.warn(
            f"a_minus ({cfg.stdp.a_minus}) should be > a_plus ({cfg.stdp.a_plus}) "
            "for STDP stability under random spiking",
            UserWarning,
            stacklevel=2,
        )

    # I/O numeric constraints
    if cfg.io.tau_motor <= 0.0:
        msg = f"tau_motor must be > 0, got {cfg.io.tau_motor}"
        raise ValueError(msg)
    if cfg.io.k_threshold < 0.0:
        msg = f"k_threshold must be >= 0, got {cfg.io.k_threshold}"
        raise ValueError(msg)
    if not (0.0 <= cfg.io.momentum < 1.0):
        msg = f"momentum must be in [0, 1), got {cfg.io.momentum}"
        raise ValueError(msg)
    if cfg.io.base_current < 0.0:
        msg = f"base_current must be >= 0, got {cfg.io.base_current}"
        raise ValueError(msg)

    # I/O constraints
    if cfg.io.sensory_axis >= cfg.torus.ndim:
        msg = f"sensory_axis ({cfg.io.sensory_axis}) must be < ndim ({cfg.torus.ndim})"
        raise ValueError(msg)
    if cfg.io.motor_axis >= cfg.torus.ndim:
        msg = f"motor_axis ({cfg.io.motor_axis}) must be < ndim ({cfg.torus.ndim})"
        raise ValueError(msg)
    if (
        cfg.io.sensory_axis == cfg.io.motor_axis
        and cfg.io.sensory_position == cfg.io.motor_position
    ):
        msg = (
            "sensory and motor planes must differ: "
            f"both at axis={cfg.io.sensory_axis}, position={cfg.io.sensory_position}"
        )
        raise ValueError(msg)

    # Neuron tau_syn
    if cfg.neuron.tau_syn <= 0:
        msg = f"tau_syn must be > 0, got {cfg.neuron.tau_syn}"
        raise ValueError(msg)

    # Plasticity constraints
    if cfg.plasticity.interval <= 0:
        msg = f"interval must be > 0, got {cfg.plasticity.interval}"
        raise ValueError(msg)
    if cfg.plasticity.weight_threshold <= 0:
        msg = f"weight_threshold must be > 0, got {cfg.plasticity.weight_threshold}"
        raise ValueError(msg)
    if (
        cfg.plasticity.init_weight <= 0
        or cfg.plasticity.init_weight > cfg.plasticity.w_max
    ):
        msg = (
            f"init_weight must be in (0, w_max], got {cfg.plasticity.init_weight} "
            f"(w_max={cfg.plasticity.w_max})"
        )
        raise ValueError(msg)
    if cfg.plasticity.inter_interval < cfg.plasticity.interval:
        msg = (
            f"inter_interval ({cfg.plasticity.inter_interval}) must be >= "
            f"interval ({cfg.plasticity.interval})"
        )
        raise ValueError(msg)


def _register_configs() -> None:
    """Register structured configs with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="config", node=SimConfig)
    cs.store(group="torus", name="default_3d", node=TorusConfig())
    cs.store(
        group="torus",
        name="small_2d",
        node=TorusConfig(ndim=2, grid_size=4, neurons_per_block=64),
    )
    cs.store(
        group="torus",
        name="large_4d",
        node=TorusConfig(ndim=4, grid_size=3, neurons_per_block=128),
    )
    cs.store(group="neuron", name="default", node=NeuronConfig())
    cs.store(group="stdp", name="rotating", node=STDPConfig(inter_mode="rotating"))
    cs.store(group="stdp", name="all", node=STDPConfig(inter_mode="all"))
    cs.store(group="plasticity", name="default", node=PlasticityConfig())
    cs.store(group="io", name="tracking_1d", node=IOConfig())
    cs.store(group="training", name="static_target", node=TrainingConfig())
    cs.store(
        group="training",
        name="slow_step",
        node=TrainingConfig(task="slow_step", num_steps=50000),
    )


_register_configs()
