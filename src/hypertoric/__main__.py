"""Hydra entry point: wires Simulator + IOManager + Task into a training loop."""

from __future__ import annotations

from pathlib import Path

import hydra
import taichi as ti
from omegaconf import DictConfig, OmegaConf

from hypertoric.config import (
    SimConfig,
)


def to_sim_config(cfg: DictConfig) -> SimConfig:
    """Convert a Hydra DictConfig to a typed SimConfig."""
    return SimConfig(
        torus=OmegaConf.to_object(cfg.torus),  # type: ignore[arg-type]
        neuron=OmegaConf.to_object(cfg.neuron),  # type: ignore[arg-type]
        stdp=OmegaConf.to_object(cfg.stdp),  # type: ignore[arg-type]
        plasticity=OmegaConf.to_object(cfg.plasticity),  # type: ignore[arg-type]
        io=OmegaConf.to_object(cfg.io),  # type: ignore[arg-type]
        training=OmegaConf.to_object(cfg.training),  # type: ignore[arg-type]
        seed=cfg.seed,
        backend=cfg.backend,
    )


CONF_DIR = str(Path(__file__).resolve().parent.parent.parent / "conf")


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the HyperToric training loop."""
    sim_cfg = to_sim_config(cfg)

    # Initialize Taichi backend
    arch = getattr(ti, sim_cfg.backend, ti.gpu)
    ti.init(arch=arch, offline_cache=False)

    from hypertoric.io import IOManager
    from hypertoric.simulator import Simulator
    from hypertoric.task import TargetTracking1D

    sim = Simulator(sim_cfg)
    io_mgr = IOManager(
        sim.topology, sim_cfg.io, sim_cfg.torus.neurons_per_block, sim_cfg.seed
    )
    task = TargetTracking1D(sim_cfg.training, seed=sim_cfg.seed)

    dt = sim_cfg.neuron.dt
    train = sim_cfg.training

    print(
        f"Starting training: {train.max_epochs} epochs, "
        f"{train.ticks_per_epoch} ticks/epoch, "
        f"{train.game_tick_steps} sim steps/tick"
    )

    for epoch in range(train.max_epochs):
        task.reset()
        epoch_distance = 0.0

        for tick in range(train.ticks_per_epoch):
            # Encode sensory input
            target_pos = task.get_target()
            io_mgr.encode_sensory(target_pos, sim.fields)

            # Run N sim steps per game tick
            for _step in range(train.game_tick_steps):
                sim.step()
                io_mgr.update_motor_rates(sim.fields, dt)

            # Decode motor output and step task
            action = io_mgr.decode_motor()
            distance = task.step(action)
            epoch_distance += distance

            # Deliver feedback
            io_mgr.deliver_feedback(int(distance), sim.fields)

            if (tick + 1) % 100 == 0:
                avg_dist = epoch_distance / (tick + 1)
                print(
                    f"  epoch {epoch} tick {tick + 1}: "
                    f"avg_dist={avg_dist:.2f} target={target_pos:.2f}"
                )

        avg_epoch_dist = epoch_distance / train.ticks_per_epoch
        print(f"Epoch {epoch} done: avg_distance={avg_epoch_dist:.2f}")


if __name__ == "__main__":
    main()
