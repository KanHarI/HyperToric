"""Tests for configuration dataclasses and Hydra integration."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from hypertoric.config import (
    SimConfig,
    TorusConfig,
    validate_config,
)

CONF_DIR = str(Path(__file__).resolve().parent.parent / "src" / "hypertoric" / "conf")


class TestDefaults:
    def test_simconfig_instantiates(self) -> None:
        cfg = SimConfig()
        assert cfg.torus.ndim == 3
        assert cfg.torus.grid_size == 4
        assert cfg.neuron.model == "izhikevich"
        assert cfg.seed == 42

    def test_default_values(self) -> None:
        cfg = SimConfig()
        assert cfg.torus.neurons_per_block == 256
        assert cfg.neuron.excitatory_ratio == 0.8
        assert cfg.neuron.dt == 0.5
        assert cfg.stdp.inter_mode == "rotating"
        assert cfg.backend == "gpu"


class TestValidation:
    def test_valid_default(self) -> None:
        validate_config(SimConfig())

    def test_ndim_zero(self) -> None:
        cfg = SimConfig(torus=TorusConfig(ndim=0))
        with pytest.raises(ValueError, match="ndim"):
            validate_config(cfg)

    def test_ndim_five(self) -> None:
        cfg = SimConfig(torus=TorusConfig(ndim=5))
        with pytest.raises(ValueError, match="ndim"):
            validate_config(cfg)

    def test_grid_size_zero(self) -> None:
        cfg = SimConfig(torus=TorusConfig(grid_size=0))
        with pytest.raises(ValueError, match="grid_size"):
            validate_config(cfg)

    def test_grid_size_negative(self) -> None:
        cfg = SimConfig(torus=TorusConfig(grid_size=-1))
        with pytest.raises(ValueError, match="grid_size"):
            validate_config(cfg)

    def test_excitatory_ratio_out_of_range(self) -> None:
        from hypertoric.config import NeuronConfig

        cfg = SimConfig(neuron=NeuronConfig(excitatory_ratio=1.5))
        with pytest.raises(ValueError, match="excitatory_ratio"):
            validate_config(cfg)

    def test_excitatory_ratio_zero(self) -> None:
        from hypertoric.config import NeuronConfig

        cfg = SimConfig(neuron=NeuronConfig(excitatory_ratio=0.0))
        with pytest.raises(ValueError, match="excitatory_ratio"):
            validate_config(cfg)

    def test_same_io_plane_raises(self) -> None:
        from hypertoric.config import IOConfig

        cfg = SimConfig(io=IOConfig(sensory_position=0, motor_position=0))
        with pytest.raises(ValueError, match="sensory and motor"):
            validate_config(cfg)

    def test_sensory_axis_out_of_range(self) -> None:
        from hypertoric.config import IOConfig

        cfg = SimConfig(io=IOConfig(sensory_axis=5))
        with pytest.raises(ValueError, match="sensory_axis"):
            validate_config(cfg)

    def test_stdp_stability_warning(self) -> None:
        from hypertoric.config import STDPConfig

        cfg = SimConfig(stdp=STDPConfig(a_plus=0.02, a_minus=0.01))
        with pytest.warns(UserWarning, match="a_minus"):
            validate_config(cfg)

    def test_neurons_per_block_one(self) -> None:
        cfg = SimConfig(torus=TorusConfig(neurons_per_block=1))
        with pytest.raises(ValueError, match="neurons_per_block"):
            validate_config(cfg)


class TestOmegaConfRoundTrip:
    def test_to_and_from_dictconfig(self) -> None:
        cfg = SimConfig()
        dc = OmegaConf.structured(cfg)
        # Round-trip through DictConfig container
        as_container = OmegaConf.to_container(dc)
        assert isinstance(as_container, dict)
        assert as_container["torus"]["ndim"] == 3  # type: ignore[index]
        assert as_container["seed"] == 42  # type: ignore[index]
        # to_object returns the original dataclass
        restored = OmegaConf.to_object(dc)
        assert isinstance(restored, SimConfig)
        assert restored.torus.ndim == 3
        assert restored.seed == 42


class TestHydraCompose:
    def test_compose_default(self) -> None:
        with initialize_config_dir(config_dir=CONF_DIR, version_base=None) as _cfg_init:
            cfg = compose(config_name="config")
            assert cfg.torus.ndim == 3
            assert cfg.torus.grid_size == 4
            assert cfg.seed == 42

    def test_compose_small_2d(self) -> None:
        with initialize_config_dir(config_dir=CONF_DIR, version_base=None) as _cfg_init:
            cfg = compose(config_name="config", overrides=["torus=small_2d"])
            assert cfg.torus.ndim == 2
            assert cfg.torus.grid_size == 4
            assert cfg.torus.neurons_per_block == 64

    def test_compose_override(self) -> None:
        with initialize_config_dir(config_dir=CONF_DIR, version_base=None) as _cfg_init:
            cfg = compose(config_name="config", overrides=["torus.grid_size=8"])
            assert cfg.torus.grid_size == 8

    def test_compose_stdp_all(self) -> None:
        with initialize_config_dir(config_dir=CONF_DIR, version_base=None) as _cfg_init:
            cfg = compose(config_name="config", overrides=["stdp=all"])
            assert cfg.stdp.inter_mode == "all"
