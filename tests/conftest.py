from collections.abc import Iterator

import pytest
import taichi as ti


@pytest.fixture(scope="session")
def ti_cpu() -> Iterator[None]:
    """Initialize Taichi with CPU backend once per session."""
    ti.init(arch=ti.cpu)
    yield
    ti.reset()
