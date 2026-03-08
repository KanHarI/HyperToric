from collections.abc import Iterator

import pytest
import taichi as ti


@pytest.fixture()
def ti_cpu() -> Iterator[None]:
    """Initialize Taichi with CPU backend. Use for tests that need Taichi."""
    ti.reset()
    ti.init(arch=ti.cpu)
    yield
    ti.reset()
