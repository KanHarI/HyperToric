from collections.abc import Iterator

import pytest
import taichi as ti


@pytest.fixture(autouse=True)
def _ti_reset() -> Iterator[None]:
    ti.reset()
    ti.init(arch=ti.cpu)
    yield
    ti.reset()
