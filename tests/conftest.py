from collections.abc import Iterator

import pytest
import taichi as ti


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--ti-arch",
        default="cpu",
        choices=["cpu", "metal", "vulkan", "cuda"],
        help="Taichi backend architecture",
    )


@pytest.fixture(scope="session")
def ti_cpu(request: pytest.FixtureRequest) -> Iterator[None]:
    """Initialize Taichi with the selected backend once per session."""
    arch_name = request.config.getoption("--ti-arch")
    arch = getattr(ti, arch_name)
    ti.init(arch=arch, offline_cache=False)
    yield
    ti.reset()
