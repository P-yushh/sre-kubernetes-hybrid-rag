"""Shared pytest configuration."""

from collections.abc import Generator

import pytest


@pytest.fixture
def anyio_backend() -> Generator[str, None, None]:
    """Run async API tests on the asyncio backend used by the service."""

    yield "asyncio"
