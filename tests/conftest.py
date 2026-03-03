from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that may require heavy dependencies/models.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-integration"):
        return

    skip_marker = pytest.mark.skip(reason="integration test; pass --run-integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)
