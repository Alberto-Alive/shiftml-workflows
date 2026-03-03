from __future__ import annotations

import pytest


@pytest.mark.integration
def test_shiftml_importable_when_integration_enabled() -> None:
    pytest.importorskip("shiftml")
