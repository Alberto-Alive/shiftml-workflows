from __future__ import annotations

from pathlib import Path

import pytest

from shiftml_workflows.config import ConfigError, build_predict_config


def test_strict_string_true_forces_on_warning_error(tmp_path: Path) -> None:
    cfg = build_predict_config(
        outdir=tmp_path / "out",
        cli_values={"strict": "true", "on_warning": "skip"},
    )
    assert cfg.on_warning == "error"


def test_strict_string_false_does_not_override_on_warning(tmp_path: Path) -> None:
    cfg = build_predict_config(
        outdir=tmp_path / "out",
        cli_values={"strict": "false", "on_warning": "skip"},
    )
    assert cfg.on_warning == "skip"


def test_invalid_boolean_value_in_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        build_predict_config(
            outdir=tmp_path / "out",
            cli_values={"strict": "maybe"},
        )
