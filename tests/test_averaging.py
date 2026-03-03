from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from shiftml_workflows.averaging import average_per_atom, load_weights


def _input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_file": ["a.xyz", "a.xyz", "a.xyz", "a.xyz"],
            "frame": [0, 0, 1, 1],
            "atom_i": [0, 1, 0, 1],
            "element": ["H", "C", "H", "C"],
            "x": [0.0, 0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 1.0, 0.0, 1.0],
            "cs_iso": [10.0, 20.0, 14.0, 28.0],
            "cs_xx": [1.0, 2.0, 3.0, 6.0],
        }
    )


def test_average_per_atom_unweighted() -> None:
    averaged = average_per_atom(_input_df())
    assert averaged["atom_i"].tolist() == [0, 1]
    assert averaged["cs_iso"].tolist() == [12.0, 24.0]
    assert averaged["cs_xx"].tolist() == [2.0, 4.0]


def test_average_per_atom_weighted() -> None:
    weights = {"a.xyz#0": 1.0, "a.xyz#1": 3.0}
    averaged = average_per_atom(_input_df(), weights=weights)
    assert averaged["cs_iso"].tolist() == [13.0, 26.0]
    assert averaged["cs_xx"].tolist() == [2.5, 5.0]


def test_average_per_atom_missing_weight_raises() -> None:
    with pytest.raises(ValueError):
        average_per_atom(_input_df(), weights={"a.xyz#0": 1.0})


def test_load_weights_json(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps({"a.xyz#0": 0.25}), encoding="utf-8")
    loaded = load_weights(weights_path)
    assert loaded == {"a.xyz#0": 0.25}
