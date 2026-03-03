"""Ensemble averaging utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import pandas as pd


def load_weights(path: Path) -> dict[str, float]:
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ValueError("YAML weights requested but PyYAML not installed") from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("Weights file must be an object mapping '<source>#<frame>' to weight")

    out: dict[str, float] = {}
    for key, value in payload.items():
        out[str(key)] = float(value)
    return out


def average_per_atom(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    work = df.copy()
    if weights:
        frame_keys = work["source_file"].astype(str) + "#" + work["frame"].astype(str)
        work["_weight"] = frame_keys.map(weights)
        if work["_weight"].isna().any():
            raise ValueError("Weights do not cover all source_file/frame combinations")
    else:
        work["_weight"] = 1.0

    required_cols = ["atom_i", "element", "x", "y", "z"]
    grouped = work.groupby(required_cols, dropna=False)

    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        weight_sum = group["_weight"].sum()
        if weight_sum == 0:
            continue
        rows.append(
            {
                "atom_i": int(keys[0]),
                "element": keys[1],
                "x": float(keys[2]),
                "y": float(keys[3]),
                "z": float(keys[4]),
                "cs_iso": float((group["cs_iso"] * group["_weight"]).sum() / weight_sum),
            }
        )

    return pd.DataFrame(rows).sort_values(["atom_i"]).reset_index(drop=True)
