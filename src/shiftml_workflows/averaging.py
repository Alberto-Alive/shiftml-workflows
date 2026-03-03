"""Ensemble averaging utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import numpy as np
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
    required_cols = ["source_file", "frame", "atom_i", "element", "x", "y", "z"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for averaging: {', '.join(missing)}")

    prediction_columns = [col for col in df.columns if col.startswith("cs_")]
    if "cs_iso" not in prediction_columns:
        raise ValueError("Input results must contain cs_iso column")

    work = df.copy()
    if weights:
        frame_keys = work["source_file"].astype(str) + "#" + work["frame"].astype(str)
        work["_weight"] = frame_keys.map(weights)
        if work["_weight"].isna().any():
            raise ValueError("Weights do not cover all source_file/frame combinations")
    else:
        work["_weight"] = 1.0

    grouped = work.groupby(["atom_i", "element", "x", "y", "z"], dropna=False)

    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        weight_values = pd.to_numeric(group["_weight"], errors="coerce").to_numpy(dtype=np.float64)
        weight_sum = float(weight_values.sum())
        if weight_sum == 0:
            continue
        row: dict[str, Any] = {
            "atom_i": int(keys[0]),
            "element": keys[1],
            "x": float(keys[2]),
            "y": float(keys[3]),
            "z": float(keys[4]),
        }

        for column in prediction_columns:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            if not finite.any():
                row[column] = float("nan")
                continue
            denom = float(weight_values[finite].sum())
            if denom == 0.0:
                row[column] = float("nan")
                continue
            row[column] = float(np.dot(values[finite], weight_values[finite]) / denom)
        rows.append(row)

    ordered = ["atom_i", "element", "x", "y", "z", *prediction_columns]
    return pd.DataFrame(rows, columns=ordered).sort_values(["atom_i"]).reset_index(drop=True)
