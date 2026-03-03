"""Output schema conversion and validation."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from shiftml_workflows.backends.base import PredictionResult
from shiftml_workflows.io import AtomsFrame

SCHEMA_VERSION = "1"

_BASE_COLUMNS = [
    "structure_id",
    "source_file",
    "frame",
    "atom_i",
    "element",
    "x",
    "y",
    "z",
    "cs_iso",
]

_TENSOR_COLUMNS = ["cs_xx", "cs_xy", "cs_xz", "cs_yy", "cs_yz", "cs_zz"]
_TENSOR_UNC_COLUMNS = [
    "cs_xx_uncertainty",
    "cs_xy_uncertainty",
    "cs_xz_uncertainty",
    "cs_yy_uncertainty",
    "cs_yz_uncertainty",
    "cs_zz_uncertainty",
]


def required_columns(property_mode: str, committee: bool) -> list[str]:
    columns = list(_BASE_COLUMNS)
    if committee:
        columns.append("cs_iso_uncertainty")
    if property_mode in {"tensor", "both"}:
        columns.extend(_TENSOR_COLUMNS)
        if committee:
            columns.extend(_TENSOR_UNC_COLUMNS)
    return columns


def _extract_tensor_components(tensor: np.ndarray, atom_index: int) -> dict[str, float]:
    return {
        "cs_xx": float(tensor[atom_index, 0, 0]),
        "cs_xy": float(tensor[atom_index, 0, 1]),
        "cs_xz": float(tensor[atom_index, 0, 2]),
        "cs_yy": float(tensor[atom_index, 1, 1]),
        "cs_yz": float(tensor[atom_index, 1, 2]),
        "cs_zz": float(tensor[atom_index, 2, 2]),
    }


def _extract_tensor_uncertainty(tensor_unc: np.ndarray, atom_index: int) -> dict[str, float]:
    return {
        "cs_xx_uncertainty": float(tensor_unc[atom_index, 0, 0]),
        "cs_xy_uncertainty": float(tensor_unc[atom_index, 0, 1]),
        "cs_xz_uncertainty": float(tensor_unc[atom_index, 0, 2]),
        "cs_yy_uncertainty": float(tensor_unc[atom_index, 1, 1]),
        "cs_yz_uncertainty": float(tensor_unc[atom_index, 1, 2]),
        "cs_zz_uncertainty": float(tensor_unc[atom_index, 2, 2]),
    }


def to_dataframe(
    frames: Iterable[AtomsFrame],
    prediction_result: PredictionResult,
    *,
    property_mode: str,
    committee: bool,
    frame_cache_keys: Optional[list[str]] = None,
) -> pd.DataFrame:
    frame_list = list(frames)
    if len(frame_list) != len(prediction_result.frames):
        raise ValueError("Prediction result count does not match frame count")

    rows: list[dict[str, object]] = []
    for idx, (frame, pred) in enumerate(zip(frame_list, prediction_result.frames)):
        positions = np.asarray(frame.atoms.positions, dtype=np.float64)
        symbols = frame.atoms.get_chemical_symbols()

        for atom_i, symbol in enumerate(symbols):
            row: dict[str, object] = {
                "structure_id": frame.structure_id,
                "source_file": frame.source_file,
                "frame": frame.frame_index,
                "atom_i": atom_i,
                "element": symbol,
                "x": float(positions[atom_i, 0]),
                "y": float(positions[atom_i, 1]),
                "z": float(positions[atom_i, 2]),
                "cs_iso": float(pred.cs_iso[atom_i]),
            }

            if frame_cache_keys is not None:
                row["frame_cache_key"] = frame_cache_keys[idx]

            if committee and pred.cs_iso_uncertainty is not None:
                row["cs_iso_uncertainty"] = float(pred.cs_iso_uncertainty[atom_i])

            if property_mode in {"tensor", "both"} and pred.tensor is not None:
                row.update(_extract_tensor_components(pred.tensor, atom_i))
                if committee and pred.tensor_uncertainty is not None:
                    row.update(_extract_tensor_uncertainty(pred.tensor_uncertainty, atom_i))

            rows.append(row)

    df = pd.DataFrame(rows)
    for col in required_columns(property_mode=property_mode, committee=committee):
        if col not in df.columns:
            df[col] = np.nan
    return df
