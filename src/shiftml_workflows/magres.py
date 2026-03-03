"""Best-effort .magres writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from shiftml_workflows.io import AtomsFrame


@dataclass(frozen=True, slots=True)
class _MagresMetadata:
    cell: np.ndarray
    pbc: tuple[int, int, int]


def _format_lattice(metadata: _MagresMetadata | None) -> list[str]:
    if metadata is None:
        return []
    if metadata.cell.shape != (3, 3):
        return []
    flat = metadata.cell.reshape(-1)
    return [
        "units lattice Angstrom",
        "lattice " + " ".join(f"{float(value):.8f}" for value in flat),
        f"# pbc {metadata.pbc[0]} {metadata.pbc[1]} {metadata.pbc[2]}",
    ]


def _build_metadata_map(frames: list[AtomsFrame] | None) -> dict[tuple[str, int], _MagresMetadata]:
    if frames is None:
        return {}
    metadata_map: dict[tuple[str, int], _MagresMetadata] = {}
    for frame in frames:
        cell = np.asarray(frame.atoms.cell.array, dtype=np.float64)
        pbc_bool = np.asarray(frame.atoms.pbc, dtype=bool)
        pbc = (int(pbc_bool[0]), int(pbc_bool[1]), int(pbc_bool[2]))
        metadata_map[(frame.structure_id, frame.frame_index)] = _MagresMetadata(cell=cell, pbc=pbc)
    return metadata_map


def _magres_block(df: pd.DataFrame, metadata: _MagresMetadata | None = None) -> str:
    lines: list[str] = ["#$magres-abinitio-v1.0", "[atoms]"]
    lines.extend(_format_lattice(metadata))
    lines.append("units atom Angstrom")
    for _, row in df.sort_values(["atom_i"]).iterrows():
        atom_i = int(row["atom_i"]) + 1
        element = str(row["element"])
        lines.append(
            f"atom {element} {element}{atom_i} {atom_i} {row['x']:.8f} {row['y']:.8f} {row['z']:.8f}"
        )

    lines.append("[/atoms]")
    lines.append("[magres]")
    lines.append("units ms ppm")
    for _, row in df.sort_values(["atom_i"]).iterrows():
        iso = float(row["cs_iso"])
        if {"cs_xx", "cs_xy", "cs_xz", "cs_yy", "cs_yz", "cs_zz"}.issubset(df.columns):
            xx = float(row.get("cs_xx", iso))
            xy = float(row.get("cs_xy", 0.0))
            xz = float(row.get("cs_xz", 0.0))
            yy = float(row.get("cs_yy", iso))
            yz = float(row.get("cs_yz", 0.0))
            zz = float(row.get("cs_zz", iso))
        else:
            xx = yy = zz = iso
            xy = xz = yz = 0.0
        atom_i = int(row["atom_i"]) + 1
        element = str(row["element"])
        lines.append(
            "ms "
            f"{element} {atom_i} "
            f"{xx:.8f} {xy:.8f} {xz:.8f} "
            f"{xy:.8f} {yy:.8f} {yz:.8f} "
            f"{xz:.8f} {yz:.8f} {zz:.8f}"
        )
    lines.append("[/magres]")
    return "\n".join(lines) + "\n"


def write_magres(
    df: pd.DataFrame,
    out_path: Path,
    mode: str,
    *,
    frames: list[AtomsFrame] | None = None,
) -> list[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if mode == "none":
        return written

    grouped = df.groupby(["structure_id", "frame"], sort=True)
    metadata_by_frame = _build_metadata_map(frames)
    if mode == "per-frame":
        magres_dir = out_path / "magres"
        magres_dir.mkdir(parents=True, exist_ok=True)
        for (structure_id, frame), group in grouped:
            path = magres_dir / f"{structure_id}_f{frame}.magres"
            metadata = metadata_by_frame.get((str(structure_id), int(frame)))
            path.write_text(_magres_block(group, metadata), encoding="utf-8")
            written.append(path)
        return written

    if mode == "single":
        target = out_path / "predictions.magres"
        blocks = []
        for (structure_id, frame), group in grouped:
            metadata = metadata_by_frame.get((str(structure_id), int(frame)))
            blocks.append(_magres_block(group, metadata))
        target.write_text("\n".join(blocks), encoding="utf-8")
        return [target]

    raise ValueError(f"Unknown magres mode: {mode}")
