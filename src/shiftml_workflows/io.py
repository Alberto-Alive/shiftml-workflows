"""I/O helpers for structures and atomic output writes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import hashlib
import glob
import os

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read

NORMALIZATION_VERSION = "1"


@dataclass(slots=True)
class AtomsFrame:
    atoms: Atoms
    source_file: str
    frame_index: int
    structure_id: str
    normalized_cell_pbc: bool = False


class InputError(ValueError):
    """Raised for invalid or unreadable input data."""


def resolve_inputs(inputs: list[str]) -> list[str]:
    resolved: list[str] = []
    for raw in inputs:
        candidate = Path(raw)
        if candidate.exists():
            resolved.append(str(candidate))
            continue

        matches = sorted(glob.glob(raw))
        if matches:
            resolved.extend(matches)
            continue

        raise InputError(f"Input path or glob did not match anything: {raw}")
    unique_sorted = sorted(dict.fromkeys(Path(p).resolve().as_posix() for p in resolved))
    if not unique_sorted:
        raise InputError("No input files were resolved")
    return unique_sorted


def parse_frame_selector(selector: str, length: int) -> list[int]:
    if selector == ":":
        return list(range(length))

    if ":" in selector:
        parts = selector.split(":")
        if len(parts) > 3:
            raise InputError(f"Invalid frame selector: {selector}")
        values = [int(p) if p else None for p in parts]
        while len(values) < 3:
            values.append(None)
        start, stop, step = values
        frame_slice = slice(start, stop, step)
        return list(range(length))[frame_slice]

    try:
        idx = int(selector)
    except ValueError as exc:
        raise InputError(f"Invalid frame selector: {selector}") from exc

    if idx < 0:
        idx = length + idx
    if idx < 0 or idx >= length:
        raise InputError(f"Frame index {idx} out of bounds for length {length}")
    return [idx]


def normalize_cell_pbc(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, bool]:
    raw_cell = np.asarray(atoms.cell.array, dtype=np.float64)
    raw_pbc = np.asarray(atoms.pbc, dtype=bool)
    missing = bool(np.allclose(raw_cell, 0.0) and not raw_pbc.any())
    if missing:
        cell = np.zeros((3, 3), dtype=np.float64)
        pbc = np.zeros(3, dtype=np.uint8)
        return cell, pbc, True
    cell = np.asarray(raw_cell, dtype=np.float64)
    pbc = np.asarray(raw_pbc, dtype=np.uint8)
    return cell, pbc, False


def compute_structure_id(atoms: Atoms) -> tuple[str, bool]:
    numbers = np.asarray(atoms.numbers, dtype="<i4")
    positions = np.asarray(atoms.positions, dtype="<f8")
    cell, pbc, normalized = normalize_cell_pbc(atoms)
    cell = np.asarray(cell, dtype="<f8")
    pbc = np.asarray(pbc, dtype=np.uint8)

    payload = hashlib.sha256()
    payload.update(f"normalization_version={NORMALIZATION_VERSION}\n".encode("ascii"))

    for name, array in (
        ("numbers", numbers),
        ("positions", positions),
        ("cell", cell),
        ("pbc", pbc),
    ):
        contiguous = np.ascontiguousarray(array)
        payload.update(f"{name}:{contiguous.shape}:{contiguous.dtype.str}\n".encode("ascii"))
        payload.update(contiguous.tobytes(order="C"))

    return payload.hexdigest(), normalized


def load_structures(inputs: list[str], frames: str = ":") -> list[AtomsFrame]:
    files = resolve_inputs(inputs)
    out: list[AtomsFrame] = []

    for source in files:
        try:
            read_frames = read(source, index=":")
        except Exception as exc:
            raise InputError(f"Could not read structures from {source}: {exc}") from exc

        if isinstance(read_frames, Atoms):
            all_frames = [read_frames]
        else:
            all_frames = list(read_frames)

        if not all_frames:
            raise InputError(f"Input file has no frames: {source}")

        selected_indices = parse_frame_selector(frames, len(all_frames))
        for idx in selected_indices:
            atoms = all_frames[idx]
            structure_id, normalized = compute_structure_id(atoms)
            out.append(
                AtomsFrame(
                    atoms=atoms,
                    source_file=source,
                    frame_index=idx,
                    structure_id=structure_id,
                    normalized_cell_pbc=normalized,
                )
            )

    if not out:
        raise InputError("No frames were selected from provided inputs")
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def atomic_write_dataframe(path: Path, df: pd.DataFrame, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")

    if output_format == "parquet":
        df.to_parquet(tmp_path, index=False)
    elif output_format == "csv":
        df.to_csv(tmp_path, index=False)
    else:
        raise ValueError(f"Unsupported output format for atomic write: {output_format}")

    os.replace(tmp_path, path)


def count_atoms(frames: Iterable[AtomsFrame]) -> int:
    return int(sum(len(frame.atoms) for frame in frames))
