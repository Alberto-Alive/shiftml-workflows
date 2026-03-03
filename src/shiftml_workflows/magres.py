"""Best-effort .magres writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _magres_block(df: pd.DataFrame) -> str:
    lines: list[str] = ["#$magres-abinitio-v1.0", "[atoms]"]
    for _, row in df.sort_values(["atom_i"]).iterrows():
        lines.append(f"atom {row['element']} {int(row['atom_i']) + 1} {row['x']:.8f} {row['y']:.8f} {row['z']:.8f}")

    lines.append("[/atoms]")
    lines.append("[magres]")
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
        lines.append(
            "ms "
            f"{int(row['atom_i']) + 1} "
            f"{xx:.8f} {xy:.8f} {xz:.8f} "
            f"{xy:.8f} {yy:.8f} {yz:.8f} "
            f"{xz:.8f} {yz:.8f} {zz:.8f}"
        )
    lines.append("[/magres]")
    return "\n".join(lines) + "\n"


def write_magres(df: pd.DataFrame, out_path: Path, mode: str) -> list[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if mode == "none":
        return written

    grouped = df.groupby(["structure_id", "frame"], sort=True)
    if mode == "per-frame":
        magres_dir = out_path / "magres"
        magres_dir.mkdir(parents=True, exist_ok=True)
        for (structure_id, frame), group in grouped:
            path = magres_dir / f"{structure_id}_f{frame}.magres"
            path.write_text(_magres_block(group), encoding="utf-8")
            written.append(path)
        return written

    if mode == "single":
        target = out_path / "predictions.magres"
        blocks = [_magres_block(group) for _, group in grouped]
        target.write_text("\n".join(blocks), encoding="utf-8")
        return [target]

    raise ValueError(f"Unknown magres mode: {mode}")
