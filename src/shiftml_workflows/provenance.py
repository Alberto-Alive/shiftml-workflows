"""Provenance and run metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import importlib.metadata
import json
import platform
import subprocess
import sys

from shiftml_workflows import __version__
from shiftml_workflows.io import atomic_write_json, file_sha256
from shiftml_workflows.schema import SCHEMA_VERSION


@dataclass(slots=True)
class Timing:
    setup_time_s: float
    cache_time_s: float
    predict_time_s: float

    def as_dict(self) -> dict[str, float]:
        return {
            "setup_time_s": round(self.setup_time_s, 6),
            "cache_time_s": round(self.cache_time_s, 6),
            "predict_time_s": round(self.predict_time_s, 6),
        }


def utc_now_rfc3339() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def dependency_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def git_commit() -> str | None:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return output or None
    except Exception:
        return None


def collect_input_hashes(paths: Iterable[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for path_str in sorted(dict.fromkeys(paths)):
        path = Path(path_str)
        if not path.exists():
            continue
        out.append({"path": str(path), "sha256": file_sha256(path)})
    return out


def build_run_record(
    *,
    command: str,
    args: dict[str, Any],
    config_snapshot: dict[str, Any],
    input_files: list[str],
    warnings: list[str],
    cache_keys: list[str],
    device: str,
    workers: int,
    chunk_size: int,
    cache_chunk_max_mb: float,
    committee_enabled: bool,
    normalized_cell_pbc: bool,
    status: str,
    timing: Timing,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "tool_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(),
        "timestamp": utc_now_rfc3339(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            "ase": dependency_version("ase"),
            "shiftml": dependency_version("shiftml"),
            "numpy": dependency_version("numpy"),
            "pandas": dependency_version("pandas"),
            "pyarrow": dependency_version("pyarrow"),
        },
        "command": command,
        "args": args,
        "config": config_snapshot,
        "input_files": collect_input_hashes(input_files),
        "cache_keys": sorted(cache_keys),
        "runtime": {
            "device": device,
            "workers": workers,
            "chunk_size": chunk_size,
            "cache_chunk_max_mb": cache_chunk_max_mb,
            "committee_enabled": committee_enabled,
        },
        "units": {
            "length": "angstrom",
            "source": "ASE",
        },
        "normalization_version": "1",
        "normalized_cell_pbc": normalized_cell_pbc,
        "warnings": warnings,
        "timings": timing.as_dict(),
        "status": status,
    }
    if extra:
        record.update(extra)
    return record


def write_run_record(path: Path, record: dict[str, Any]) -> None:
    atomic_write_json(path, json.dumps(record, indent=2, sort_keys=True) + "\n")
