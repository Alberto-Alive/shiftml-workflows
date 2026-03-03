from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


def test_python_module_info_command() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "shiftml_workflows", "info"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_help_command() -> None:
    shiftmlwf = shutil.which("shiftmlwf")
    if shiftmlwf:
        cmd = [shiftmlwf, "--help"]
    else:
        cmd = [sys.executable, "-m", "shiftml_workflows", "--help"]

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_python_module_average_command(tmp_path: Path) -> None:
    input_csv = tmp_path / "results.csv"
    input_csv.write_text(
        "\n".join(
            [
                "source_file,frame,atom_i,element,x,y,z,cs_iso",
                "a.xyz,0,0,H,0,0,0,10",
                "a.xyz,1,0,H,0,0,0,14",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "avg.csv"

    proc = subprocess.run(
        [sys.executable, "-m", "shiftml_workflows", "average", str(input_csv), "--out", str(out_csv)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_csv.exists()


def test_python_module_cache_compact_command(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    index_file = cache_dir / "index" / "index_a.jsonl"
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text(
        json.dumps(
            {
                "cache_key": "k1",
                "chunk_path": "chunk.csv",
                "output_format": "csv",
                "seq": 1,
                "writer_id": "writer-a",
                "written_at_utc": "2026-01-01T00:00:00Z",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "shiftml_workflows", "cache-compact", str(cache_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    compacted = list((cache_dir / "index").glob("index_compacted_*.jsonl"))
    assert compacted
