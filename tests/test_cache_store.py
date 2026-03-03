from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shiftml_workflows.cache import CacheStore


def _write_chunk(path: Path, cache_key: str, cs_iso: float) -> None:
    df = pd.DataFrame(
        {
            "frame_cache_key": [cache_key],
            "structure_id": ["sid"],
            "source_file": ["src.xyz"],
            "frame": [0],
            "atom_i": [0],
            "element": ["H"],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "cs_iso": [cs_iso],
        }
    )
    df.to_csv(path, index=False)


def _append_index_line(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _record(cache_key: str, chunk_path: Path, *, writer_id: str, seq: int, written_at_utc: str) -> dict[str, object]:
    return {
        "cache_key": cache_key,
        "chunk_path": str(chunk_path),
        "output_format": "csv",
        "writer_id": writer_id,
        "seq": seq,
        "written_at_utc": written_at_utc,
    }


def _store(tmp_path: Path) -> CacheStore:
    return CacheStore(
        cache_dir=tmp_path / "cache",
        output_format="csv",
        model_name="ShiftML3",
        model_version="1",
        schema_version="1",
    )


def test_duplicate_key_merge_prefers_latest_timestamp(tmp_path: Path) -> None:
    store = _store(tmp_path)

    chunk_old = store.chunk_dir / "old.csv"
    chunk_new = store.chunk_dir / "new.csv"
    _write_chunk(chunk_old, "k1", 1.0)
    _write_chunk(chunk_new, "k1", 2.0)

    index_a = store.index_dir / "index_a.jsonl"
    index_b = store.index_dir / "index_b.jsonl"
    _append_index_line(
        index_a,
        _record("k1", chunk_old, writer_id="writer-a", seq=1, written_at_utc="2026-01-01T00:00:00Z"),
    )
    _append_index_line(
        index_b,
        _record("k1", chunk_new, writer_id="writer-b", seq=1, written_at_utc="2026-01-01T00:00:01Z"),
    )

    hit_df = store.lookup_many(["k1"])
    assert hit_df["cs_iso"].tolist() == [2.0]


def test_duplicate_key_merge_prefers_last_line_in_same_file(tmp_path: Path) -> None:
    store = _store(tmp_path)

    chunk_first = store.chunk_dir / "line1.csv"
    chunk_second = store.chunk_dir / "line2.csv"
    _write_chunk(chunk_first, "k1", 1.0)
    _write_chunk(chunk_second, "k1", 3.0)

    index_path = store.index_dir / "index_tie.jsonl"
    base = {"writer_id": "writer-a", "seq": 9, "written_at_utc": "2026-01-01T00:00:00Z"}
    _append_index_line(index_path, _record("k1", chunk_first, **base))
    _append_index_line(index_path, _record("k1", chunk_second, **base))

    hit_df = store.lookup_many(["k1"])
    assert hit_df["cs_iso"].tolist() == [3.0]


def test_duplicate_key_merge_prefers_lexical_path_after_equal_line(tmp_path: Path) -> None:
    store = _store(tmp_path)

    chunk_a = store.chunk_dir / "path_a.csv"
    chunk_b = store.chunk_dir / "path_b.csv"
    _write_chunk(chunk_a, "k1", 4.0)
    _write_chunk(chunk_b, "k1", 5.0)

    base = {"writer_id": "writer-a", "seq": 1, "written_at_utc": "2026-01-01T00:00:00Z"}
    index_a = store.index_dir / "index_a.jsonl"
    index_b = store.index_dir / "index_b.jsonl"
    _append_index_line(index_a, _record("k1", chunk_a, **base))
    _append_index_line(index_b, _record("k1", chunk_b, **base))

    hit_df = store.lookup_many(["k1"])
    assert hit_df["cs_iso"].tolist() == [5.0]


def test_lookup_cache_invalidated_after_store(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.lookup_many(["k1"]).empty

    df = pd.DataFrame(
        {
            "frame_cache_key": ["k1"],
            "structure_id": ["sid"],
            "source_file": ["src.xyz"],
            "frame": [0],
            "atom_i": [0],
            "element": ["H"],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "cs_iso": [9.0],
        }
    )
    store.store_dataframe(df, chunk_size=1, max_chunk_mb=1.0)

    hit_df = store.lookup_many(["k1"])
    assert hit_df["cs_iso"].tolist() == [9.0]


def test_lookup_prefers_matching_output_format(tmp_path: Path) -> None:
    store = _store(tmp_path)

    chunk_csv = store.chunk_dir / "csv.csv"
    chunk_mismatch = store.chunk_dir / "mismatch.csv"
    _write_chunk(chunk_csv, "k1", 6.0)
    _write_chunk(chunk_mismatch, "k1", 7.0)

    index_path = store.index_dir / "index_format.jsonl"
    _append_index_line(
        index_path,
        {
            "cache_key": "k1",
            "chunk_path": str(chunk_csv),
            "output_format": "csv",
            "writer_id": "writer-a",
            "seq": 1,
            "written_at_utc": "2026-01-01T00:00:00Z",
        },
    )
    _append_index_line(
        index_path,
        {
            "cache_key": "k1",
            "chunk_path": str(chunk_mismatch),
            "output_format": "parquet",
            "writer_id": "writer-a",
            "seq": 2,
            "written_at_utc": "2026-01-01T00:00:01Z",
        },
    )

    hit_df = store.lookup_many(["k1"])
    assert hit_df["cs_iso"].tolist() == [6.0]
