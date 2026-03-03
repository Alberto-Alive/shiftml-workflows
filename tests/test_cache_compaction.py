from __future__ import annotations

import json
from pathlib import Path

from shiftml_workflows.cache import compact_index_files


def _append_index_line(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _index_record(cache_key: str, chunk_path: str, *, output_format: str, written_at_utc: str, seq: int) -> dict[str, object]:
    return {
        "cache_key": cache_key,
        "chunk_path": chunk_path,
        "model_name": "ShiftML3",
        "model_version": "1.0",
        "output_format": output_format,
        "schema_version": "1",
        "seq": seq,
        "writer_id": "writer-a",
        "written_at_utc": written_at_utc,
    }


def test_compact_index_keeps_latest_per_key_and_format(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    index_a = cache_dir / "index" / "index_a.jsonl"
    index_b = cache_dir / "index" / "index_b.jsonl"

    _append_index_line(
        index_a,
        _index_record("k1", "chunk_old.csv", output_format="csv", written_at_utc="2026-01-01T00:00:00Z", seq=1),
    )
    _append_index_line(
        index_b,
        _index_record("k1", "chunk_new.csv", output_format="csv", written_at_utc="2026-01-01T00:00:01Z", seq=1),
    )
    _append_index_line(
        index_a,
        _index_record("k1", "chunk_parquet.parquet", output_format="parquet", written_at_utc="2026-01-01T00:00:02Z", seq=1),
    )
    with index_b.open("a", encoding="utf-8") as fh:
        fh.write('{"bad":')

    summary = compact_index_files(cache_dir=cache_dir)
    assert summary["scanned_records"] == 3
    assert summary["kept_records"] == 2
    assert summary["removed_source_index_files"] == 0

    compact_path = Path(str(summary["compacted_index_file"]))
    payloads = [json.loads(line) for line in compact_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_key_and_format = {(p["cache_key"], p["output_format"]): p for p in payloads}
    assert by_key_and_format[("k1", "csv")]["chunk_path"] == "chunk_new.csv"
    assert by_key_and_format[("k1", "parquet")]["chunk_path"] == "chunk_parquet.parquet"


def test_compact_index_can_remove_source_index_files(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    index_a = cache_dir / "index" / "index_a.jsonl"
    _append_index_line(
        index_a,
        _index_record("k1", "chunk.csv", output_format="csv", written_at_utc="2026-01-01T00:00:00Z", seq=1),
    )

    summary = compact_index_files(cache_dir=cache_dir, remove_source_indexes=True)
    compact_path = Path(str(summary["compacted_index_file"]))
    assert compact_path.exists()
    assert not index_a.exists()
    assert summary["removed_source_index_files"] == 1
