"""Cache key and chunked cache storage utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import json
import os
import secrets
import socket

import hashlib

import pandas as pd


def utc_now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_rfc3339_utc(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError("timestamp must end with Z")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def compute_cache_key(
    model_name: str,
    model_version: str,
    schema_version: str,
    flags: dict[str, object],
    structure_id: str,
) -> str:
    payload = {
        "flags": flags,
        "model_name": model_name,
        "model_version": model_version,
        "schema_version": schema_version,
        "structure_id": structure_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class _IndexRecord:
    cache_key: str
    chunk_path: str
    output_format: str
    writer_id: str
    seq: int
    written_at_utc: str
    written_at: datetime
    source_index_path: str
    source_line_no: int

    @property
    def rank(self) -> tuple[datetime, str, int, int, str]:
        return (
            self.written_at,
            self.writer_id,
            self.seq,
            self.source_line_no,
            self.source_index_path,
        )


class CacheStore:
    """Chunked cache store with per-writer index logs."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        output_format: str,
        model_name: str,
        model_version: str,
        schema_version: str,
    ) -> None:
        self.cache_dir = cache_dir
        self.output_format = output_format
        self.model_name = model_name
        self.model_version = model_version
        self.schema_version = schema_version
        self.writer_id = f"{socket.gethostname()}-{os.getpid()}-{secrets.token_hex(3)}"

        self.chunk_dir = self.cache_dir / "chunks"
        self.index_dir = self.cache_dir / "index"
        self.index_path = self.index_dir / f"index_{self.writer_id}.jsonl"

        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._line_seq = 0
        self._chunk_seq = 0
        self._manifest: dict[str, _IndexRecord] | None = None
        self._resolved_lookup_cache: dict[str, _IndexRecord | None] = {}
        self._chunk_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    def _next_line_seq(self) -> int:
        self._line_seq += 1
        return self._line_seq

    def _next_chunk_seq(self) -> int:
        self._chunk_seq += 1
        return self._chunk_seq

    def _append_index_line(self, payload: dict[str, object]) -> None:
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _iter_index_records(self, index_path: Path) -> Iterator[_IndexRecord]:
        try:
            with index_path.open("r", encoding="utf-8") as fh:
                for line_no, raw in enumerate(fh, start=1):
                    if not raw.strip():
                        continue
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        # Tolerate partially-written trailing lines.
                        continue

                    try:
                        cache_key = str(payload["cache_key"])
                        chunk_path = str(payload["chunk_path"])
                        output_format = str(payload["output_format"])
                        writer_id = str(payload["writer_id"])
                        seq = int(payload["seq"])
                        written_at_utc = str(payload["written_at_utc"])
                        written_at = parse_rfc3339_utc(written_at_utc)
                    except Exception:
                        continue

                    yield _IndexRecord(
                        cache_key=cache_key,
                        chunk_path=chunk_path,
                        output_format=output_format,
                        writer_id=writer_id,
                        seq=seq,
                        written_at_utc=written_at_utc,
                        written_at=written_at,
                        source_index_path=str(index_path),
                        source_line_no=line_no,
                    )
        except OSError:
            return

    def _prefer_record(self, record: _IndexRecord, existing: _IndexRecord) -> bool:
        record_pref = record.output_format == self.output_format
        existing_pref = existing.output_format == self.output_format
        if record_pref != existing_pref:
            return record_pref
        return record.rank > existing.rank

    def _read_chunk(self, chunk_path: str, output_format: str) -> pd.DataFrame | None:
        chunk_key = (chunk_path, output_format)
        if chunk_key in self._chunk_cache:
            return self._chunk_cache[chunk_key]

        chunk_file = Path(chunk_path)
        if not chunk_file.exists():
            self._chunk_cache[chunk_key] = None
            return None
        try:
            if output_format == "parquet":
                chunk_df = pd.read_parquet(chunk_file)
            else:
                chunk_df = pd.read_csv(chunk_file)
        except Exception:
            self._chunk_cache[chunk_key] = None
            return None

        if "frame_cache_key" not in chunk_df.columns:
            self._chunk_cache[chunk_key] = None
            return None

        self._chunk_cache[chunk_key] = chunk_df
        return chunk_df

    def _load_manifest(self) -> dict[str, _IndexRecord]:
        manifest: dict[str, _IndexRecord] = {}
        for index_path in sorted(self.index_dir.glob("index_*.jsonl")):
            for record in self._iter_index_records(index_path):
                existing = manifest.get(record.cache_key)
                if existing is None or self._prefer_record(record, existing):
                    manifest[record.cache_key] = record

        self._manifest = manifest
        self._resolved_lookup_cache.clear()
        return manifest

    def _manifest_or_load(self) -> dict[str, _IndexRecord]:
        if self._manifest is None:
            return self._load_manifest()
        return self._manifest

    def lookup_many(self, keys: Iterable[str]) -> pd.DataFrame:
        key_list = [str(key) for key in keys]
        if not key_list:
            return pd.DataFrame()

        manifest = self._manifest_or_load()
        unique_keys = list(dict.fromkeys(key_list))
        unresolved = [key for key in unique_keys if key not in self._resolved_lookup_cache]
        for key in unresolved:
            self._resolved_lookup_cache[key] = manifest.get(key)

        by_chunk: dict[tuple[str, str], set[str]] = {}
        for key in unique_keys:
            record = self._resolved_lookup_cache.get(key)
            if record is None:
                continue
            by_chunk.setdefault((record.chunk_path, record.output_format), set()).add(key)

        parts: list[pd.DataFrame] = []
        for (chunk_path, output_format), wanted_keys in by_chunk.items():
            chunk_df = self._read_chunk(chunk_path, output_format)
            if chunk_df is None:
                continue
            hit = chunk_df[chunk_df["frame_cache_key"].isin(wanted_keys)]
            if not hit.empty:
                parts.append(hit)

        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def store_dataframe(
        self,
        df: pd.DataFrame,
        *,
        chunk_size: int,
        max_chunk_mb: float,
    ) -> None:
        if df.empty:
            return
        if "frame_cache_key" not in df.columns:
            raise ValueError("Cannot store cache dataframe without frame_cache_key column")

        frame_groups = [group for _, group in df.groupby("frame_cache_key", sort=False)]
        max_chunk_bytes = int(max_chunk_mb * 1024 * 1024)
        idx = 0

        while idx < len(frame_groups):
            remaining = len(frame_groups) - idx
            effective_chunk_size = min(remaining, chunk_size)
            current_groups: list[pd.DataFrame] = []
            current_frames = 0
            current_bytes = 0

            while idx < len(frame_groups):
                next_group = frame_groups[idx]
                next_size = int(next_group.memory_usage(index=True, deep=True).sum())
                exceed_size = current_frames > 0 and (current_bytes + next_size) > max_chunk_bytes
                exceed_count = current_frames >= effective_chunk_size
                if exceed_size or exceed_count:
                    break
                current_groups.append(next_group)
                current_frames += 1
                current_bytes += next_size
                idx += 1

            if not current_groups:
                current_groups = [frame_groups[idx]]
                idx += 1

            chunk_df = pd.concat(current_groups, ignore_index=True)
            chunk_seq = self._next_chunk_seq()
            suffix = "parquet" if self.output_format == "parquet" else "csv"
            chunk_path = self.chunk_dir / f"chunk_{self.writer_id}_{chunk_seq:06d}.{suffix}"
            tmp_chunk_path = chunk_path.with_suffix(chunk_path.suffix + f".tmp-{os.getpid()}")
            if self.output_format == "parquet":
                chunk_df.to_parquet(tmp_chunk_path, index=False)
            else:
                chunk_df.to_csv(tmp_chunk_path, index=False)
            os.replace(tmp_chunk_path, chunk_path)

            now = utc_now_rfc3339()
            keys = list(dict.fromkeys(chunk_df["frame_cache_key"].astype(str).tolist()))
            for key in keys:
                seq = self._next_line_seq()
                self._append_index_line(
                    {
                        "cache_key": key,
                        "chunk_path": str(chunk_path),
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                        "output_format": self.output_format,
                        "schema_version": self.schema_version,
                        "seq": seq,
                        "writer_id": self.writer_id,
                        "written_at_utc": now,
                    }
                )

        # Manifest cache is stale after writes.
        self._manifest = None
        self._resolved_lookup_cache.clear()
