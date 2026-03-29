# Changelog

## [0.5.0] - Unreleased

## [0.4.0] - 2026-03-04
- Implemented CPU multi-worker prediction execution via frame chunking and worker-local backend instances.
- Preserved deterministic final output row ordering by `(source_file, frame, atom_i)` for multi-worker runs.
- Extended `.magres` export to include lattice/cell lines and explicit PBC metadata comments (best-effort Soprano compatibility).
- Kept tensor-versus-isotropic fallback behavior unchanged in magres emission.
- Added tests for CPU multi-worker worker behavior, deterministic output across runs, and magres metadata emission.
- Kept `SCHEMA_VERSION="1"` and stable output columns unchanged.

## [0.3.0] - 2026-03-03
- Added `shiftmlwf average` CLI for per-atom weighted/unweighted averaging across `results.csv|parquet` tables, with JSON/YAML frame-weight maps.
- Added `shiftmlwf cache-compact` CLI and cache utilities to compact `index_*.jsonl` logs deterministically, keeping latest records per `(cache_key, output_format)`.
- Added test coverage for averaging behavior, cache-index compaction semantics, CLI smoke paths for new commands, and magres emission in pipeline runs.
- Polished integration docs for Snakemake and AiiDA, including average workflow examples.

## [0.2.0] - 2026-03-03
- Hardened cache/index scanning with streaming JSONL parsing, deterministic duplicate-key precedence, malformed trailing-line tolerance, and stale lookup-cache invalidation after writes.
- Improved cache hit materialization so cached predictions are remapped to current frame metadata, including duplicate-key input handling.
- Hardened validation policy handling so `--strict` maps to the same path as `--on-warning error`, while unsupported elements remain hard errors even with `--on-warning skip`.
- Hardened runtime device behavior: `auto` resolves predictably to `cuda` or `cpu`, worker coercion is based on resolved runtime device, and runtime metadata reports the resolved device/workers.
- Expanded test coverage for cache precedence/invalidation, warning policy semantics, device resolution/coercion, and failure-path `run.json` metadata.

## [0.1.0] - 2026-03-03
- Initial scaffold for ShiftML workflows package and CLI.
- Added deterministic schema, hashing, provenance, and cache primitives.
- Added test suite and CI workflow.
