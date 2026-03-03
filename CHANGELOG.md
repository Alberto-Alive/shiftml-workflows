# Changelog

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
