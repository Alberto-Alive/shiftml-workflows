# shiftml-workflows

CLI and Python workflows for running ShiftML3 predictions with deterministic outputs, cache support, and provenance.

## Installation

### Editable install

```bash
pip install -e .
```

### From git

```bash
pip install "git+https://github.com/your-org/shiftml-workflows.git"
```

### Optional extras

```bash
pip install -e ".[parquet,yaml,pretty,spectra]"
```

## CLI quickstart

### Laptop CPU run

```bash
shiftmlwf predict examples/minimal.xyz --out out/cpu --device cpu
```

### HPC-style run with cache

```bash
shiftmlwf predict "data/*.extxyz" \
  --out out/hpc \
  --workers 8 \
  --chunk-size 200 \
  --cache-chunk-max-mb 100 \
  --cache-dir /scratch/$USER/shiftml-cache \
  --format auto
```

### Post-processing average

```bash
shiftmlwf average out/hpc/results.csv --out out/hpc/avg.csv
```

### Cache index compaction

```bash
shiftmlwf cache-compact /scratch/$USER/shiftml-cache
```

## Outputs

- `results.csv` or `results.parquet`
- optional magres files (`--magres per-frame|single`) with lattice/cell and PBC metadata
- `run.json` with provenance, warnings, timings, and runtime metadata

### Stable schema columns

Always present:
- `structure_id`, `source_file`, `frame`, `atom_i`, `element`, `x`, `y`, `z`, `cs_iso`

Optional:
- `cs_iso_uncertainty` when committee mode is enabled
- tensor columns when `--property tensor|both`

Row order is deterministic and sorted by `(source_file, frame, atom_i)`.

## Caching

Cache keys are per frame and depend on:
- model name/version
- schema version
- prediction flags
- normalized structure hash

Cache storage is chunked to avoid many tiny files:
- frame-count threshold: `--chunk-size`
- size threshold: `--cache-chunk-max-mb` (must be `> 0`; recommend `50-200` MB)

Cache index uses per-writer JSONL files:
- `index_<hostname>-<pid>-<run_suffix>.jsonl`
- malformed trailing lines are ignored safely
- duplicate-key merge precedence is deterministic: latest by `(written_at_utc, writer_id, seq)`, then line order, then index path

Cache lookups remap cached predictions onto the current frame metadata (`source_file`, `frame`, coordinates), so reused keys remain correct across files/runs.

For long-lived caches with many writer logs, compact index files to speed startup scans:
- `shiftmlwf cache-compact <cache_dir>`
- add `--remove-source-indexes` to delete old `index_*.jsonl` logs after successful compaction

Compaction keeps the latest entry per `(cache_key, output_format)` using the same deterministic precedence as lookup.

## Averaging

`shiftmlwf average` computes per-atom weighted or unweighted averages across frames.

- input: `results.csv` or `results.parquet`
- output: CSV or parquet (`--format auto|csv|parquet`)
- optional weights file: JSON/YAML mapping `"<source_file>#<frame>" -> weight`

## Committee uncertainty

If `--committee` is set and `--property` is omitted, property resolves to `iso`.
If you explicitly set `--property tensor|both`, it is honored.

## Validation policy

- `--on-warning warn`: keep warning-class frames and report warnings.
- `--on-warning error`: fail immediately on warning-class issues.
- `--on-warning skip`: skip only warning-class frames; if all are skipped, exits with validation error.
- `--strict` is an alias for `--on-warning error` and uses the same validation codepath.
- Unsupported elements are always hard validation errors (never skippable).

## Runtime device selection

- `--device auto` resolves to `cuda` when available, otherwise `cpu`.
- `--device cpu|cuda` is used as requested.
- `device=cuda` with `workers>1` is coerced to `workers=1` unless `--force-multi-gpu` is set.
- `device=cpu` with `workers>1` runs chunked multi-worker prediction and instantiates one backend per worker thread.
- `run.json` runtime metadata records the resolved runtime device and worker count.

## Workflow integrations

- Snakemake wrapper and environment: `integrations/snakemake/`
- AiiDA notes and CalcJob skeleton: `integrations/aiida/`

## Limitations

- Supported element list is conservative and validated before prediction.
- Missing cell/PBC is normalized to `zeros` + `pbc=(0,0,0)` and recorded in provenance.
- Tensor output availability depends on backend capabilities.
- Ensemble averaging assumes consistent atom indexing when averaging across frames.

## Reproducibility and locking

Dependency ranges are bounded in `pyproject.toml`.
For fully pinned environments, use your own lock workflow (`uv`, `pip-tools`, or Poetry) in downstream projects.
