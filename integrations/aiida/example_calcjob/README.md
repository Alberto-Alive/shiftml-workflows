# Minimal CalcJob skeleton

Expected inputs:
- structure file(s)
- CLI flags or JSON config

Expected retrieved outputs:
- `results.csv` or `results.parquet`
- `run.json`
- optional `predictions.magres` or `magres/*.magres`

Implementation sketch:
1. Stage input structures in working directory.
2. Execute: `shiftmlwf predict <inputs> --out .`
3. Parse `run.json` for metadata and attach outputs to AiiDA nodes.
