# AiiDA integration notes

Recommended path:
1. Use an AiiDA `CalcJob` that calls `shiftmlwf predict`.
2. Store `results.*` and `run.json` as retrieved outputs.
3. Optionally run `shiftmlwf average` in a follow-up calcfunction for ensemble post-processing.

Alternative path:
1. Use a PythonJob/calcfunction that imports `shiftml_workflows.pipeline.run_predict`.
2. Pass file inputs and output folder paths explicitly.

See `example_calcjob/` for a minimal skeleton.
