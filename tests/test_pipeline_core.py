from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import write

from shiftml_workflows.backends.base import FramePrediction, PredictionResult
from shiftml_workflows.cache import CacheStore
from shiftml_workflows.config import PredictConfig
from shiftml_workflows.pipeline import ValidationError, run_predict


class DummyBackend:
    name = "dummy"

    def predict(self, frames, *, device="auto", committee=False, property="iso"):
        predictions = []
        for atoms in frames:
            n = len(atoms)
            iso = np.linspace(1.0, float(n), n)
            predictions.append(FramePrediction(cs_iso=iso))
        return PredictionResult(frames=predictions)


def test_pipeline_writes_sorted_output_and_provenance(tmp_path: Path) -> None:
    frames = [
        Atoms("HCO", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        Atoms("HCO", positions=[[0.1, 0, 0], [0, 0.1, 1], [0, 1.1, 0]]),
    ]
    in_path = tmp_path / "sample.xyz"
    write(in_path, frames)

    cfg = PredictConfig(
        outdir=tmp_path / "out",
        output_format="csv",
        cache_dir=tmp_path / "cache",
    )

    summary = run_predict(inputs=[str(in_path)], config=cfg, backend_factory=lambda _device: DummyBackend())
    assert summary.output_path.exists()

    df = pd.read_csv(summary.output_path)
    sorted_df = df.sort_values(["source_file", "frame", "atom_i"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), sorted_df)

    run_json = json.loads((cfg.outdir / "run.json").read_text(encoding="utf-8"))
    assert run_json["normalized_cell_pbc"] is True
    assert "timings" in run_json


def test_on_warning_skip_drops_all_and_fails(tmp_path: Path) -> None:
    frame = Atoms("HCO", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    in_path = tmp_path / "sample.xyz"
    write(in_path, [frame])

    cfg = PredictConfig(
        outdir=tmp_path / "out",
        output_format="csv",
        on_warning="skip",
    )

    with pytest.raises(ValidationError):
        run_predict(inputs=[str(in_path)], config=cfg, backend_factory=lambda _device: DummyBackend())


def test_unsupported_elements_remain_hard_error_with_skip(tmp_path: Path) -> None:
    frame = Atoms("He", positions=[[0, 0, 0]])
    in_path = tmp_path / "sample.xyz"
    write(in_path, [frame])

    cfg = PredictConfig(
        outdir=tmp_path / "out",
        output_format="csv",
        on_warning="skip",
    )

    with pytest.raises(ValidationError):
        run_predict(inputs=[str(in_path)], config=cfg, backend_factory=lambda _device: DummyBackend())


def test_cache_store_ignores_malformed_trailing_line(tmp_path: Path) -> None:
    store = CacheStore(
        cache_dir=tmp_path / "cache",
        output_format="csv",
        model_name="ShiftML3",
        model_version="1",
        schema_version="1",
    )

    df = pd.DataFrame(
        {
            "frame_cache_key": ["k1", "k1", "k2"],
            "source_file": ["a", "a", "b"],
            "frame": [0, 0, 1],
            "atom_i": [0, 1, 0],
            "cs_iso": [1.0, 2.0, 3.0],
        }
    )
    store.store_dataframe(df, chunk_size=1, max_chunk_mb=0.0001)

    # Corrupt trailing line.
    with store.index_path.open("a", encoding="utf-8") as fh:
        fh.write('{"bad":')

    hit_df = store.lookup_many(["k1", "k2"])
    assert sorted(hit_df["frame_cache_key"].unique().tolist()) == ["k1", "k2"]
