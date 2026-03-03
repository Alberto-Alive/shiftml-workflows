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
from shiftml_workflows.config import PredictConfig, build_predict_config
import shiftml_workflows.pipeline as pipeline
from shiftml_workflows.pipeline import BackendRuntimeError, ValidationError, run_predict


class DummyBackend:
    name = "dummy"

    def predict(self, frames, *, device="auto", committee=False, property="iso"):
        predictions = []
        for atoms in frames:
            n = len(atoms)
            iso = np.linspace(1.0, float(n), n)
            predictions.append(FramePrediction(cs_iso=iso))
        return PredictionResult(frames=predictions)


class FailingBackend:
    name = "failing"

    def predict(self, frames, *, device="auto", committee=False, property="iso"):
        raise RuntimeError("boom")


def _cell_frame() -> Atoms:
    return Atoms("HCO", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[8, 8, 8], pbc=True)


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


def test_strict_uses_same_validation_path_as_on_warning_error(tmp_path: Path) -> None:
    in_path = tmp_path / "sample.xyz"
    write(in_path, [Atoms("HCO", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])])

    cfg_error = build_predict_config(
        outdir=tmp_path / "out_error",
        cli_values={"on_warning": "error"},
    )
    cfg_strict = build_predict_config(
        outdir=tmp_path / "out_strict",
        cli_values={"on_warning": "warn", "strict": True},
    )
    assert cfg_strict.on_warning == "error"

    with pytest.raises(ValidationError) as error_path:
        run_predict(inputs=[str(in_path)], config=cfg_error, backend_factory=lambda _device: DummyBackend())
    with pytest.raises(ValidationError) as strict_path:
        run_predict(inputs=[str(in_path)], config=cfg_strict, backend_factory=lambda _device: DummyBackend())
    assert str(error_path.value) == str(strict_path.value)


def test_on_warning_skip_keeps_non_warning_frames(tmp_path: Path) -> None:
    warn_path = tmp_path / "warn.xyz"
    ok_path = tmp_path / "ok.extxyz"
    write(warn_path, [Atoms("H", positions=[[0, 0, 0]])])
    write(ok_path, [_cell_frame()])

    cfg = PredictConfig(
        outdir=tmp_path / "out",
        output_format="csv",
        on_warning="skip",
    )
    summary = run_predict(
        inputs=[str(warn_path), str(ok_path)],
        config=cfg,
        backend_factory=lambda _device: DummyBackend(),
    )
    assert summary.frame_count == 1

    df = pd.read_csv(summary.output_path)
    assert df["source_file"].str.endswith("ok.extxyz").all()


def test_auto_device_resolves_to_cpu_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline, "_cuda_available", lambda: False)
    in_path = tmp_path / "sample.extxyz"
    write(in_path, [_cell_frame()])

    calls: dict[str, str] = {}

    class CapturingBackend(DummyBackend):
        def predict(self, frames, *, device="auto", committee=False, property="iso"):
            calls["predict_device"] = str(device)
            return super().predict(frames, device=device, committee=committee, property=property)

    def backend_factory(device: str) -> DummyBackend:
        calls["factory_device"] = str(device)
        return CapturingBackend()

    cfg = PredictConfig(
        outdir=tmp_path / "out",
        output_format="csv",
        device="auto",
    )
    summary = run_predict(inputs=[str(in_path)], config=cfg, backend_factory=backend_factory)
    assert summary.device == "cpu"
    assert calls["factory_device"] == "cpu"
    assert calls["predict_device"] == "cpu"


def test_auto_cuda_coerces_workers_without_force(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline, "_cuda_available", lambda: True)
    in_path = tmp_path / "sample.extxyz"
    write(in_path, [_cell_frame()])

    cfg = PredictConfig(
        outdir=tmp_path / "out",
        output_format="csv",
        device="auto",
        workers=4,
        force_multi_gpu=False,
    )
    summary = run_predict(inputs=[str(in_path)], config=cfg, backend_factory=lambda _device: DummyBackend())
    assert summary.device == "cuda"
    assert summary.workers == 1
    assert any("coerced to workers=1" in warning for warning in summary.warnings)

    run_json = json.loads((cfg.outdir / "run.json").read_text(encoding="utf-8"))
    assert run_json["runtime"]["device"] == "cuda"
    assert run_json["runtime"]["workers"] == 1


def test_auto_cuda_respects_force_multi_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline, "_cuda_available", lambda: True)
    in_path = tmp_path / "sample.extxyz"
    write(in_path, [_cell_frame()])

    cfg = PredictConfig(
        outdir=tmp_path / "out",
        output_format="csv",
        device="auto",
        workers=4,
        force_multi_gpu=True,
    )
    summary = run_predict(inputs=[str(in_path)], config=cfg, backend_factory=lambda _device: DummyBackend())
    assert summary.device == "cuda"
    assert summary.workers == 4
    assert not any("coerced to workers=1" in warning for warning in summary.warnings)


def test_run_json_written_on_backend_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline, "_cuda_available", lambda: False)
    in_path = tmp_path / "sample.extxyz"
    write(in_path, [_cell_frame()])

    cfg = PredictConfig(
        outdir=tmp_path / "out",
        output_format="csv",
        device="auto",
        workers=2,
    )

    with pytest.raises(BackendRuntimeError):
        run_predict(inputs=[str(in_path)], config=cfg, backend_factory=lambda _device: FailingBackend())

    run_json = json.loads((cfg.outdir / "run.json").read_text(encoding="utf-8"))
    assert run_json["status"] == "failed"
    assert "Backend prediction failed: boom" in run_json["error"]
    assert run_json["runtime"]["device"] == "cpu"
    assert run_json["runtime"]["workers"] == 2
    assert set(run_json["timings"]) == {"setup_time_s", "cache_time_s", "predict_time_s"}


def test_run_json_written_with_cli_like_path_args(tmp_path: Path) -> None:
    in_path = tmp_path / "sample.extxyz"
    write(in_path, [_cell_frame()])

    cfg = build_predict_config(
        outdir=tmp_path / "out",
        cli_values={
            "output_format": "csv",
            "cache_dir": tmp_path / "cache",
            "strict": False,
        },
    )

    run_predict(inputs=[str(in_path)], config=cfg, backend_factory=lambda _device: DummyBackend())
    run_json = json.loads((cfg.outdir / "run.json").read_text(encoding="utf-8"))
    assert run_json["status"] == "success"
    assert isinstance(run_json["config"]["command_args"]["cache_dir"], str)


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


def test_pipeline_writes_magres_outputs(tmp_path: Path) -> None:
    in_path = tmp_path / "sample.extxyz"
    write(in_path, [_cell_frame()])

    cfg = PredictConfig(
        outdir=tmp_path / "out",
        output_format="csv",
        magres_mode="per-frame",
    )
    run_predict(inputs=[str(in_path)], config=cfg, backend_factory=lambda _device: DummyBackend())

    per_frame_files = sorted((cfg.outdir / "magres").glob("*.magres"))
    assert len(per_frame_files) == 1
    assert "[magres]" in per_frame_files[0].read_text(encoding="utf-8")

    cfg_single = PredictConfig(
        outdir=tmp_path / "out_single",
        output_format="csv",
        magres_mode="single",
    )
    run_predict(inputs=[str(in_path)], config=cfg_single, backend_factory=lambda _device: DummyBackend())
    single_file = cfg_single.outdir / "predictions.magres"
    assert single_file.exists()
    assert single_file.read_text(encoding="utf-8").count("#$magres-abinitio-v1.0") == 1
