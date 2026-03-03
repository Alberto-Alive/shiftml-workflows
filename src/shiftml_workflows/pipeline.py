"""Prediction pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import importlib.metadata
import time

import numpy as np
import pandas as pd

from shiftml_workflows.backends.base import Backend
from shiftml_workflows.cache import CacheStore, compute_cache_key
from shiftml_workflows.config import PredictConfig
from shiftml_workflows.io import AtomsFrame, InputError, atomic_write_dataframe, count_atoms, load_structures, resolve_inputs
from shiftml_workflows.magres import write_magres
from shiftml_workflows.provenance import Timing, build_run_record, write_run_record
from shiftml_workflows.schema import SCHEMA_VERSION, required_columns, to_dataframe

SUPPORTED_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


class PipelineError(RuntimeError):
    exit_code = 1


class ValidationError(PipelineError):
    exit_code = 3


class MissingDependencyError(PipelineError):
    exit_code = 4


class BackendRuntimeError(PipelineError):
    exit_code = 5


@dataclass(slots=True)
class RunSummary:
    outdir: Path
    output_path: Path
    output_format: str
    frame_count: int
    atom_count: int
    cache_hits: int
    cache_misses: int
    device: str
    workers: int
    property_mode: str
    committee_enabled: bool
    warnings: list[str]
    timings: Timing
    cache_keys: list[str]
    normalized_cell_pbc: bool


def _dependency_version(pkg_name: str, fallback: str = "0") -> str:
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def resolve_output_format(requested: str) -> str:
    if requested == "csv":
        return "csv"
    if requested == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise MissingDependencyError("Parquet output requested but pyarrow is not installed") from exc
        return "parquet"

    try:
        import pyarrow  # noqa: F401

        return "parquet"
    except ImportError:
        return "csv"


def _validate_frames(frames: list[AtomsFrame], on_warning: str) -> tuple[list[AtomsFrame], list[str], bool]:
    warnings: list[str] = []
    warning_frame_ids: set[tuple[str, int]] = set()

    for frame in frames:
        atoms = frame.atoms
        if len(atoms) == 0:
            raise ValidationError(f"Empty structure frame found in {frame.source_file}#{frame.frame_index}")

        positions = np.asarray(atoms.positions, dtype=np.float64)
        if not np.isfinite(positions).all():
            raise ValidationError(
                f"Invalid atomic positions (NaN/inf) in {frame.source_file}#{frame.frame_index}"
            )

        symbols = atoms.get_chemical_symbols()
        unsupported = sorted({symbol for symbol in symbols if symbol not in SUPPORTED_ELEMENTS})
        if unsupported:
            raise ValidationError(
                f"Unsupported elements in {frame.source_file}#{frame.frame_index}: {', '.join(unsupported)}"
            )

        if frame.normalized_cell_pbc:
            warning_frame_ids.add((frame.source_file, frame.frame_index))

    if warning_frame_ids:
        warnings.append(
            "Missing cell/PBC normalized to zeros for frames: "
            + ", ".join(f"{src}#{idx}" for src, idx in sorted(warning_frame_ids))
        )

    if on_warning == "error" and warnings:
        raise ValidationError(warnings[0])

    if on_warning == "skip" and warning_frame_ids:
        kept = [f for f in frames if (f.source_file, f.frame_index) not in warning_frame_ids]
        if not kept:
            raise ValidationError("All frames were skipped by --on-warning skip")
        return kept, warnings, True

    return frames, warnings, bool(warning_frame_ids)


def _coerce_workers(cfg: PredictConfig, warnings: list[str]) -> int:
    if cfg.device == "cuda" and cfg.workers > 1 and not cfg.force_multi_gpu:
        warnings.append("device=cuda with workers>1 coerced to workers=1")
        return 1
    return cfg.workers


def _resolve_property_mode(cfg: PredictConfig) -> str:
    if cfg.property_mode is None:
        if cfg.committee:
            return "iso"
        return "iso"
    return cfg.property_mode


def _default_backend_factory(device: str) -> Backend:
    from shiftml_workflows.backends.shiftml3 import ShiftML3Backend

    return ShiftML3Backend(device=device)


def run_predict(
    *,
    inputs: list[str],
    config: PredictConfig,
    backend_factory: Callable[[str], Backend] | None = None,
) -> RunSummary:
    backend_factory = backend_factory or _default_backend_factory
    config.outdir.mkdir(parents=True, exist_ok=True)

    all_warnings: list[str] = []
    resolved_inputs: list[str] = []
    cache_keys: list[str] = []
    normalized_cell_pbc = False
    timings = Timing(setup_time_s=0.0, cache_time_s=0.0, predict_time_s=0.0)
    status = "failed"
    error_text: str | None = None
    summary: RunSummary | None = None
    runtime_workers = config.workers

    try:
        setup_start = time.perf_counter()
        resolved_inputs = resolve_inputs(inputs)
        loaded_frames = load_structures(resolved_inputs, frames=config.frames)
        frames, validation_warnings, normalized_cell_pbc = _validate_frames(loaded_frames, config.on_warning)
        all_warnings.extend(validation_warnings)

        workers = _coerce_workers(config, all_warnings)
        runtime_workers = workers
        property_mode = _resolve_property_mode(config)
        output_format = resolve_output_format(config.output_format)

        if config.dry_run:
            timings.setup_time_s = time.perf_counter() - setup_start
            return RunSummary(
                outdir=config.outdir,
                output_path=config.outdir / f"results.{output_format}",
                output_format=output_format,
                frame_count=len(frames),
                atom_count=count_atoms(frames),
                cache_hits=0,
                cache_misses=len(frames),
                device=config.device,
                    workers=workers,
                property_mode=property_mode,
                committee_enabled=config.committee,
                warnings=all_warnings,
                timings=timings,
                cache_keys=[],
                normalized_cell_pbc=normalized_cell_pbc,
            )

        model_version = _dependency_version("shiftml", fallback="unknown")
        flags = {
            "committee": config.committee,
            "property": property_mode,
        }
        cache_keys = [
            compute_cache_key(
                model_name="ShiftML3",
                model_version=model_version,
                schema_version=SCHEMA_VERSION,
                flags=flags,
                structure_id=frame.structure_id,
            )
            for frame in frames
        ]

        cache_store: CacheStore | None = None
        if config.cache_dir is not None:
            cache_store = CacheStore(
                cache_dir=config.cache_dir,
                output_format=output_format,
                model_name="ShiftML3",
                model_version=model_version,
                schema_version=SCHEMA_VERSION,
            )

        timings.setup_time_s = time.perf_counter() - setup_start

        cache_start = time.perf_counter()
        cached_df = pd.DataFrame()
        hit_keys: set[str] = set()
        if cache_store is not None:
            cached_df = cache_store.lookup_many(cache_keys)
            if not cached_df.empty and "frame_cache_key" in cached_df.columns:
                hit_keys = set(cached_df["frame_cache_key"].astype(str).unique().tolist())
        timings.cache_time_s = time.perf_counter() - cache_start

        miss_pairs = [(frame, key) for frame, key in zip(frames, cache_keys) if key not in hit_keys]

        predict_start = time.perf_counter()
        predicted_df = pd.DataFrame()
        if miss_pairs:
            miss_frames = [frame for frame, _ in miss_pairs]
            miss_keys = [key for _, key in miss_pairs]

            try:
                backend = backend_factory(config.device)
                prediction = backend.predict(
                    [frame.atoms for frame in miss_frames],
                    device=config.device,
                    committee=config.committee,
                    property=property_mode,
                )
            except ValidationError:
                raise
            except Exception as exc:
                raise BackendRuntimeError(f"Backend prediction failed: {exc}") from exc

            predicted_df = to_dataframe(
                miss_frames,
                prediction,
                property_mode=property_mode,
                committee=config.committee,
                frame_cache_keys=miss_keys,
            )
            if cache_store is not None and not predicted_df.empty:
                cache_store.store_dataframe(
                    predicted_df,
                    chunk_size=config.chunk_size,
                    max_chunk_mb=config.cache_chunk_max_mb,
                )

        timings.predict_time_s = time.perf_counter() - predict_start

        if not cached_df.empty and "frame_cache_key" not in cached_df.columns:
            cached_df["frame_cache_key"] = ""

        frames_df = pd.concat([cached_df, predicted_df], ignore_index=True) if not cached_df.empty or not predicted_df.empty else pd.DataFrame()
        if frames_df.empty:
            raise ValidationError("No results were produced")

        for col in required_columns(property_mode=property_mode, committee=config.committee):
            if col not in frames_df.columns:
                frames_df[col] = np.nan

        frames_df = frames_df.sort_values(["source_file", "frame", "atom_i"]).reset_index(drop=True)

        output_path = config.outdir / f"results.{output_format}"
        atomic_write_dataframe(output_path, frames_df, output_format)

        if config.magres_mode != "none":
            write_magres(frames_df, config.outdir, config.magres_mode)

        summary = RunSummary(
            outdir=config.outdir,
            output_path=output_path,
            output_format=output_format,
            frame_count=len(frames),
            atom_count=count_atoms(frames),
            cache_hits=len(hit_keys),
            cache_misses=len(miss_pairs),
            device=config.device,
                workers=workers,
            property_mode=property_mode,
            committee_enabled=config.committee,
            warnings=all_warnings,
            timings=timings,
            cache_keys=cache_keys,
            normalized_cell_pbc=normalized_cell_pbc,
        )
        status = "success"
        return summary

    except InputError as exc:
        error_text = str(exc)
        raise ValidationError(str(exc)) from exc
    except ValidationError as exc:
        error_text = str(exc)
        raise
    except MissingDependencyError as exc:
        error_text = str(exc)
        raise
    except BackendRuntimeError as exc:
        error_text = str(exc)
        raise
    except Exception as exc:  # pragma: no cover
        error_text = str(exc)
        raise PipelineError(str(exc)) from exc
    finally:
        if config.dry_run:
            pass
        else:
            if summary is None:
                status = "failed"
            try:
                record = build_run_record(
                    command="shiftmlwf predict",
                    args=config.command_args or {},
                    config_snapshot=config.as_dict(),
                    input_files=resolved_inputs or inputs,
                    warnings=all_warnings,
                    cache_keys=cache_keys,
                    device=config.device,
                    workers=runtime_workers,
                    chunk_size=config.chunk_size,
                    cache_chunk_max_mb=config.cache_chunk_max_mb,
                    committee_enabled=config.committee,
                    normalized_cell_pbc=normalized_cell_pbc,
                    status=status,
                    timing=timings,
                    extra={"error": error_text} if error_text else None,
                )
                write_run_record(config.outdir / "run.json", record)
            except Exception:
                # Provenance should not hide the underlying failure.
                pass
