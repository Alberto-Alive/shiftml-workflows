"""Prediction pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import threading
import time

import numpy as np
import pandas as pd

from shiftml_workflows.backends.base import Backend
from shiftml_workflows.cache import CacheStore, compute_cache_key
from shiftml_workflows.config import PredictConfig
from shiftml_workflows.io import AtomsFrame, InputError, atomic_write_dataframe, count_atoms, load_structures, resolve_inputs
from shiftml_workflows.magres import write_magres
from shiftml_workflows.parallel import chunked
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


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_runtime_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if _cuda_available() else "cpu"
    return requested_device


def _coerce_workers(runtime_device: str, cfg: PredictConfig, warnings: list[str]) -> int:
    if runtime_device == "cuda" and cfg.workers > 1 and not cfg.force_multi_gpu:
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


def _canonicalize_cache_rows(group: pd.DataFrame) -> pd.DataFrame | None:
    if "atom_i" not in group.columns:
        return None
    atom_i = pd.to_numeric(group["atom_i"], errors="coerce")
    if atom_i.isna().any():
        return None
    atom_i_int = atom_i.astype(np.int64)
    if not np.allclose(atom_i.to_numpy(dtype=np.float64), atom_i_int.to_numpy(dtype=np.float64)):
        return None
    out = group.copy()
    out["_cache_atom_i"] = atom_i_int
    out = out.sort_values("_cache_atom_i").drop_duplicates(subset="_cache_atom_i", keep="last")
    return out.sort_values("_cache_atom_i").reset_index(drop=True)


def _materialize_cached_rows(
    frames: list[AtomsFrame],
    cache_keys: list[str],
    cached_df: pd.DataFrame,
) -> tuple[pd.DataFrame, set[int]]:
    if cached_df.empty or "frame_cache_key" not in cached_df.columns:
        return pd.DataFrame(), set()

    by_key = {
        str(cache_key): group.reset_index(drop=True)
        for cache_key, group in cached_df.groupby("frame_cache_key", sort=False)
    }
    parts: list[pd.DataFrame] = []
    hit_indices: set[int] = set()

    for idx, (frame, cache_key) in enumerate(zip(frames, cache_keys)):
        cached_rows = by_key.get(cache_key)
        if cached_rows is None:
            continue
        canonical = _canonicalize_cache_rows(cached_rows)
        if canonical is None:
            continue

        atom_count = len(frame.atoms)
        expected_atom_i = np.arange(atom_count, dtype=np.int64)
        if len(canonical) != atom_count:
            continue
        if not np.array_equal(canonical["_cache_atom_i"].to_numpy(dtype=np.int64), expected_atom_i):
            continue

        remapped = canonical.drop(columns=["_cache_atom_i"]).copy()
        positions = np.asarray(frame.atoms.positions, dtype=np.float64)
        remapped["frame_cache_key"] = cache_key
        remapped["structure_id"] = frame.structure_id
        remapped["source_file"] = frame.source_file
        remapped["frame"] = frame.frame_index
        remapped["atom_i"] = expected_atom_i
        remapped["element"] = frame.atoms.get_chemical_symbols()
        remapped["x"] = positions[:, 0]
        remapped["y"] = positions[:, 1]
        remapped["z"] = positions[:, 2]
        parts.append(remapped)
        hit_indices.add(idx)

    if not parts:
        return pd.DataFrame(), set()
    return pd.concat(parts, ignore_index=True), hit_indices


def _prepare_rows_for_cache_store(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "frame_cache_key" not in df.columns:
        return df

    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("frame_cache_key", sort=False):
        canonical = _canonicalize_cache_rows(group)
        if canonical is None:
            parts.append(group)
            continue
        parts.append(canonical.drop(columns=["_cache_atom_i"]))

    if not parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(parts, ignore_index=True)


def _predict_miss_frames_dataframe(
    *,
    miss_frames: list[AtomsFrame],
    miss_keys: list[str],
    runtime_device: str,
    workers: int,
    chunk_size: int,
    committee: bool,
    property_mode: str,
    backend_factory: Callable[[str], Backend],
) -> pd.DataFrame:
    if not miss_frames:
        return pd.DataFrame()

    if runtime_device == "cpu" and workers > 1:
        chunk_pairs = list(chunked(list(zip(miss_frames, miss_keys)), chunk_size))
        if not chunk_pairs:
            return pd.DataFrame()

        thread_state = threading.local()

        def predict_chunk(
            chunk_index: int,
            pairs: list[tuple[AtomsFrame, str]],
        ) -> tuple[int, pd.DataFrame]:
            backend = getattr(thread_state, "backend", None)
            if backend is None:
                backend = backend_factory(runtime_device)
                thread_state.backend = backend

            frames = [frame for frame, _ in pairs]
            keys = [key for _, key in pairs]
            prediction = backend.predict(
                [frame.atoms for frame in frames],
                device=runtime_device,
                committee=committee,
                property=property_mode,
            )
            return (
                chunk_index,
                to_dataframe(
                    frames,
                    prediction,
                    property_mode=property_mode,
                    committee=committee,
                    frame_cache_keys=keys,
                ),
            )

        ordered_parts: list[pd.DataFrame | None] = [None] * len(chunk_pairs)
        max_workers = min(workers, len(chunk_pairs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(predict_chunk, chunk_index, pairs): chunk_index
                for chunk_index, pairs in enumerate(chunk_pairs)
            }
            for future in as_completed(futures):
                chunk_index, chunk_df = future.result()
                ordered_parts[chunk_index] = chunk_df

        parts = [part for part in ordered_parts if part is not None and not part.empty]
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    backend = backend_factory(runtime_device)
    prediction = backend.predict(
        [frame.atoms for frame in miss_frames],
        device=runtime_device,
        committee=committee,
        property=property_mode,
    )
    return to_dataframe(
        miss_frames,
        prediction,
        property_mode=property_mode,
        committee=committee,
        frame_cache_keys=miss_keys,
    )


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
    runtime_device = config.device
    setup_start = time.perf_counter()
    setup_complete = False

    try:
        resolved_inputs = resolve_inputs(inputs)
        loaded_frames = load_structures(resolved_inputs, frames=config.frames)
        frames, validation_warnings, normalized_cell_pbc = _validate_frames(loaded_frames, config.on_warning)
        all_warnings.extend(validation_warnings)

        runtime_device = _resolve_runtime_device(config.device)
        workers = _coerce_workers(runtime_device, config, all_warnings)
        runtime_workers = workers
        property_mode = _resolve_property_mode(config)
        output_format = resolve_output_format(config.output_format)

        if config.dry_run:
            timings.setup_time_s = time.perf_counter() - setup_start
            setup_complete = True
            return RunSummary(
                outdir=config.outdir,
                output_path=config.outdir / f"results.{output_format}",
                output_format=output_format,
                frame_count=len(frames),
                atom_count=count_atoms(frames),
                cache_hits=0,
                cache_misses=len(frames),
                device=runtime_device,
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
        setup_complete = True

        cache_start = time.perf_counter()
        cached_df = pd.DataFrame()
        hit_indices: set[int] = set()
        if cache_store is not None:
            unique_cache_keys = list(dict.fromkeys(cache_keys))
            cached_lookup_df = cache_store.lookup_many(unique_cache_keys)
            cached_df, hit_indices = _materialize_cached_rows(frames, cache_keys, cached_lookup_df)
        timings.cache_time_s = time.perf_counter() - cache_start

        miss_pairs = [
            (frame, key)
            for idx, (frame, key) in enumerate(zip(frames, cache_keys))
            if idx not in hit_indices
        ]

        predict_start = time.perf_counter()
        predicted_df = pd.DataFrame()
        if miss_pairs:
            miss_frames = [frame for frame, _ in miss_pairs]
            miss_keys = [key for _, key in miss_pairs]

            try:
                predicted_df = _predict_miss_frames_dataframe(
                    miss_frames=miss_frames,
                    miss_keys=miss_keys,
                    runtime_device=runtime_device,
                    workers=workers,
                    chunk_size=config.chunk_size,
                    committee=config.committee,
                    property_mode=property_mode,
                    backend_factory=backend_factory,
                )
            except ValidationError:
                raise
            except Exception as exc:
                raise BackendRuntimeError(f"Backend prediction failed: {exc}") from exc

            if cache_store is not None and not predicted_df.empty:
                cacheable_df = _prepare_rows_for_cache_store(predicted_df)
                cache_store.store_dataframe(
                    cacheable_df,
                    chunk_size=config.chunk_size,
                    max_chunk_mb=config.cache_chunk_max_mb,
                )

        timings.predict_time_s = time.perf_counter() - predict_start

        frames_df = (
            pd.concat([cached_df, predicted_df], ignore_index=True)
            if not cached_df.empty or not predicted_df.empty
            else pd.DataFrame()
        )
        if frames_df.empty:
            raise ValidationError("No results were produced")

        for col in required_columns(property_mode=property_mode, committee=config.committee):
            if col not in frames_df.columns:
                frames_df[col] = np.nan

        frames_df = frames_df.sort_values(["source_file", "frame", "atom_i"]).reset_index(drop=True)

        output_path = config.outdir / f"results.{output_format}"
        atomic_write_dataframe(output_path, frames_df, output_format)

        if config.magres_mode != "none":
            write_magres(frames_df, config.outdir, config.magres_mode, frames=frames)

        summary = RunSummary(
            outdir=config.outdir,
            output_path=output_path,
            output_format=output_format,
            frame_count=len(frames),
            atom_count=count_atoms(frames),
            cache_hits=len(hit_indices),
            cache_misses=len(miss_pairs),
            device=runtime_device,
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
            if not setup_complete:
                timings.setup_time_s = time.perf_counter() - setup_start
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
                    device=runtime_device,
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
