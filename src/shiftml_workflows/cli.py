"""Typer CLI for shiftml-workflows."""

from __future__ import annotations

from pathlib import Path

import json

import typer

from shiftml_workflows import __version__
from shiftml_workflows.config import ConfigError, build_predict_config
from shiftml_workflows.pipeline import (
    BackendRuntimeError,
    MissingDependencyError,
    PipelineError,
    ValidationError,
    run_predict,
)
from shiftml_workflows.schema import SCHEMA_VERSION

app = typer.Typer(help="Workflow-friendly ShiftML3 predictions")


def _print_json(payload: dict[str, object]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def info() -> None:
    """Show installed versions and backend details."""
    _print_json(
        {
            "tool": "shiftml-workflows",
            "version": __version__,
            "schema_version": SCHEMA_VERSION,
            "backend": "ShiftML3",
            "notes": [
                "Supported elements are validated before prediction.",
                "If --committee is set and --property is omitted, property resolves to iso.",
                "Output rows are sorted by source_file, frame, atom_i for deterministic diffs.",
            ],
        }
    )


@app.command()
def predict(
    inputs: list[str] = typer.Argument(..., help="Input structure files or glob patterns"),
    out: Path = typer.Option(..., "--out", help="Output directory"),
    frames: str = typer.Option(":", "--frames", help="Frame selector, e.g. :, 0:10:2, 5"),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda"),
    workers: int = typer.Option(1, "--workers", help="Number of workers"),
    chunk_size: int = typer.Option(200, "--chunk-size", help="Frames per chunk"),
    cache_chunk_max_mb: float = typer.Option(100.0, "--cache-chunk-max-mb", help="Max cache chunk size (MB)"),
    committee: bool = typer.Option(False, "--committee", help="Enable committee uncertainty"),
    property_mode: str | None = typer.Option(None, "--property", help="iso|tensor|both"),
    output_format: str = typer.Option("auto", "--format", help="auto|csv|parquet"),
    magres_mode: str = typer.Option("none", "--magres", help="none|per-frame|single"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Optional cache directory"),
    config: Path | None = typer.Option(None, "--config", help="JSON/YAML config file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print execution plan and exit"),
    on_warning: str = typer.Option("warn", "--on-warning", help="warn|error|skip"),
    strict: bool = typer.Option(False, "--strict", help="Alias for --on-warning error"),
    force_multi_gpu: bool = typer.Option(False, "--force-multi-gpu", help="Allow workers>1 on cuda"),
) -> None:
    """Run ShiftML3 predictions."""
    try:
        cfg = build_predict_config(
            outdir=out,
            config_path=config,
            cli_values={
                "frames": frames,
                "device": device,
                "workers": workers,
                "chunk_size": chunk_size,
                "cache_chunk_max_mb": cache_chunk_max_mb,
                "committee": committee,
                "property_mode": property_mode,
                "output_format": output_format,
                "magres_mode": magres_mode,
                "cache_dir": cache_dir,
                "dry_run": dry_run,
                "on_warning": on_warning,
                "strict": strict,
                "force_multi_gpu": force_multi_gpu,
            },
        )
        summary = run_predict(inputs=inputs, config=cfg)

        if cfg.dry_run:
            _print_json(
                {
                    "dry_run": True,
                    "outdir": str(summary.outdir),
                    "frame_count": summary.frame_count,
                    "atom_count": summary.atom_count,
                    "device": summary.device,
                    "workers": summary.workers,
                    "property_mode": summary.property_mode,
                    "committee_enabled": summary.committee_enabled,
                    "warnings": summary.warnings,
                }
            )
            raise typer.Exit(code=0)

        _print_json(
            {
                "status": "ok",
                "output": str(summary.output_path),
                "format": summary.output_format,
                "frames": summary.frame_count,
                "atoms": summary.atom_count,
                "cache_hits": summary.cache_hits,
                "cache_misses": summary.cache_misses,
                "device": summary.device,
                "workers": summary.workers,
                "committee_enabled": summary.committee_enabled,
                "timings": summary.timings.as_dict(),
                "warnings": summary.warnings,
            }
        )

    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=2)
    except ValidationError as exc:
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=3)
    except MissingDependencyError as exc:
        typer.echo(f"Missing dependency: {exc}", err=True)
        raise typer.Exit(code=4)
    except BackendRuntimeError as exc:
        typer.echo(f"Backend runtime error: {exc}", err=True)
        raise typer.Exit(code=5)
    except PipelineError as exc:
        typer.echo(f"Pipeline error: {exc}", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    app()
