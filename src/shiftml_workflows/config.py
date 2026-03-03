"""Configuration parsing and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import json


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


@dataclass(slots=True)
class PredictConfig:
    outdir: Path
    frames: str = ":"
    device: str = "auto"
    workers: int = 1
    chunk_size: int = 200
    cache_chunk_max_mb: float = 100.0
    committee: bool = False
    property_mode: Optional[str] = None
    output_format: str = "auto"
    magres_mode: str = "none"
    cache_dir: Optional[Path] = None
    config_path: Optional[Path] = None
    dry_run: bool = False
    on_warning: str = "warn"
    force_multi_gpu: bool = False
    command_args: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["outdir"] = str(self.outdir)
        data["cache_dir"] = str(self.cache_dir) if self.cache_dir is not None else None
        data["config_path"] = str(self.config_path) if self.config_path is not None else None
        return data


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("YAML config requested but PyYAML is not installed. Use JSON or install [yaml] extras.") from exc

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a mapping/object at top level")
    return data


def load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        return _load_yaml(path)

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a JSON object")
    return data


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def build_predict_config(
    *,
    outdir: Path,
    cli_values: dict[str, Any],
    config_path: Optional[Path] = None,
) -> PredictConfig:
    file_values: dict[str, Any] = {}
    if config_path is not None:
        file_values = load_config_file(config_path)

    def resolve(key: str, default: Any) -> Any:
        return _first_non_none(cli_values.get(key), file_values.get(key), default)

    on_warning = resolve("on_warning", "warn")
    strict = bool(resolve("strict", False))
    if strict:
        on_warning = "error"

    property_mode = _first_non_none(
        cli_values.get("property_mode"),
        file_values.get("property_mode"),
        file_values.get("property"),
        None,
    )
    output_format = _first_non_none(
        cli_values.get("output_format"),
        file_values.get("output_format"),
        file_values.get("format"),
        "auto",
    )
    magres_mode = _first_non_none(
        cli_values.get("magres_mode"),
        file_values.get("magres_mode"),
        file_values.get("magres"),
        "none",
    )

    cfg = PredictConfig(
        outdir=outdir,
        frames=str(resolve("frames", ":")),
        device=str(resolve("device", "auto")),
        workers=int(resolve("workers", 1)),
        chunk_size=int(resolve("chunk_size", 200)),
        cache_chunk_max_mb=float(resolve("cache_chunk_max_mb", 100.0)),
        committee=bool(resolve("committee", False)),
        property_mode=property_mode,
        output_format=str(output_format),
        magres_mode=str(magres_mode),
        cache_dir=Path(resolve("cache_dir", "")) if resolve("cache_dir", None) else None,
        config_path=config_path,
        dry_run=bool(resolve("dry_run", False)),
        on_warning=on_warning,
        force_multi_gpu=bool(resolve("force_multi_gpu", False)),
        command_args={k: v for k, v in cli_values.items() if v is not None},
    )
    validate_predict_config(cfg)
    return cfg


def validate_predict_config(cfg: PredictConfig) -> None:
    if cfg.device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("--device must be one of auto|cpu|cuda")
    if cfg.workers < 1:
        raise ConfigError("--workers must be >= 1")
    if cfg.chunk_size < 1:
        raise ConfigError("--chunk-size must be >= 1")
    if cfg.cache_chunk_max_mb <= 0:
        raise ConfigError("--cache-chunk-max-mb must be > 0")
    if cfg.output_format not in {"auto", "csv", "parquet"}:
        raise ConfigError("--format must be one of auto|csv|parquet")
    if cfg.magres_mode not in {"none", "per-frame", "single"}:
        raise ConfigError("--magres must be one of none|per-frame|single")
    if cfg.on_warning not in {"warn", "error", "skip"}:
        raise ConfigError("--on-warning must be one of warn|error|skip")
    if cfg.property_mode is not None and cfg.property_mode not in {"iso", "tensor", "both"}:
        raise ConfigError("--property must be one of iso|tensor|both")
