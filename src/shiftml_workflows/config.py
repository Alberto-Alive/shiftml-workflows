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
        def _json_safe(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): _json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_json_safe(v) for v in value]
            return value

        data = asdict(self)
        return _json_safe(data)


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


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in {0, 1}:
            return bool(value)
        raise ConfigError(f"{field_name} must be a boolean")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"{field_name} must be a boolean")
    raise ConfigError(f"{field_name} must be a boolean")


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

    on_warning = str(resolve("on_warning", "warn")).strip().lower()
    strict = _coerce_bool(resolve("strict", False), field_name="strict")
    if strict:
        on_warning = "error"

    property_mode = _first_non_none(
        cli_values.get("property_mode"),
        file_values.get("property_mode"),
        file_values.get("property"),
        None,
    )
    if property_mode is not None:
        property_mode = str(property_mode).strip().lower()
    output_format = _first_non_none(
        cli_values.get("output_format"),
        file_values.get("output_format"),
        file_values.get("format"),
        "auto",
    )
    output_format = str(output_format).strip().lower()
    magres_mode = _first_non_none(
        cli_values.get("magres_mode"),
        file_values.get("magres_mode"),
        file_values.get("magres"),
        "none",
    )
    magres_mode = str(magres_mode).strip().lower()
    cache_dir_value = resolve("cache_dir", None)

    cfg = PredictConfig(
        outdir=outdir,
        frames=str(resolve("frames", ":")),
        device=str(resolve("device", "auto")).strip().lower(),
        workers=int(resolve("workers", 1)),
        chunk_size=int(resolve("chunk_size", 200)),
        cache_chunk_max_mb=float(resolve("cache_chunk_max_mb", 100.0)),
        committee=_coerce_bool(resolve("committee", False), field_name="committee"),
        property_mode=property_mode,
        output_format=output_format,
        magres_mode=magres_mode,
        cache_dir=Path(cache_dir_value) if cache_dir_value else None,
        config_path=config_path,
        dry_run=_coerce_bool(resolve("dry_run", False), field_name="dry_run"),
        on_warning=on_warning,
        force_multi_gpu=_coerce_bool(resolve("force_multi_gpu", False), field_name="force_multi_gpu"),
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
