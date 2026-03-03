"""ShiftML workflows package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shiftml-workflows")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
