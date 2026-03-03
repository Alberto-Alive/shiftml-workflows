"""Simple chunking helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

T = TypeVar("T")


def chunked(items: list[T], chunk_size: int) -> Iterator[list[T]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]
