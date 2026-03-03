from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write

from shiftml_workflows.io import InputError, load_structures


def _make_frames() -> list[Atoms]:
    return [
        Atoms("HCO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        Atoms("HCO", positions=[[0.1, 0.0, 0.0], [0.0, 0.1, 1.0], [0.0, 1.1, 0.0]]),
    ]


def test_load_structures_xyz_multiple_frames(tmp_path: Path) -> None:
    xyz_path = tmp_path / "frames.xyz"
    write(xyz_path, _make_frames())

    loaded = load_structures([str(xyz_path)], frames=":")
    assert len(loaded) == 2
    assert loaded[0].frame_index == 0
    assert loaded[1].frame_index == 1


def test_load_structures_extxyz_frame_selector(tmp_path: Path) -> None:
    extxyz_path = tmp_path / "frames.extxyz"
    write(extxyz_path, _make_frames())

    loaded = load_structures([str(extxyz_path)], frames="1")
    assert len(loaded) == 1
    assert loaded[0].frame_index == 1


def test_load_structures_invalid_selector_raises(tmp_path: Path) -> None:
    xyz_path = tmp_path / "frames.xyz"
    write(xyz_path, _make_frames())

    try:
        load_structures([str(xyz_path)], frames="bad")
    except InputError:
        pass
    else:
        raise AssertionError("Expected InputError for invalid frame selector")
