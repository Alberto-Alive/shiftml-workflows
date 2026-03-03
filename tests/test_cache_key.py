from __future__ import annotations

from shiftml_workflows.cache import compute_cache_key


def test_cache_key_stable_for_identical_inputs() -> None:
    key1 = compute_cache_key(
        model_name="ShiftML3",
        model_version="1.0",
        schema_version="1",
        flags={"committee": False, "property": "iso"},
        structure_id="abc",
    )
    key2 = compute_cache_key(
        model_name="ShiftML3",
        model_version="1.0",
        schema_version="1",
        flags={"committee": False, "property": "iso"},
        structure_id="abc",
    )
    assert key1 == key2


def test_cache_key_changes_when_flags_change() -> None:
    key1 = compute_cache_key(
        model_name="ShiftML3",
        model_version="1.0",
        schema_version="1",
        flags={"committee": False, "property": "iso"},
        structure_id="abc",
    )
    key2 = compute_cache_key(
        model_name="ShiftML3",
        model_version="1.0",
        schema_version="1",
        flags={"committee": True, "property": "iso"},
        structure_id="abc",
    )
    assert key1 != key2
