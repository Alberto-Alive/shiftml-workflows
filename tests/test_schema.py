from __future__ import annotations

import numpy as np
from ase import Atoms

from shiftml_workflows.backends.base import FramePrediction, PredictionResult
from shiftml_workflows.io import AtomsFrame, compute_structure_id
from shiftml_workflows.schema import required_columns, to_dataframe


def test_schema_required_columns_iso_mode() -> None:
    atoms = Atoms("HCO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1], [0.0, 1.0, 0.0]])
    sid, normalized = compute_structure_id(atoms)
    frame = AtomsFrame(atoms=atoms, source_file="a.xyz", frame_index=0, structure_id=sid, normalized_cell_pbc=normalized)

    prediction = PredictionResult(frames=[FramePrediction(cs_iso=np.array([1.0, 2.0, 3.0]))])
    df = to_dataframe([frame], prediction, property_mode="iso", committee=False)

    for col in required_columns("iso", committee=False):
        assert col in df.columns
