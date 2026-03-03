"""Backend interface and shared prediction types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from ase import Atoms


@dataclass(slots=True)
class FramePrediction:
    """Predictions for a single structure frame."""

    cs_iso: np.ndarray
    cs_iso_uncertainty: Optional[np.ndarray] = None
    tensor: Optional[np.ndarray] = None
    tensor_uncertainty: Optional[np.ndarray] = None


@dataclass(slots=True)
class PredictionResult:
    """Predictions for a list of frames."""

    frames: list[FramePrediction]


class Backend(Protocol):
    """Prediction backend protocol."""

    name: str

    def predict(
        self,
        frames: list[Atoms],
        *,
        device: str = "auto",
        committee: bool = False,
        property: str = "iso",
    ) -> PredictionResult:
        """Predict chemical shifts for a list of ASE atoms frames."""
