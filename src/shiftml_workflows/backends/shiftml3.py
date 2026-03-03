"""ShiftML3 backend implementation.

This is intentionally the only module importing `shiftml`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from ase import Atoms
from shiftml.ase import ShiftML

from shiftml_workflows.backends.base import FramePrediction, PredictionResult


class ShiftML3Backend:
    """ShiftML3 backend wrapper with lazy model initialization."""

    name = "ShiftML3"

    def __init__(self, device: str = "auto") -> None:
        self._device = self._normalize_device(device)
        self._model: Optional[Any] = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
        except Exception:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @classmethod
    def _normalize_device(cls, device: str | None) -> str:
        if device in {None, "auto"}:
            return "cuda" if cls._cuda_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be one of auto|cpu|cuda")
        return device

    def _model_instance(self) -> Any:
        if self._model is None:
            self._model = ShiftML("ShiftML3", device=self._device)
        return self._model

    def predict(
        self,
        frames: list[Atoms],
        *,
        device: str = "auto",
        committee: bool = False,
        property: str = "iso",
    ) -> PredictionResult:
        resolved_device = self._normalize_device(device)
        model = (
            self._model_instance()
            if resolved_device == self._device
            else ShiftML("ShiftML3", device=resolved_device)
        )
        out_frames: list[FramePrediction] = []

        for atoms in frames:
            atom_count = len(atoms)
            iso = self._predict_iso(model, atoms)
            if iso.shape[0] != atom_count:
                raise RuntimeError("ShiftML returned unexpected iso output length")

            iso_unc = None
            if committee:
                iso_unc = self._predict_iso_uncertainty(model, atoms, atom_count)

            tensor = None
            tensor_unc = None
            if property in {"tensor", "both"}:
                tensor = self._predict_tensor(model, atoms, atom_count)
                if committee and tensor is not None:
                    tensor_unc = self._predict_tensor_uncertainty(model, atoms, atom_count)

            out_frames.append(
                FramePrediction(
                    cs_iso=iso,
                    cs_iso_uncertainty=iso_unc,
                    tensor=tensor,
                    tensor_uncertainty=tensor_unc,
                )
            )

        return PredictionResult(frames=out_frames)

    @staticmethod
    def _predict_iso(model: Any, atoms: Atoms) -> np.ndarray:
        if hasattr(model, "get_cs_iso"):
            return np.asarray(model.get_cs_iso(atoms), dtype=np.float64)
        if hasattr(model, "get_cs"):
            return np.asarray(model.get_cs(atoms), dtype=np.float64)
        raise RuntimeError("ShiftML backend does not expose isotropic prediction API")

    @staticmethod
    def _predict_iso_uncertainty(model: Any, atoms: Atoms, atom_count: int) -> np.ndarray:
        if hasattr(model, "get_cs_iso_ensemble"):
            ensemble = np.asarray(model.get_cs_iso_ensemble(atoms), dtype=np.float64)
            if ensemble.ndim == 2 and ensemble.shape[0] == atom_count:
                return ensemble.std(axis=-1)
            if ensemble.ndim == 2 and ensemble.shape[1] == atom_count:
                return ensemble.std(axis=0)
        if hasattr(model, "get_cs_committee"):
            committee_preds = np.asarray(model.get_cs_committee(atoms), dtype=np.float64)
            if committee_preds.ndim == 2 and committee_preds.shape[0] == atom_count:
                return committee_preds.std(axis=-1)
            if committee_preds.ndim == 2 and committee_preds.shape[1] == atom_count:
                return committee_preds.std(axis=0)
        return np.zeros(atom_count, dtype=np.float64)

    @staticmethod
    def _predict_tensor(model: Any, atoms: Atoms, atom_count: int) -> np.ndarray | None:
        if hasattr(model, "get_cs_tensor"):
            tensor = np.asarray(model.get_cs_tensor(atoms), dtype=np.float64)
            if tensor.shape == (atom_count, 3, 3):
                return tensor
        if hasattr(model, "get_tensor"):
            tensor = np.asarray(model.get_tensor(atoms), dtype=np.float64)
            if tensor.shape == (atom_count, 3, 3):
                return tensor
        return None

    @staticmethod
    def _predict_tensor_uncertainty(model: Any, atoms: Atoms, atom_count: int) -> np.ndarray | None:
        if hasattr(model, "get_cs_tensor_ensemble"):
            ensemble = np.asarray(model.get_cs_tensor_ensemble(atoms), dtype=np.float64)
            if ensemble.ndim == 4 and ensemble.shape[0] == atom_count and ensemble.shape[1:3] == (3, 3):
                return ensemble.std(axis=-1)
            if ensemble.ndim == 4 and ensemble.shape[1:] == (atom_count, 3, 3):
                return ensemble.std(axis=0)
        return None
