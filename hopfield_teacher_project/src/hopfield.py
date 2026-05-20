from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def to_bipolar(x: np.ndarray) -> np.ndarray:
    """Convierte vectores {0,1} a {-1,+1}."""
    x = np.asarray(x, dtype=int)
    return np.where(x > 0, 1, -1).astype(int)


def from_bipolar(x: np.ndarray) -> np.ndarray:
    """Convierte vectores {-1,+1} a {0,1}."""
    x = np.asarray(x, dtype=int)
    return np.where(x > 0, 1, 0).astype(int)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    return int(np.sum(a != b))


@dataclass
class HopfieldModel:
    weights: np.ndarray
    thresholds: np.ndarray
    class_names: List[str]
    class_prototypes: np.ndarray
    feature_names: List[str]
    threshold_value: float

    @property
    def n_neurons(self) -> int:
        return int(self.weights.shape[0])

    def recall(self, pattern: np.ndarray, max_iter: int = 20) -> Tuple[np.ndarray, int]:
        """Recupera un patrón usando actualización asíncrona."""
        x = np.asarray(pattern, dtype=int).copy()
        for step in range(1, max_iter + 1):
            prev = x.copy()
            for i in range(self.n_neurons):
                net = float(np.dot(self.weights[i], x) - self.thresholds[i])
                x[i] = 1 if net >= 0 else -1
            if np.array_equal(x, prev):
                return x, step
        return x, max_iter

    def classify(self, pattern_bipolar: np.ndarray, max_iter: int = 20) -> Dict[str, object]:
        recalled, steps = self.recall(pattern_bipolar, max_iter=max_iter)
        distances = [
            hamming_distance(recalled, proto)
            for proto in self.class_prototypes
        ]
        idx = int(np.argmin(distances))
        return {
            "class_name": self.class_names[idx],
            "class_index": idx,
            "recalled_pattern": recalled,
            "steps": steps,
            "distances": distances,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.weights,
            thresholds=self.thresholds,
            class_names=np.array(self.class_names, dtype=object),
            class_prototypes=self.class_prototypes,
            feature_names=np.array(self.feature_names, dtype=object),
            threshold_value=np.array(self.threshold_value, dtype=float),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HopfieldModel":
        data = np.load(path, allow_pickle=True)
        return cls(
            weights=data["weights"],
            thresholds=data["thresholds"],
            class_names=list(data["class_names"].tolist()),
            class_prototypes=data["class_prototypes"],
            feature_names=list(data["feature_names"].tolist()),
            threshold_value=float(data["threshold_value"]),
        )


def train_hopfield(patterns_bipolar: np.ndarray) -> np.ndarray:
    """Entrena la red con la regla hebbiana clásica."""
    patterns_bipolar = np.asarray(patterns_bipolar, dtype=int)
    n_patterns, n_neurons = patterns_bipolar.shape
    w = np.zeros((n_neurons, n_neurons), dtype=float)

    for p in patterns_bipolar:
        w += np.outer(p, p)

    w /= n_neurons
    np.fill_diagonal(w, 0.0)
    return w


def build_model(
    class_prototypes_binary: np.ndarray,
    class_names: List[str],
    feature_names: List[str],
    threshold_value: float,
) -> HopfieldModel:
    prototypes_bipolar = to_bipolar(class_prototypes_binary)
    weights = train_hopfield(prototypes_bipolar)
    thresholds = np.zeros(weights.shape[0], dtype=float)
    return HopfieldModel(
        weights=weights,
        thresholds=thresholds,
        class_names=class_names,
        class_prototypes=prototypes_bipolar,
        feature_names=feature_names,
        threshold_value=threshold_value,
    )


def energy(weights: np.ndarray, state_bipolar: np.ndarray) -> float:
    x = np.asarray(state_bipolar, dtype=float)
    return float(-0.5 * x.T @ weights @ x)
