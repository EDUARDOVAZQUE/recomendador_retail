from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hopfield import build_model, from_bipolar


FEATURES = [
    "puntualidad",
    "planificacion",
    "dominio_contenido",
    "participacion",
    "evaluacion",
    "retroalimentacion",
    "uso_tecnologia",
    "gestion_aula",
    "comunicacion",
    "empatia",
    "asistencia",
    "innovacion",
]

CLASS_NAMES = ["Sobresaliente", "Bueno", "Regular", "Insuficiente"]


def binarize_features(df: pd.DataFrame, threshold: float = 60.0) -> np.ndarray:
    x = df[FEATURES].to_numpy(dtype=float)
    return (x >= threshold).astype(int)


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y guarda el modelo Hopfield.")
    parser.add_argument("--data", type=str, default="data/dataset_docente.csv", help="CSV del dataset")
    parser.add_argument("--model", type=str, default="model/hopfield_docente.npz", help="Archivo del modelo")
    parser.add_argument("--threshold", type=float, default=60.0, help="Umbral para binarizar indicadores")
    args = parser.parse_args()

    df = load_dataset(args.data)
    binary = binarize_features(df, threshold=args.threshold)

    prototypes = []
    for label in CLASS_NAMES:
        subset = df[df["nivel_desempeno"] == label]
        bits = binarize_features(subset, threshold=args.threshold)
        proto = (bits.mean(axis=0) >= 0.5).astype(int)
        prototypes.append(proto)

    prototypes = np.asarray(prototypes, dtype=int)
    model = build_model(
        class_prototypes_binary=prototypes,
        class_names=CLASS_NAMES,
        feature_names=FEATURES,
        threshold_value=args.threshold,
    )
    model.save(args.model)

    print(f"Modelo guardado en: {Path(args.model).resolve()}")
    print("Prototipos binarios por clase:")
    for label, proto in zip(CLASS_NAMES, prototypes):
        print(f"- {label}: {proto.tolist()}")


if __name__ == "__main__":
    main()
