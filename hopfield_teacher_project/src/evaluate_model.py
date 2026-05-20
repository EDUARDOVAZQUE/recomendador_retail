from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from hopfield import HopfieldModel, to_bipolar


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


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def binarize(df: pd.DataFrame, threshold: float) -> np.ndarray:
    return (df[FEATURES].to_numpy(dtype=float) >= threshold).astype(int)


def corrupt_bits(pattern: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    x = pattern.copy()
    mask = rng.random(len(x)) < noise
    x[mask] = 1 - x[mask]
    return x


def confusion_matrix(labels_true, labels_pred, class_names):
    idx = {c: i for i, c in enumerate(class_names)}
    m = np.zeros((len(class_names), len(class_names)), dtype=int)
    for t, p in zip(labels_true, labels_pred):
        m[idx[t], idx[p]] += 1
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el modelo Hopfield con ruido sobre el dataset.")
    parser.add_argument("--data", type=str, default="data/dataset_docente.csv", help="CSV del dataset")
    parser.add_argument("--model", type=str, default="model/hopfield_docente.npz", help="Archivo del modelo")
    parser.add_argument("--noise", type=float, default=0.20, help="Probabilidad de voltear cada bit")
    parser.add_argument("--out", type=str, default="results/reporte_evaluacion.csv", help="Ruta del reporte")
    parser.add_argument("--seed", type=int, default=123, help="Semilla aleatoria")
    args = parser.parse_args()

    df = load_dataset(args.data)
    model = HopfieldModel.load(args.model)
    rng = np.random.default_rng(args.seed)

    X = binarize(df, model.threshold_value)
    y_true = df["nivel_desempeno"].tolist()

    preds = []
    steps_list = []

    for row in X:
        noisy = corrupt_bits(row, args.noise, rng)
        noisy_bipolar = to_bipolar(noisy)
        result = model.classify(noisy_bipolar)
        preds.append(result["class_name"])
        steps_list.append(result["steps"])

    acc = float(np.mean(np.array(y_true) == np.array(preds)))
    cm = confusion_matrix(y_true, preds, model.class_names)

    report = pd.DataFrame(
        {
            "real": y_true,
            "predicho": preds,
            "pasos_reconstruccion": steps_list,
        }
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Exactitud global: {acc:.4f}")
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(pd.DataFrame(cm, index=model.class_names, columns=model.class_names).to_string())
    print(f"Reporte guardado en: {out.resolve()}")

    summary = Counter(preds)
    print("Distribución de predicciones:", dict(summary))


if __name__ == "__main__":
    main()
