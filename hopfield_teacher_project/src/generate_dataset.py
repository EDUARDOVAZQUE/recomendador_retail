from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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

# Prototipos binarios por nivel de desempeño.
# 1 = presencia fuerte del indicador; 0 = presencia débil.
PROTOTYPES = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Sobresaliente
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],  # Bueno
    [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0],  # Regular
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Insuficiente
], dtype=int)


def _sample_continuous_from_bits(bits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Indicadores altos para 1 y bajos para 0; luego se añaden pequeñas variaciones.
    high = rng.normal(loc=85, scale=8, size=bits.shape[0])
    low = rng.normal(loc=30, scale=10, size=bits.shape[0])
    x = np.where(bits == 1, high, low)
    return np.clip(x, 0, 100)


def generate_dataset(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    n_classes = len(CLASS_NAMES)
    per_class = [n // n_classes] * n_classes
    for i in range(n % n_classes):
        per_class[i] += 1

    for class_index, (label, prototype, count) in enumerate(zip(CLASS_NAMES, PROTOTYPES, per_class)):
        for _ in range(count):
            # Ruido bit a bit sobre el prototipo.
            noisy_bits = prototype.copy()
            flip_mask = rng.random(len(prototype)) < rng.uniform(0.05, 0.20)
            noisy_bits[flip_mask] = 1 - noisy_bits[flip_mask]

            scores = _sample_continuous_from_bits(noisy_bits, rng)
            score_global = float(np.mean(scores))

            row = {f: round(float(v), 2) for f, v in zip(FEATURES, scores)}
            row["nivel_desempeno"] = label
            row["codigo_clase"] = class_index
            row["score_global"] = round(score_global, 2)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un dataset sintético de desempeño docente.")
    parser.add_argument("--n", type=int, default=400, help="Número total de registros")
    parser.add_argument("--seed", type=int, default=7, help="Semilla aleatoria")
    parser.add_argument("--out", type=str, default="data/dataset_docente.csv", help="Ruta de salida CSV")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.n, args.seed)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Dataset generado: {out.resolve()}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
