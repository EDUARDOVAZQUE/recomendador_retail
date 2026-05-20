import argparse
import numpy as np
import os


class RedHopfield:

    def __init__(self):

        self.weights = None
        self.class_names = None
        self.class_prototypes = None

    def cargar_modelo(self, ruta):

        datos = np.load(ruta, allow_pickle=True)

        self.weights = datos["weights"]
        self.class_names = datos["class_names"]
        self.class_prototypes = datos["class_prototypes"]

    def recuperar(self, patron, iteraciones=15):

        estado = patron.copy()

        n = len(estado)

        for _ in range(iteraciones):

            nuevo_estado = estado.copy()

            for i in range(n):

                suma = np.dot(
                    self.weights[i],
                    estado
                )

                if suma >= 0:
                    nuevo_estado[i] = 1
                else:
                    nuevo_estado[i] = -1

            estado = nuevo_estado

        return estado

    def clasificar(self, patron):

        mejor_clase = None
        menor_distancia = float("inf")

        for i, prototipo in enumerate(self.class_prototypes):

            distancia = np.sum(
                patron != prototipo
            )

            if distancia < menor_distancia:

                menor_distancia = distancia
                mejor_clase = self.class_names[i]

        return mejor_clase


def bipolar(valor):

    if valor >= 0.5:
        return 1

    return -1


parser = argparse.ArgumentParser()

parser.add_argument("--puntualidad", type=float, required=True)
parser.add_argument("--dominio", type=float, required=True)
parser.add_argument("--comunicacion", type=float, required=True)
parser.add_argument("--evaluacion", type=float, required=True)
parser.add_argument("--tecnologia", type=float, required=True)
parser.add_argument("--participacion", type=float, required=True)

args = parser.parse_args()

# 6 variables base
base = np.array([
    bipolar(args.puntualidad),
    bipolar(args.dominio),
    bipolar(args.comunicacion),
    bipolar(args.evaluacion),
    bipolar(args.tecnologia),
    bipolar(args.participacion)
])

# duplicar para llegar a 12 neuronas
entrada = np.concatenate([base, base])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ruta_modelo = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "hopfield_docente.npz"
)

red = RedHopfield()

red.cargar_modelo(ruta_modelo)

resultado = red.recuperar(entrada)

clase = red.clasificar(resultado)

print("\nEntrada original:")
print(entrada)

print("\nPatrón recuperado:")
print(resultado)

print("\nClasificación final:")
print(clase)