from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def ejecutar(comando):
    print("\nEjecutando:", comando)
    os.system(comando)


while True:
    print("\n==============================")
    print("   RED DE HOPFIELD - DOCENTES")
    print("==============================")
    print("1. Generar dataset sintético")
    print("2. Entrenar modelo")
    print("3. Evaluar modelo")
    print("4. Probar docente manualmente")
    print("5. Salir")

    opcion = input("\nSelecciona una opción: ")

    if opcion == "1":

        n = input("Cantidad de registros [400]: ").strip()

        if n == "":
            n = "400"

        comando = (
            f'python "{BASE_DIR}\\generate_dataset.py" '
            f'--n {n} --seed 7'
        )

        ejecutar(comando)

    elif opcion == "2":

        comando = f'python "{BASE_DIR}\\train_model.py"'

        ejecutar(comando)

    elif opcion == "3":

        ruido = input("Nivel de ruido [0.20]: ").strip()

        if ruido == "":
            ruido = "0.20"

        comando = (
            f'python "{BASE_DIR}\\evaluate_model.py" '
            f'--noise {ruido}'
        )

        ejecutar(comando)

    elif opcion == "4":

        print("\nIngresa valores entre 0 y 1")

        puntualidad = float(input("Puntualidad: "))
        dominio = float(input("Dominio del tema: "))
        comunicacion = float(input("Comunicación: "))
        evaluacion = float(input("Evaluación justa: "))
        tecnologia = float(input("Uso de tecnología: "))
        participacion = float(input("Participación estudiantil: "))

        comando = (
            f'python "{BASE_DIR}\\predict.py" '
            f'--puntualidad {puntualidad} '
            f'--dominio {dominio} '
            f'--comunicacion {comunicacion} '
            f'--evaluacion {evaluacion} '
            f'--tecnologia {tecnologia} '
            f'--participacion {participacion}'
        )

        ejecutar(comando)

    elif opcion == "5":

        print("\nSaliendo del sistema...")
        break

    else:

        print("\nOpción inválida")