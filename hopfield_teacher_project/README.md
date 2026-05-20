# Proyecto escolar: evaluación del desempeño docente con Red de Hopfield

Este proyecto usa una red de Hopfield clásica como memoria asociativa para reconocer perfiles docentes sintéticos y clasificar su desempeño en cuatro niveles:

- Sobresaliente
- Bueno
- Regular
- Insuficiente

## Qué incluye

- **A) Dataset sintético**: se genera desde Python con ruido controlado.
- **B) Archivo del modelo**: se guarda en `.npz`.
- **C) Evaluación del archivo del modelo**: calcula exactitud, matriz de confusión y reporte por clase.
- **D) Aplicación básica para prueba del modelo**: menú por consola.

## Estructura

```text
src/
  hopfield.py
  generate_dataset.py
  train_model.py
  evaluate_model.py
  app_cmd.py
data/
model/
results/
```

## Cómo ejecutar desde CMD

```bat
cd rutal\proyecto
python -m pip install -r requirements.txt
python src\generate_dataset.py --n 400 --seed 7
python src	rain_model.py
python src\evaluate_model.py --noise 0.20
python srcpp_cmd.py
```

## Idea técnica

Cada docente se representa con 12 indicadores binarios:

- puntualidad
- planificación
- dominio del contenido
- participación del grupo
- evaluación
- retroalimentación
- uso de tecnología
- gestión de aula
- comunicación
- empatía
- asistencia
- innovación

La red de Hopfield memoriza prototipos de cada nivel de desempeño y, cuando recibe una versión ruidosa, intenta reconstruir el patrón más cercano.

## Nota académica

Este proyecto es demostrativo. Usa una versión clásica de Hopfield, apropiada para memorias asociativas pequeñas y fáciles de explicar en clase.
