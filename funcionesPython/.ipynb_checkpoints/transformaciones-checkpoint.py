import math
import pandas as pd

def normalizar(x):
    if not all(isinstance(valor, (int, float)) and not isinstance(valor, bool) for valor in x):
        raise ValueError("La variable debe ser numérica.")

    if len(x) == 0:
        return []

    minimo = min(x)
    maximo = max(x)
    if maximo == minimo:
        return [0.0] * len(x)
    x_normalizado = []
    denominador = maximo - minimo

    for valor in x:
        x_normalizado.append((valor - minimo) / denominador)

    return x_normalizado


def estandarizar(x):
    if not all(isinstance(valor, (int, float)) and not isinstance(valor, bool) for valor in x):
        raise ValueError("La variable debe ser numérica.")

    n = len(x)
    if n <= 1:
        return [0.0] * n
    media = sum(x) / n
    desviacion = math.sqrt(sum((valor - media) ** 2 for valor in x) / (n - 1))
    if desviacion == 0:
        return [0.0] * n
    x_estandarizado = []
    for valor in x:
        x_estandarizado.append((valor - media) / desviacion)

    return x_estandarizado


def transformar_dataset(dataset, metodo):
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("El dataset debe ser un DataFrame.")

    if metodo not in ["normalizacion", "estandarizacion"]:
        raise ValueError("El método debe ser 'normalizacion' o 'estandarizacion'.")
    resultado = dataset.copy()
    for nombre in dataset.columns:
        if pd.api.types.is_numeric_dtype(dataset[nombre]):

            vector_datos = dataset[nombre].tolist()

            if metodo == "normalizacion":
                resultado[nombre] = normalizar(vector_datos)

            elif metodo == "estandarizacion":
                resultado[nombre] = estandarizar(vector_datos)

    return resultado