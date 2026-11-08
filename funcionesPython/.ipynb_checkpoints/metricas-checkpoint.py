import math
from collections import Counter
import numpy as np
import pandas as pd

def calcular_varianza(x):
    n = len(x)
    if n <= 1:
        return None
    media = sum(x) / n
    varianza = sum((valor - media) ** 2 for valor in x) / (n - 1)
    return varianza


def calcular_entropia(x):
    n = len(x)
    if n <= 1:
        return "No hay elementos"
    frecuencias = Counter(x)
    probabilidades = []

    for frecuencia in frecuencias.values():
        probabilidades.append(frecuencia / n)
    entropia = -sum(p * math.log2(p) for p in probabilidades)
    return entropia


def calcular_auc(x, clase):
    if not all(isinstance(valor, (int, float)) and not isinstance(valor, bool) for valor in x):
        raise ValueError("x debe ser numérico.")

    if len(x) != len(clase):
        raise ValueError("x y clase deben tener la misma longitud.")

    if any(valor is None or (isinstance(valor, float) and math.isnan(valor)) for valor in x + clase):
        raise ValueError("x y clase no pueden contener valores NA.")

    clases_unicas = []
    for c in clase:
        if c not in clases_unicas:
            clases_unicas.append(c)

    if len(clases_unicas) != 2:
        raise ValueError("La variable clase debe ser binaria (2 valores distintos).")

    neg_class = clases_unicas[0]
    pos_class = clases_unicas[1]

    positivos = [x[i] for i in range(len(x)) if clase[i] == pos_class]
    negativos = [x[i] for i in range(len(x)) if clase[i] == neg_class]

    n_positivos = len(positivos)
    n_negativos = len(negativos)

    if n_positivos == 0 or n_negativos == 0:
        raise ValueError("La clase debe contener observaciones de ambas clases.")

    suma = 0.0
    for p in positivos:
        for n in negativos:
            if p > n:
                suma += 1.0
            elif p == n:
                suma += 0.5

    auc = suma / (n_positivos * n_negativos)

    return auc


def calcular_roc(x, clase):
    if not all(isinstance(valor, (int, float)) and not isinstance(valor, bool) for valor in x):
        raise ValueError("x debe ser numérico.")

    if len(x) != len(clase):
        raise ValueError("x y clase deben tener la misma longitud.")

    if any(valor is None or (isinstance(valor, float) and math.isnan(valor)) for valor in x + clase):
        raise ValueError("x y clase no pueden contener valores NA.")

    clases_unicas = []
    for c in clase:
        if c not in clases_unicas:
            clases_unicas.append(c)

    if len(clases_unicas) != 2:
        raise ValueError("La variable clase debe ser binaria (2 valores distintos).")

    positivos = [x[i] for i in range(len(x)) if clase[i] == clases_unicas[1]]
    negativos = [x[i] for i in range(len(x)) if clase[i] == clases_unicas[0]]

    n_positivos = len(positivos)
    n_negativos = len(negativos)

    if n_positivos == 0 or n_negativos == 0:
        raise ValueError("La clase debe contener observaciones de ambas clases.")

    umbrales = np.sort(np.unique(x))[::-1]
    umbrales = np.append(umbrales, -np.inf)
    umbrales = np.insert(umbrales, 0, np.inf)

    tpr = np.zeros(len(umbrales))
    fpr = np.zeros(len(umbrales))

    for i, umbral in enumerate(umbrales):
        verdaderos_positivos = np.sum(positivos >= umbral)
        falsos_positivos = np.sum(negativos >= umbral)

        tpr[i] = verdaderos_positivos / n_positivos
        fpr[i] = falsos_positivos / n_negativos

    resultado = pd.DataFrame({
        "Umbral": umbrales,
        "FPR": fpr,
        "TPR": tpr
    })

    return resultado


def calcular_metricas_dataset(dataset, var_clase):
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("El dataset debe ser un pandas DataFrame.")

    if var_clase not in dataset.columns:
        raise ValueError("La variable clase no existe en el dataset.")

    vector_clase = dataset[var_clase].tolist()
    resultados = []

    for nombre in dataset.columns:
        if nombre == var_clase:
            continue

        columna = dataset[nombre]
        x = columna.tolist()
        if pd.api.types.is_numeric_dtype(columna):
            resultados.append({
                "Atributo": nombre,
                "Tipo": "Continua",
                "Varianza": calcular_varianza(x),
                "AUC": calcular_auc(x, vector_clase),
                "Entropia": None
            })
        else:
            resultados.append({
                "Atributo": nombre,
                "Tipo": "Discreta",
                "Varianza": None,
                "AUC": None,
                "Entropia": calcular_entropia(x)
            })

    return pd.DataFrame(resultados)