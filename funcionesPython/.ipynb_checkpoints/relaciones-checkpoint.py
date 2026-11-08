from collections import Counter
import math
import numpy as np
import pandas as pd

def calcular_correlacion(x, y):
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x + y):
        raise ValueError("Las dos variables deben ser numéricas.")

    if len(x) != len(y):
        raise ValueError("Las variables deben tener la misma longitud.")

    if len(x) < 2:
        raise ValueError("Se necesitan al menos dos observaciones.")

    for v in x + y:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            raise ValueError("Las variables no pueden contener valores NA.")

    n = len(x)

    media_x = sum(x) / n
    media_y = sum(y) / n

    numerador = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n))
    denominador = math.sqrt(sum((v - media_x) ** 2 for v in x) * sum((v - media_y) ** 2 for v in y))

    if denominador == 0:
        return None

    resultado = numerador / denominador

    return resultado

def calcular_informacion_mutua(x, y):
    if len(x) != len(y):
        raise ValueError("Las variables deben tener la misma longitud.")

    if len(x) == 0:
        raise ValueError("Las variables no pueden estar vacías.")

    for valor in x + y:
        if valor is None or (isinstance(valor, float) and math.isnan(valor)):
            raise ValueError("Las variables no pueden contener valores NA.")

    n = len(x)

    frecuencias_x = Counter(x)
    frecuencias_y = Counter(y)
    
    px = {}
    py = {}

    for categoria in frecuencias_x:
        px[categoria] = frecuencias_x[categoria] / n

    for categoria in frecuencias_y:
        py[categoria] = frecuencias_y[categoria] / n

    frecuencias_conjuntas = Counter()
    for i in range(n):
        pareja = (x[i], y[i])
        frecuencias_conjuntas[pareja] += 1

    informacion = 0.0

    for pareja in frecuencias_conjuntas:
        categoria_x = pareja[0]
        categoria_y = pareja[1]

        p_xy = frecuencias_conjuntas[pareja] / n
        p_x = px[categoria_x]
        p_y = py[categoria_y]

        informacion += p_xy * math.log2(p_xy / (p_x * p_y))

    return informacion


def calcular_matriz_relaciones(dataset):
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("El dataset debe ser un pandas DataFrame.")

    if dataset.shape[1] < 2:
        raise ValueError("El dataset debe tener al menos dos variables.")

    n = dataset.shape[1]
    
    matriz = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):

            col_i = dataset.iloc[:, i]
            col_j = dataset.iloc[:, j]

            x = col_i.tolist()
            y = col_j.tolist()

            x_numerica = pd.api.types.is_numeric_dtype(col_i)
            y_numerica = pd.api.types.is_numeric_dtype(col_j)

            if i == j:
                if x_numerica and len(set(x)) > 1:
                    matriz[i, j] = 1.0
                elif not x_numerica:
                    matriz[i, j] = calcular_informacion_mutua(x, x)

            elif x_numerica and y_numerica:
                matriz[i, j] = calcular_correlacion(x, y)

            elif not x_numerica and not y_numerica:
                matriz[i, j] = calcular_informacion_mutua(x, y)

            else:
                matriz[i, j] = np.nan
                
    matriz_df = pd.DataFrame(
        matriz,
        index=dataset.columns,
        columns=dataset.columns
    )

    return matriz_df