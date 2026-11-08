import math
import pandas as pd

def discretizar_anchura_atributo(vector_datos, num_intervalos):
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector_datos):
        raise ValueError("El atributo debe ser numérico.")

    if len(vector_datos) == 0:
        raise ValueError("El atributo no puede estar vacío.")

    if any(math.isnan(x) if isinstance(x, float) else False for x in vector_datos):
        raise ValueError("El atributo no puede contener valores NA.")

    if not isinstance(num_intervalos, int) or isinstance(num_intervalos, bool) or num_intervalos < 1:
        raise ValueError("num_intervalos debe ser un entero positivo.")

    valor_min = min(vector_datos)
    valor_max = max(vector_datos)

    ancho = (valor_max - valor_min) / num_intervalos
    
    limites = []
    for i in range(num_intervalos + 1):
        limites.append(valor_min + i * ancho)

    limites_red = [round(limite, 2) for limite in limites]

    etiquetas = []
    for k in range(num_intervalos):
        if k == num_intervalos - 1:
            etiqueta = f"[{limites_red[k]},{limites_red[k + 1]}]"
        else:
            etiqueta = f"[{limites_red[k]},{limites_red[k + 1]})"

        etiquetas.append(etiqueta)

    resultado = []
    for x in vector_datos:
        if x == valor_max:
            indice_intervalo = num_intervalos - 1
        else:
            indice_intervalo = math.floor((x - valor_min) / ancho)

        resultado.append(etiquetas[indice_intervalo])

    return {
        "datos_discretizados": resultado,
        "limites": limites
    }


def discretizar_frecuencia_atributo(vector_datos, num_intervalos):
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector_datos):
        raise ValueError("El atributo debe ser numérico.")

    if len(vector_datos) == 0:
        raise ValueError("El atributo no puede estar vacío.")

    if any(isinstance(x, float) and math.isnan(x) for x in vector_datos):
        raise ValueError("El atributo no puede contener valores NA.")

    if any(math.isinf(x) for x in vector_datos if isinstance(x, (int, float))):
        raise ValueError("El atributo no puede contener valores infinitos.")

    if not isinstance(num_intervalos, int) or isinstance(num_intervalos, bool) or num_intervalos < 1:
        raise ValueError("num_intervalos debe ser un entero mayor o igual a 1.")

    n = len(vector_datos)
    posiciones_ordenadas = list(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if vector_datos[posiciones_ordenadas[i]] > vector_datos[posiciones_ordenadas[j]]:
                temporal = posiciones_ordenadas[i]
                posiciones_ordenadas[i] = posiciones_ordenadas[j]
                posiciones_ordenadas[j] = temporal

    posiciones = [0] * n
    for posicion, indice_original in enumerate(posiciones_ordenadas):
        posiciones[indice_original] = posicion + 1

    elementos_por_intervalo = n / num_intervalos
    indices_intervalo = []

    for posicion in posiciones:
        indice = math.ceil(posicion / elementos_por_intervalo)
        if indice > num_intervalos:
            indice = num_intervalos
        indices_intervalo.append(indice)

    datos_ordenados = sorted(vector_datos)
    limites = [0] * (num_intervalos + 1)
    limites[0] = datos_ordenados[0]

    for k in range(1, num_intervalos):
        idx = round(k * elementos_por_intervalo)
        idx = max(1, min(idx, n))
        limites[k] = datos_ordenados[idx - 1]

    limites[num_intervalos] = datos_ordenados[n - 1]

    limites_red = [round(limite, 2) for limite in limites]
    etiquetas = []

    for k in range(num_intervalos):
        if k == num_intervalos - 1:
            etiqueta = f"[{limites_red[k]},{limites_red[k + 1]}]"
        else:
            etiqueta = f"[{limites_red[k]},{limites_red[k + 1]})"

        etiquetas.append(etiqueta)

    resultado = [etiquetas[indice - 1] for indice in indices_intervalo]

    return {
        "datos_discretizados": resultado,
        "limites": limites
    }


def discretizar_dataset(dataset, num_intervalos, metodo):
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("El dataset debe ser un pandas DataFrame.")

    if dataset.shape[1] == 0:
        raise ValueError("El dataset no contiene columnas.")

    if metodo not in ["anchura", "frecuencia"]:
        raise ValueError("El método debe ser 'anchura' o 'frecuencia'.")

    dataset_discretizado = dataset.copy()

    for nombre_columna in dataset.columns:
        columna = dataset[nombre_columna]

        if not pd.api.types.is_numeric_dtype(columna):
            continue

        vector_datos = columna.tolist()

        if metodo == "anchura":
            res = discretizar_anchura_atributo(vector_datos, num_intervalos)
        elif metodo == "frecuencia":
            res = discretizar_frecuencia_atributo(vector_datos, num_intervalos)

        dataset_discretizado[nombre_columna] = res["datos_discretizados"]

    return dataset_discretizado