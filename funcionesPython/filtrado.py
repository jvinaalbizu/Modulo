import pandas as pd

def filtrar_variables(dataset, nombre_var_clase):
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("El dataset debe ser un pandas.DataFrame.")

    if nombre_var_clase not in dataset.columns:
        raise ValueError(f"La variable clase '{nombre_var_clase}' no existe en el dataset.")
    metricas = calcular_metricas_dataset(dataset, nombre_var_clase)
    variables_mantener = []
    for i in range(len(metricas)):

        nombre_variable = metricas.loc[i, "Atributo"]
        tipo = metricas.loc[i, "Tipo"]
        if nombre_variable == nombre_var_clase:
            continue
        if tipo == "Continua":
            variable_normalizada = normalizar(dataset[nombre_variable].tolist())
            varianza_normalizada = calcular_varianza(variable_normalizada)
            auc = metricas.loc[i, "AUC"]
            if varianza_normalizada > 0.05 and auc > 0.60:
                variables_mantener.append(nombre_variable)
        else:
            entropia = metricas.loc[i, "Entropia"]
            if entropia > 0.50:
                variables_mantener.append(nombre_variable)
    if nombre_var_clase not in variables_mantener:
        variables_mantener.append(nombre_var_clase)
    dataset_filtrado = dataset[variables_mantener].copy()
    return dataset_filtrado