import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_auc(dataset, nombre_var_clase):
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("El dataset debe ser un DataFrame.")

    if nombre_var_clase not in dataset.columns:
        raise ValueError("La variable de clase no existe en el dataset.")

    clase = dataset[nombre_var_clase].tolist()

    if pd.isna(clase).any() or len(set(clase)) != 2:
        raise ValueError("La variable de clase debe ser binaria y no contener NA.")

    variables = [
        nombre for nombre in dataset.columns
        if nombre != nombre_var_clase and pd.api.types.is_numeric_dtype(dataset[nombre])
    ]

    if len(variables) == 0:
        raise ValueError("No hay variables numéricas para representar.")

    for nombre in variables:

        x = dataset[nombre].tolist()

        if pd.isna(x).any():
            print(f"Se omite {nombre}: hay valores NA.")
            continue

        if len(set(x)) < 2:
            print(f"Se omite {nombre}: variable constante.")
            continue
        auc = calcular_auc(x, clase)
        roc = calcular_roc(x, clase)

        plt.figure()

        plt.plot(
            roc["FPR"], roc["TPR"],
            color="blue",
            linewidth=2,
            label="Curva ROC"
        )

        plt.plot(
            [0, 1], [0, 1],
            color="red",
            linestyle="--",
            label="Clasificador aleatorio"
        )

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f"{nombre}\nAUC = {auc:.3f}")
        plt.xlabel("Tasa de falsos positivos (FPR)")
        plt.ylabel("Tasa de verdaderos positivos (TPR)")
        plt.legend()
        plt.grid(False)

        plt.show()


def plot_correlacion_infmutua(dataset):
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("El dataset debe ser un DataFrame.")

    n = dataset.shape[1]

    if n < 2:
        raise ValueError("Se necesitan al menos dos variables para evaluar relaciones pareadas.")

    for i in range(n - 1):
        for j in range(i + 1, n):

            x = dataset.iloc[:, i].tolist()
            y = dataset.iloc[:, j].tolist()

            nombre_x = dataset.columns[i]
            nombre_y = dataset.columns[j]

            x_numerica = pd.api.types.is_numeric_dtype(dataset.iloc[:, i])
            y_numerica = pd.api.types.is_numeric_dtype(dataset.iloc[:, j])

            if x_numerica and y_numerica:

                r = calcular_correlacion(x, y)

                plt.figure()
                plt.scatter(x, y, color="blue")

                if r is not None and not np.isnan(r):
                    titulo_r = f"{nombre_x} vs {nombre_y}\nCorrelación = {r:.3f}"
                else:
                    titulo_r = f"{nombre_x} vs {nombre_y}\nCorrelación = NA"

                plt.title(titulo_r)
                plt.xlabel(nombre_x)
                plt.ylabel(nombre_y)
 
                if r is not None and not np.isnan(r) and len(set(x)) > 1:

                    media_x = sum(x) / len(x)
                    media_y = sum(y) / len(y)

                    numerador = sum((x[k] - media_x) * (y[k] - media_y) for k in range(len(x)))
                    denominador = sum((valor - media_x) ** 2 for valor in x)

                    if denominador != 0:
                        pendiente = numerador / denominador
                        ordenada = media_y - pendiente * media_x

                        valores_x = np.array([min(x), max(x)])
                        valores_y = ordenada + pendiente * valores_x

                        plt.plot(valores_x, valores_y, color="red", linewidth=2)

                plt.tight_layout()
                plt.show()

            elif not x_numerica and not y_numerica:
                mi = calcular_informacion_mutua(x, y)

                tabla = pd.crosstab(dataset[nombre_x], dataset[nombre_y])

                tabla.plot(kind="bar", figsize=(8, 5))

                plt.title(
                    f"{nombre_x} vs {nombre_y}\n"
                    f"Información mutua = {mi:.3f} bits"
                )
                plt.xlabel(nombre_x)
                plt.ylabel("Frecuencia")
                plt.legend(title=nombre_y)

                plt.tight_layout()
                plt.show()