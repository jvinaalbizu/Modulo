from .Discretizacion import discretizar_anchura_atributo, discretizar_frecuencia_atributo, discretizar_dataset
from .metricas import calcular_varianza, calcular_entropia, calcular_auc, calcular_roc, calcular_metricas_dataset
from .transformaciones import normalizar, estandarizar, transformar_dataset
from .filtrado import filtrar_variables
from .relaciones import calcular_correlacion, calcular_informacion_mutua, calcular_matriz_relaciones
from .graficos import plot_auc, plot_correlacion_infmutua