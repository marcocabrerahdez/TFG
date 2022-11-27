''' Archivo de configuración de la aplicación '''

import os

# Rutas
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data") # Directorio de datos
DATA_PREDICTIONS = os.path.join(ROOT_DIR, "predictions") # Directorio de predicciones
MODEL_DIR = os.path.join(ROOT_DIR, "model") # Directorio de modelos
FIGURES_DIR = os.path.join(ROOT_DIR, "figures") # Directorio de gráficas

# Variables de los datos
DATA_NAME = "2021-07-22 datos_clusterizados_ML.xlsx"