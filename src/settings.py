''' Archivo de configuración de la aplicación '''

import os

# Directorios
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
SRC_DIR = os.path.join(ROOT_DIR, 'src')
PLOTS_DIR = os.path.join(ROOT_DIR, 'figures')
PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions')
CONFIG_DIR = os.path.join(SRC_DIR, 'config')

# Archivos
DATASET_NAME = "2021-07-22 datos_clusterizados_ML.xlsx"
PARAM_MODELS = 'model_config.json'