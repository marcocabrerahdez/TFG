''' Archivo de configuración de la aplicación '''

import os

# Directorios de la aplicación
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# MODELOS
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
SINGLE_MODEL_DIR = os.path.join(MODEL_DIR, 'single')
MULTIPLE_MODEL_DIR = os.path.join(MODEL_DIR, 'multiple')

SRC_DIR = os.path.join(ROOT_DIR, 'src')

# FIGURAS
PLOTS_DIR = os.path.join(ROOT_DIR, 'figures')
SINGLE_PLOTS_DIR = os.path.join(PLOTS_DIR, 'single')
MULTIPLE_PLOTS_DIR = os.path.join(PLOTS_DIR, 'multiple')

# PREDICCIONES
PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions')
SINGLE_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'single')
MULTIPLE_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'multiple')
COMPARISION_DIR = os.path.join(PLOTS_DIR, 'comparisions')

# CONFIGURACION
CONFIG_DIR = os.path.join(SRC_DIR, 'config')

# Archivos
DATASET_NAME = "2021-07-22 datos_clusterizados_ML.xlsx"
PARAM_MODELS = 'model_config.json'
COMPARE_MODELS = 'compare_config.json'

# Parámetros
TEST_SIZE = 0.2
RANDOM_STATE = 42