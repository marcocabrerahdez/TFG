''' Archivo de configuración de la aplicación '''

import os

# Directorios
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

MODEL_DIR = os.path.join(ROOT_DIR, 'model')
SINGLE_MODEL_DIR = os.path.join(MODEL_DIR, 'single')
MULTIPLE_MODEL_DIR = os.path.join(MODEL_DIR, 'multiple')

SRC_DIR = os.path.join(ROOT_DIR, 'src')

PLOTS_DIR = os.path.join(ROOT_DIR, 'figures')

PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions')
SINGLE_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'single')
MULTIPLE_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'multiple')

COMPARISION_DIR = os.path.join(PLOTS_DIR, 'comparisions')
CONFIG_DIR = os.path.join(SRC_DIR, 'config')

METRICS_DIR = os.path.join(PLOTS_DIR, 'metrics')
SINGLE_METRICS_DIR = os.path.join(METRICS_DIR, 'single')
MULTIPLE_METRICS_DIR = os.path.join(METRICS_DIR, 'multiple')


# Archivos
DATASET_NAME = "2021-07-22 datos_clusterizados_ML.xlsx"
PARAM_MODELS = 'model_config.json'
COMPARE_MODELS = 'compare_config.json'

# Parámetros
TEST_SIZE = 0.2
RANDOM_STATE = 42