''' Archivo de configuración de la aplicación '''

import os

# Directorios de la aplicación
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CONFIG_DIR = os.path.join(SRC_DIR, 'config')

# DATOS DE ENTRENAMIENTO Y PRUEBA
SPLITED_DATA = os.path.join(DATA_DIR, 'splited data')
SPLITED_DATA_MULTIPLE = os.path.join(SPLITED_DATA, 'multiple')
SPLITED_DATA_SINGLE = os.path.join(SPLITED_DATA, 'single')
SPLITED_DATA_GLOBAL = os.path.join(SPLITED_DATA, 'global')

# MODELOS
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
SINGLE_MODEL_DIR = os.path.join(MODEL_DIR, 'single')
MULTIPLE_MODEL_DIR = os.path.join(MODEL_DIR, 'multiple')
GLOBAL_MODEL_DIR = os.path.join(MODEL_DIR, 'global')

# FIGURAS
PLOTS_DIR = os.path.join(ROOT_DIR, 'figures')
SINGLE_PLOTS_DIR = os.path.join(PLOTS_DIR, 'single')
MULTIPLE_PLOTS_DIR = os.path.join(PLOTS_DIR, 'multiple')
GLOBAL_PLOTS_DIR = os.path.join(PLOTS_DIR, 'global')
R2_PLOTS_DIR = os.path.join(PLOTS_DIR, 'R2')
R2_AVERAGE_TIME_PLOT_DIR = os.path.join(R2_PLOTS_DIR, 'average time')
R2_UPTO_TIME_PLOT_DIR = os.path.join(R2_PLOTS_DIR, 'upto time')
R2_AVERAGE_UPTO_TIME_PLOT_DIR = os.path.join(R2_PLOTS_DIR, 'average & upto')


# PREDICCIONES
PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions')
SINGLE_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'single')
MULTIPLE_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'multiple')
GLOBAL_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'global')
COMPARISION_DIR = os.path.join(PLOTS_DIR, 'comparisions')

# METRICAS
METRICS_DIR = os.path.join(ROOT_DIR, 'metrics')
SINGLE_METRICS_DIR = os.path.join(METRICS_DIR, 'single')
MULTIPLE_METRICS_DIR = os.path.join(METRICS_DIR, 'multiple')
GLOBAL_METRICS_DIR = os.path.join(METRICS_DIR, 'global')
R2_TABLE_DIR = os.path.join(METRICS_DIR, 'R2')
SINGLE_R2_TABLE_DIR = os.path.join(R2_TABLE_DIR, 'single')
MULTIPLE_R2_TABLE_DIR = os.path.join(R2_TABLE_DIR, 'multiple')
GLOBAL_R2_TABLE_DIR = os.path.join(R2_TABLE_DIR, 'global')
MAPE_TABLE_DIR = os.path.join(METRICS_DIR, 'MAPE')
SINGLE_MAPE_TABLE_DIR = os.path.join(MAPE_TABLE_DIR, 'single')
MULTIPLE_MAPE_TABLE_DIR = os.path.join(MAPE_TABLE_DIR, 'multiple')
GLOBAL_MAPE_TABLE_DIR = os.path.join(MAPE_TABLE_DIR, 'global')
R2_AVERAGE_TIME_DIR = os.path.join(R2_TABLE_DIR, 'average time')
R2_UPTO_TIME_DIR = os.path.join(R2_TABLE_DIR, 'upto time')
R2_AVERAGE_UPTO_TIME_DIR = os.path.join(R2_TABLE_DIR, 'average & upto')

# ARCHIVOS DE CONFIGURACIÓN
DATASET_NAME = "2023-01-11 datos_clusterizados_ML.xlsx"
PARAM_MODELS = 'model_config.json'
COMPARE_MODELS = 'compare_config.json'

# PARÁMETROS DE ENTRENAMIENTO
TEST_SIZE = 0.2
RANDOM_STATE = 42
VERSION=1.0