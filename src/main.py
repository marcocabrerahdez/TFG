''' Programa principal '''

import os

from utils import load as utils_load
from utils import plot as utils_plot
from utils import save as utils_save

from model import linear_regression as linear_regression_model
from model import random_forest_regressor as random_forest_regressor_model

import settings as st

def main():
    '''
        Carga los datos, el modelo y guarda los resultados.
    '''
    df = utils_load.load_dataset(os.path.join(st.DATA_DIR, st.DATA_NAME))

    # Modelo de regresión lineal
    linear_regression_model.linear_regression_model(df);

    # Modelo de bosque aleatorio de regresión
    random_forest_regressor_model.random_forest_regressor_model(df);

if __name__ == '__main__':
    main()