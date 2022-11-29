''' Programa principal.
    Este programa es parte de un Trabajo de Fin de Grado de la Universidad de La Laguna.

    El objetivo de este programa es generar modelos de predicción para el tiempo de espera
    de pacientes con diabetes tipo 1 en Canarias. Y comparar los resultados de los modelos
    de simulación con los resultados de los modelos de predicción.

    El programa genera los siguientes modelos de predicción:
        - Modelo de regresión lineal
        - Modelo de bosque aleatorio de regresión

    Los datos han sido previamente procesados y se encuentran en el directorio data.
    Se han generado gráficas con los resultados y se encuentran en el directorio figures. Además,
    se han generado los modelos y se encuentran en el directorio model. Y por último, se han generado
    los resultados de las predicciones y se encuentran en el directorio predictions.

    Parámetros:
        -h, --help: Muestra la ayuda del programa.
        -v, --version: Muestra la versión del programa.

    Ejemplo:
        python3 main.py -v

    Autor:
        Marco Antonio Cabrera Hernández
'''

import os
import argparse

import settings as st

from utils import load as utils_load
from utils import plot as utils_plot

from model import linear_regression as linear_regression_model
from model import random_forest_regressor as random_forest_regressor_model

def main() -> None:
    '''
        Carga los datos y genera los modelos.
    '''
    # Carga los datos
    df = utils_load.load_dataset(os.path.join(st.DATA_DIR, st.DATA_NAME))

    # Genera la gráfica de correlación entre las variables
    utils_plot.plot_correlations(df, 'Correlation')

    # Modelo de regresión lineal
    linear_regression_model.linear_regression_model(df);

    # Modelo de bosque aleatorio de regresión
    random_forest_regressor_model.random_forest_regressor_model(df);

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ejecuta varios modelos de regresión sobre un dataset de diabetes tipo 1.')
    args = parser.parse_args()

    main()