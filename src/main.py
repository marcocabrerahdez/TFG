''' Programa principal.
    Este programa es parte de un Trabajo de Fin de Grado de la Universidad de La Laguna.

    El objetivo de este programa es generar modelos de predicción para el tiempo de espera
    de pacientes con diabetes tipo 1 en Canarias. Y comparar los resultados de los modelos
    de simulación con los resultados de los modelos de predicción.

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
import json
import pandas as pd
import argparse
import joblib

import settings as st
from scripts import automl as ml

# Main function
def main() -> None:
  # Open param_models.json
  with open(os.path.join(st.CONFIG_DIR, st.PARAM_MODELS), 'r') as f:
    config_list = json.load(f)
  # Read data
  df = pd.read_excel(os.path.join(st.DATA_DIR, st.DATASET_NAME), 'Processed')

  # Store data from param_models.json
  for config in config_list['config_list']:
    # Create AutoML object
    automl = ml.AutoML(config['name'], df, config['model'], config['params'], config['columns_X'], config['columns_Y'])
    # Run
    automl.run()

if __name__ == '__main__':
  main()