''' Programa principal.
    Este programa es parte de un Trabajo de Fin de Grado de la Universidad de La Laguna.

    El objetivo de este programa es generar modelos de predicción para el tiempo de espera
    de pacientes con diabetes tipo 1 en Canarias. Y comparar los resultados de los modelos
    de simulación con los resultados de los modelos de predicción.

    Los datos han sido previamente procesados y se encuentran en el directorio data.
    Se han generado gráficas con los resultados y se encuentran en el directorio figures. Además,
    se han generado los modelos y se encuentran en el directorio model. Y por último, se han generado
    los resultados de las predicciones y se encuentran en el directorio predictions.

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
from scripts import compare as cp

def main() -> None:
  ''' Función principal.

      Parámetros:
          -h, --help: Muestra la ayuda del programa.
          -v, --version: Muestra la versión del programa.

      Ejemplo:
          python3 main.py -v
  '''
  # Abrir el archivo de configuración del modelo
  with open(os.path.join(st.CONFIG_DIR, st.PARAM_MODELS), 'r') as f:
    config_list = json.load(f)

  # Abrir el archivo de configuración de la comparación
  with open(os.path.join(st.CONFIG_DIR, st.COMPARE_MODELS), 'r') as f:
    compare_list = json.load(f)

  # Leer los datos
  df = pd.read_excel(os.path.join(st.DATA_DIR, st.DATASET_NAME), 'Processed')
  # Para cada modelo en la lista de modelos
  for config in config_list['config_list']:
    # Crear el objeto AutoML
    automl = ml.AutoML(config['name'], config['class_name'], config['model'], config['type'], config['params'], df[config['columns_X']], df[config['columns_Y']])

    # Entrenar el modelo
    automl.train()

    # Predecir con el modelo
    automl.predict()

    # Guardar el modelo
    automl.save()

    # Graficar los resultados
    automl.plot()

  # Comparar los modelos
  for model in compare_list['compare']:
    cp.compare(model['model'], model['directory'], model['name'])

  cp.compare_models(compare_list['compare_model']['list'], compare_list['compare_model']['directory'], compare_list['compare_model']['name'])

if __name__ == '__main__':
  main()