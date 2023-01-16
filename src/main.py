''' Programa principal.
    Este programa es parte de un Trabajo de Fin de Grado
    de la Universidad de La Laguna.

    El objetivo de este programa es generar
    modelos de predicción para el tiempo de espera
    de pacientes con diabetes tipo 1 en Canarias.
    Y comparar los resultados de los modelos
    de simulación con los resultados de los modelos de predicción.

    Los datos han sido previamente procesados
    y se encuentran en el directorio data.
    Se han generado gráficas con los resultados
    y se encuentran en el directorio figures.
    Además, se han generado los modelos y se encuentran en el directorio model.
    Y por último, se han generadolos resultados de las predicciones
    y se encuentran en el directorio predictions.

    Autor:
        Marco Antonio Cabrera Hernández
'''

import os
import json
import pandas as pd

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
  with open(os.path.join(st.CONFIG_DIR, st.PARAM_MODELS), 'r',
            encoding='utf8') as file_name:
    config_list = json.load(file_name)

  # Abrir el archivo de configuración de la comparación
  with open(os.path.join(st.CONFIG_DIR, st.COMPARE_MODELS), 'r',
            encoding='utf8') as file_name:
    compare_list = json.load(file_name)

  # Leer los datos
  data_frame = pd.read_excel(os.path.join(st.DATA_DIR, st.DATASET_NAME),
                              'Processed')
  # Para cada modelo en la lista de modelos
  for config in config_list['config_list']:
    # Crear el objeto AutoML
    if config['type'] == 'single':
      automl = ml.AutoML(config['name'], config['class_name'],
                          config['model'], config['type'], config['params'],
                          columns_X=data_frame[config['columns_X']],
                          columns_Y=data_frame[config['columns_Y']])
    else:
      automl = ml.AutoML(config['name'], config['class_name'],
                          config['model'], config['type'],
                          config['params'], config['trained_data_names'])
    # Entrenar el modelo
    automl.train()

    # Predecir con el modelo
    automl.predict()

      # Calcular las métricas
    automl.metrics()

    # Guardar el modelo, las predicciones y las metricas
    automl.save()

    # Graficar los resultados
    automl.plot()

  # Comparar las métricas de los resultados de los modelos
  for model in compare_list['compare']:
    cp.compare_metrics(model['model'], model['directory'], model['name'])

  # Comparar los modelos
  cp.compare_models(compare_list['compare_model']['list'],
                    compare_list['compare_model']['directory'],
                    compare_list['compare_model']['name'])

if __name__ == '__main__':
  main()
