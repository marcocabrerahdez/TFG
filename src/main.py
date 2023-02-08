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
import sys
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
          -f, --file: Archivo de datos. Debe ser un archivo de Excel.
            Además, se debe especificar:
              - La hoja del archivo de datos.
              - El archivo de configuración de modelos.
              - El archivo de configuración de comparación de modelos.

      Ejemplo:
          python3 main.py -v
  '''
  if len(sys.argv) > 1:
    # Ayuda del programa
    if sys.argv[1] == '-h' or sys.argv[1] == '--help':
      print(main.__doc__)
      sys.exit()
    # Versión del programa
    elif sys.argv[1] == '-v' or sys.argv[1] == '--version':
      print(st.VERSION)
      sys.exit()

    # Archivos de datos
    elif sys.argv[1] == '-f' or sys.argv[1] == '--file':
      data_file_path = os.path.join(st.ROOT_DIR, sys.argv[2])
      data_file_sheet = sys.argv[3]
      if os.path.isfile(data_file_path):
         data_frame = pd.read_excel(data_file_path, data_file_sheet)

      # Abrir el archivo de configuración de datos
      configuration_model_file_path = os.path.join(st.ROOT_DIR, sys.argv[4])
      if os.path.isfile(configuration_model_file_path):
        with open(configuration_model_file_path, 'r', encoding='utf8') as file_name:
          config_list = json.load(file_name)

      # Abrir el archivo de configuración de la comparación
      compare_file_path = os.path.join(st.ROOT_DIR, sys.argv[5])
      if os.path.isfile(compare_file_path):
        with open(compare_file_path, 'r', encoding='utf8') as file_name:
          compare_list = json.load(file_name)
    else:
      print('Argumento no válido.')
      sys.exit()

  # Preprocesar los datos
  df_cols = data_frame.columns[data_frame.columns.str.contains('UPTO')]
  data_frame[df_cols] = data_frame[df_cols].div(500) * 100
  """
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

    # R2 score
    automl.r2_score_to_table()

    # Guardar el modelo, las predicciones y las metricas
    automl.save()

    # Graficar los resultados
    #automl.plot_upto_time()
    automl.plot_avg_time()
  """
  # Comparar las métricas de los resultados de los modelos
  cp.create_r2_score_table(compare_list['r2']['list'], compare_list['r2']['path'])
  """
  for model in compare_list['compare']:
    cp.compare_avg_metrics(model['model'], model['directory'], model['name'])
    #cp.compare_upto_metrics(model['model'], model['directory'], model['name'])

  # Comparar los modelos
  cp.compare_models(compare_list['compare_model']['list'],
                    compare_list['compare_model']['directory'],
                    compare_list['compare_model']['name'])
  """
if __name__ == '__main__':
  main()
