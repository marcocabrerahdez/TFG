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

import json
import os
import sys

import pandas as pd

import settings as st
from api import api
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
  """
  # Preprocesar los datos
  df_cols = data_frame.columns[data_frame.columns.str.contains('UPTO')]
  data_frame[df_cols] = data_frame[df_cols].div(500) * 100
  df_cols = data_frame.columns[data_frame.columns.str.contains('INC')]
  data_frame[df_cols] = data_frame[df_cols].div(500) * 100

  # Cambiar SEX: MAN -> 0, WOMAN -> 1
  data_frame['SEX'] = data_frame['SEX'].replace({'MAN': 0, 'WOMAN': 1})

  # Guardar posiciones de los valores NaN
  nan_pos = data_frame.isna()

  # Valores NaN a 0
  data_frame = data_frame.fillna(0)

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

    # R2 and MAPE score
    automl.metrics(nan_pos, increment=3)

    # Guardar el modelo, las predicciones y las metricas
    automl.save()

    # Graficar los resultados
    #automl.plot_upto_time()
    automl.plot_avg_time(nan_pos)
  """
  # Comparar las métricas de los resultados de los modelos
  #cp.create_score_table(compare_list['r2']['list'], compare_list['r2']['name_list'], st.R2_TABLE_DIR, st.R2_AVERAGE_TIME_DIR)
  #cp.create_score_table(compare_list['mape']['list'], compare_list['mape']['name_list'], st.MAPE_TABLE_DIR, st.MAPE_AVERAGE_TIME_DIR)
  cp.compare_r2_tables(compare_list['r2']['name_list'], st.R2_INCIDENCE_PLOT_DIR, st.R2_INCIDENCE_DIR)
  """
  # Carga los datos de la API
  patient = api.load_data()

  # Transformaciones del JSON
  # Cambiamos los nombres de las columnas
  patient_data_base = patient.rename(columns={'baseHbA1cLevel': 'HBA1C', 'age': 'AGE', 'durationOfDiabetes': 'DURATION', 'hypoRate': 'HYPO_RATE', 'man': 'SEX'})
  patient_data_int = patient.rename(columns={'objHbA1cLevel': 'HBA1C', 'age': 'AGE', 'durationOfDiabetes': 'DURATION', 'hypoRateRR': 'HYPO_RATE', 'man': 'SEX'})

  # Multiplicamos los valores de HYPO_RATE por el valor de base
  patient_data_int['HYPO_RATE'] = patient_data_int['HYPO_RATE'] * patient_data_base['HYPO_RATE']

  # Cambiar SEX: man = true -> 0, woman = false -> 1
  patient_data_base['SEX'] = patient_data_base['SEX'].replace({'true': 0, 'false': 1})
  patient_data_int['SEX'] = patient_data_int['SEX'].replace({'true': 0, 'false': 1})

  # El coste anual se multiplica por el valor de HYPO_RATE
  patient_data_int['annualCost'] = patient_data_int['annualCost'] * patient_data_int['HYPO_RATE']

  # Llamada a la api para obtener los datos
  data = api.run(patient_data_base, patient_data_int)
  print(data)

  # Hace la petición POST
  url = ""
  headers = {"Content-Type": "application/json"}
  response = requests.post(url, data=data, headers=headers)

  print(response.status_code) # Imprime el código de estado HTTP de la respuesta
  print(response.json()) # Imprime los datos de la respuesta en formato JSON
  """
if __name__ == '__main__':
  main()
  os.system('afplay /System/Library/Sounds/Glass.aiff')
  os.system('afplay /System/Library/Sounds/Glass.aiff')
