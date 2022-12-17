import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import settings as st

from typing import List

def search_model_file(model_name: str) -> str:
  ''' Busca el archivo de un modelo.
  '''
  # Recorremos todos los subdirectorios del directorio de predicciones
  for root, dirs, files in os.walk(st.PREDICTIONS_DIR):
    # Buscamos el archivo con el nombre del modelo
    model_file = glob.glob(os.path.join(root, model_name + "*"))
    # Si encontramos el archivo, lo devolvemos
    if model_file:
      return model_file[0]



def get_model_results(model_name: str) -> pd.DataFrame:
  ''' Obtiene los resultados de un modelo.
  '''
  # Buscamos el archivo del modelo
  model_file = search_model_file(model_name)
  # Si no encontramos el archivo, devolvemos error
  if not model_file:
    return print('No se encontró el archivo del modelo.', model_name)

  # Leemos el archivo y lo devolvemos
  return pd.read_excel(model_file)



def compare(models_list: List[str], plot_name: str) -> None:
  ''' Muestra una gráfica de comparación de modelos.
      Compara los resultados de los modelos entrenados individualmente
      con los resultados de un modelo entrenado con el conjunto.
  '''
  # Creamos un diccionario para guardar cada dataframe
  df_results = pd.DataFrame()

  # Recorremos la lista de modelos
  for model in models_list:
    # Obtenemos los resultados del modelo
    df_model_results = get_model_results(model)

    # Si no encontramos el archivo, devolvemos error
    if type(df_model_results) == str:
      return df_model_results

    # Concatenamos los resultados de cada dataframe
    df_results = pd.concat([df_results, df_model_results], axis=0)

  # Seleccionar las filas que coincidan en la columna 'Enfermedad'
  df_results.reset_index(inplace=True)
  df_results.drop('index', axis=1, inplace=True)

  # Quitar las filas que coincidan en la columna 'Enfermedad' que empiecen por L95CI o U95CI
  df_results = df_results[~df_results['Enfermedad'].str.contains('L95CI')]
  df_results = df_results[~df_results['Enfermedad'].str.contains('U95CI')]

  # Agrupar los resultados por enfermedad
  df_results = df_results.groupby('Enfermedad')

  # Obtener las enfermedades y el número de enfermedades
  diseases = list(df_results.groups.keys())
  diseases_count = len(diseases)

  # Graficar los resultados
  fig, ax = plt.subplots(figsize=(25, 25), nrows=diseases_count, ncols=1)

  # Graficar los resultados en una misma figura
  for i, disease in enumerate(diseases):
    df_results.get_group(disease).plot.bar(x='Tipo', y=['R2', 'MSE'], rot=0, legend=False, ax=ax[i], colormap='winter', xlabel=disease)

    # Anotar los valores de R2 y MSE en cada barra y escribirlo encima de la barra centrado
    for p in ax[i].patches:
      ax[i].annotate(str(round(p.get_height(), 6)), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Añadir leyenda en un lateral
    ax[i].set_ylabel('Valor')

  # Añadir leyenda en un lateral
  fig.legend(labels=['R2', 'MSE'], ncol=2, fontsize=20)

  # Añadir título
  fig.suptitle(plot_name, fontsize=30)

  # Ajustar el espacio entre subgráficas
  fig.tight_layout()

  # Ajustar el espacio entre subgráficas y el título
  fig.subplots_adjust(top=0.95)

  # Guardar la gráfica
  plt.savefig(os.path.join(st.COMPARISION_DIR, plot_name + '.png'))

  # Centrar la gráfica
  plt.close()