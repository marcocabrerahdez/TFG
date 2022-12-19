import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import settings as st

from typing import List

def search_model_file(model_name: str, directory_name: str) -> str:
  ''' Busca el archivo de un modelo.
  '''
  # directory_name es el directori padre de model_name
  # Buscamos el archivo del modelo
  model_file = glob.glob(os.path.join(st.PREDICTIONS_DIR, '**', directory_name, model_name + '*.xlsx'))

  # Si no encontramos el archivo, devolvemos error
  if not model_file:
    return print('No se encontró el archivo del modelo.', model_name)

  # Devolvemos el archivo
  return model_file[0]



def get_model_results(model_name: str, directory_name: str) -> pd.DataFrame:
  ''' Obtiene los resultados de un modelo.
  '''
  # Buscamos el archivo del modelo
  model_file = search_model_file(model_name, directory_name)
  # Si no encontramos el archivo, devolvemos error
  if not model_file or not directory_name:
    return print('No se encontró el archivo del modelo o directorio.', model_name, directory_name)

  # Leemos el archivo y lo devolvemos
  return pd.read_excel(model_file)



def compare(models_list: List[str], directory_name: List[str], plot_name: str) -> None:
  ''' Muestra una gráfica de comparación de modelos.
      Compara los resultados de los modelos entrenados individualmente
      con los resultados de un modelo entrenado con el conjunto.
  '''
  # Creamos un diccionario para guardar cada dataframe
  df_results = pd.DataFrame()

  # Recorremos la lista de modelos
  for directory in directory_name:
    for model in models_list:
      # Obtenemos los resultados del modelo
      df_model_results = get_model_results(model, directory)

      # Si no encontramos el archivo, devolvemos error
      if type(df_model_results) == str:
        return df_model_results

      # Concatenamos los resultados de cada dataframe
      df_results = pd.concat([df_results, df_model_results], axis=0)

    # Seleccionar las filas que coincidan en la columna 'Enfermedad'
    df_results.reset_index(inplace=True)
    df_results.drop('index', axis=1, inplace=True)

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

    # Guardar la gráfica en el directorio de comparaciones y en subdirectorio directorio
    if not os.path.exists(os.path.join(st.COMPARISION_DIR, directory)):
      os.makedirs(os.path.join(st.COMPARISION_DIR, directory))
    plt.savefig(os.path.join(st.COMPARISION_DIR, directory, plot_name + '.png'))

    # Centrar la gráfica
    plt.close()

    # Limpiar el dataframe
    df_results = pd.DataFrame()