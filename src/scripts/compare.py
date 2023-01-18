import os
import glob
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

import settings as st
import utils as ut


def compare_metrics(models_list: List[str], directory_name: List[str],
            plot_name: str) -> None:
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
      df_model_results = ut.get_model_results(model, directory)

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
    fig, ax = plt.subplots(figsize=(30, 40), nrows=diseases_count, ncols=1)

    # Graficar los resultados en una misma figura
    for i, disease in enumerate(diseases):
      df_results.get_group(disease).plot.bar(
        x='Tipo',
        y=['R2', 'MSE', 'MAE', 'Elapsed Time', 'CPU'],
        rot=0,
        legend=False,
        ax=ax[i],
        colormap='winter',
        xlabel=disease
      )

      # Anotar los valores de R2 y MSE en cada barra
      # y escribirlo encima de la barra centrado
      for p in ax[i].patches:
        ax[i].annotate(
          str(round(p.get_height(), 6)),
          (p.get_x() + p.get_width() / 2., p.get_height()),
          ha='center',
          va='center',
          xytext=(0, 10),
          textcoords='offset points'
        )

      # Añadir leyenda en un lateral
      ax[i].set_ylabel('Valor')

    # Añadir leyenda en un lateral
    fig.legend(
      labels=['R2', 'MSE', 'MAE', 'Elapsed Time', 'CPU'],
      ncol=2,
      fontsize=20
    )

    # Añadir título
    fig.suptitle(plot_name, fontsize=30)

    # Ajustar el espacio entre subgráficas
    fig.tight_layout()

    # Ajustar el espacio entre subgráficas y el título
    fig.subplots_adjust(top=0.95, hspace=0.5)

    # Guardar la gráfica en el directorio de comparaciones
    # y en subdirectorio directorio
    if not os.path.exists(os.path.join(st.COMPARISION_DIR, directory)):
      os.makedirs(os.path.join(st.COMPARISION_DIR, directory))
    plt.savefig(os.path.join(st.COMPARISION_DIR, directory, plot_name + '.png'))

    # Centrar la gráfica
    plt.close()

    # Limpiar el dataframe
    df_results = pd.DataFrame()



def compare_models(model: str, directory_name: List[str],
                  plot_name: str) -> None:
  "Compara los resultados según el modelo usado para entrenarlos."

  # Creamos un diccionario para guardar cada dataframe
  df_results = pd.DataFrame()

  # Recorremos la lista de modelos
  for directory in directory_name:
    df_model_results = ut.get_model_results(model, directory)

    # Calcular la media de los resultados de cada modelo
    df_mean_r2_results = df_model_results['R2'].mean()
    df_mean_mse_results = df_model_results['MSE'].mean()
    df_mean_mae_results = df_model_results['MAE'].mean()
    df_mean_elapsed_time_results = df_model_results['Elapsed Time'].mean()
    df_mean_cpu_results = df_model_results['CPU'].mean()

    # Añadir los resultados a un dataframe
    df_results = pd.concat([
      df_results,
      pd.DataFrame(
        {
          'Modelo': directory,
          'R2': [df_mean_r2_results],
          'MSE': [df_mean_mse_results],
          'MAE': [df_mean_mae_results],
          'Elapsed Time': [df_mean_elapsed_time_results],
          'CPU': [df_mean_cpu_results]
        }
      )
    ], axis=0)

  # Graficar los resultados
  fig, ax = plt.subplots(figsize=(10, 10))

  # Graficar los resultados en una misma figura
  df_results.plot.bar(x='Modelo', y=['R2', 'MSE', 'MAE', 'Elapsed Time', 'CPU'],
                      rot=0, legend=False, ax=ax,
                      colormap='winter', xlabel=model)

  # Anotar los valores de R2 y MSE en cada barra
  # y escribirlo encima de la barra centrado
  for p in ax.patches:
    ax.annotate(
      str(round(p.get_height(), 6)),
      (p.get_x() + p.get_width() / 2., p.get_height()),
      ha='center', va='center', xytext=(0, 10), textcoords='offset points'
    )

  # Añadir leyenda en un lateral
  ax.set_ylabel('Valor')

  # Añadir leyenda en un lateral
  fig.legend(labels=['R2', 'MSE', 'MAE', 'Elapsed Time', 'CPU'],
              ncol=2, fontsize=10)

  # Añadir título
  fig.suptitle(plot_name, fontsize=20)

  # Ajustar el espacio entre subgráficas
  fig.tight_layout()

  # Ajustar el espacio entre subgráficas y el título
  fig.subplots_adjust(top=0.95)

  # Guardar la gráfica en el directorio de comparaciones
  if not os.path.exists(st.COMPARISION_DIR):
    os.makedirs(st.COMPARISION_DIR)
  plt.savefig(os.path.join(st.COMPARISION_DIR, plot_name + '.png'))
