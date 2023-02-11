import os
import glob
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import settings as st
import utils as ut


def compare_avg_metrics(models_list: List[str], directory_name: List[str],
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

    # Separar las filas de la columna enfermedad que coincidadn con 'L95CI' y 'U95CI'
    df_l95ci = df_results[df_results['Enfermedad'].str.contains('L95CI')]
    df_u95ci = df_results[df_results['Enfermedad'].str.contains('U95CI')]

    # Eliminar las filas de df_results
    df_results = df_results[~df_results['Enfermedad'].str.startswith(('L95CI', 'U95CI'))]


    df_results.reset_index(inplace=True)
    df_results.drop('index', axis=1, inplace=True)

    df_l95ci.reset_index(inplace=True)
    df_l95ci.drop('index', axis=1, inplace=True)

    df_u95ci.reset_index(inplace=True)
    df_u95ci.drop('index', axis=1, inplace=True)

    # Renombrsr las filas de la columna enfermedad, reemplazando L95CI por AVG
    mask_l95ci = df_l95ci['Enfermedad'].str.startswith('L95CI')
    df_l95ci.loc[mask_l95ci, 'Enfermedad'] = df_l95ci.loc[mask_l95ci, 'Enfermedad'].str.replace('L95CI', 'AVG')

    mask_u95ci = df_u95ci['Enfermedad'].str.startswith('U95CI')
    df_u95ci.loc[mask_u95ci, 'Enfermedad'] = df_u95ci.loc[mask_u95ci, 'Enfermedad'].str.replace('U95CI', 'AVG')

    # Agrupar los resultados por enfermedad
    df_results = df_results.groupby('Enfermedad')
    df_l95ci = df_l95ci.groupby('Enfermedad')
    df_u95ci = df_u95ci.groupby('Enfermedad')

    # Obtener las enfermedades y el número de enfermedades
    diseases = list(df_results.groups.keys())
    diseases_count = len(diseases)

    # Graficar los resultados
    fig, ax = plt.subplots(figsize=(25, 40), dpi=300, nrows=diseases_count, ncols=5)

    colors = [(0.8, 0.7, 0.2), (0.3, 0.8, 0.5), (0.7, 0.8, 0.2),  (0.2, 0.2, 0.5), (0.4, 0.9, 0.4)]

    # Graficar los resultados en una misma figura
    for i, disease in enumerate(diseases):
      for j, metric in enumerate(['R2', 'MSE', 'MAE', 'Elapsed Time', 'CPU']):
        df_disease = df_results.get_group(disease)
        df_disease_l95ci = df_l95ci.get_group(disease)
        df_disease_u95ci = df_u95ci.get_group(disease)
        bar_colors = [colors[j] for x in range(df_disease.shape[0])]
        df_disease.plot.bar(
          x='Tipo',
          y=metric,
          rot=0,
          legend=False,
          ax=ax[i, j],
          color=bar_colors,
          xlabel=disease
        )

    # Añadir leyenda
    fig.legend(
      labels=['R2', 'MSE', 'MAE', 'Elapsed Time', 'CPU'],
      ncols=5,
      fontsize=15
    )
    for i, disease in enumerate(diseases):
      for j, metric in enumerate(['R2', 'MSE', 'MAE', 'Elapsed Time', 'CPU']):
        # Anotar los valores de la métrica en cada barra
        # y escribirlo encima de la barra centrado
        for p in ax[i, j].patches:
          ax[i, j].annotate(
            str(round(p.get_height(), 6)),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 10),
            textcoords='offset points'
          )
          plot2 = ax[i, j].plot(
            [0,0],
            [df_disease_l95ci[metric].iloc[1], df_disease_u95ci[metric].iloc[1]],
            color='black',
            linestyle='-',
            marker='.'
          )

        # Añadir etiquetas de eje y solo en el primer subplot de la izquierda
        if j == 0:
          ax[i, j].set_ylabel('Valor', fontweight='bold')
        else:
          ax[i, j].set_ylabel('')

        # Añadir títulos a los subplots
        if j == 2:
          ax[i, j].set_title(disease, fontweight='bold', fontsize=15)

        # Añadir etiquetas de eje x solo
        ax[i, j].set_xlabel(metric, fontweight='bold', fontsize=15)

    # Añadir un título a la figura
    fig.suptitle(f'Resultados de {plot_name}', fontsize=20, fontweight='bold')

    # Ajustar el espacio entre subgráficas para que ocupen todo el espacio posible de la figura y el titulo no se solape
    fig.set_layout_engine('compressed')

    # Guardar la gráfica en el directorio de comparaciones
    # y en subdirectorio directorio
    if not os.path.exists(os.path.join(st.COMPARISION_DIR, directory)):
      os.makedirs(os.path.join(st.COMPARISION_DIR, directory))
    plt.savefig(os.path.join(st.COMPARISION_DIR, directory, plot_name + '.png'))

    # Centrar la gráfica
    plt.close()

    # Limpiar el dataframe
    df_results = pd.DataFrame()
    df_l95ci = pd.DataFrame()
    df_u95ci = pd.DataFrame()



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



def create_score_table(metrics_list: List[str], name_list: List[str], path=st.R2_TABLE_DIR) -> None:
  # Creamos un diccionario para guardar cada dataframe
  df_results = pd.DataFrame()

  # Recorremos la lista de modelos
  for metric, name in zip(metrics_list, name_list):
    # Obtenemos el dataframe de los resultados
    result = ut.get_score_file(metric, path)

    # Guardamos cada dataframe en un excel
    if not os.path.exists(os.path.join(path)):
      os.mkdir(path)
    result.to_excel(os.path.join(path, f'{name}.xlsx'), index=False)

    # Reseteamos el dataframe
    result = pd.DataFrame()



def compare_values_by_type(model_list: List[str], path=st.R2_TABLE_DIR) -> None:
  """
  Compara los valores predichos y reales de cada modelo.
  Eligiendo una técnica de ML y una comorbilidad,
  muestra una gráfica de dispersión donde cada serie de datos
  corresponde a un tipo de modelo (single, multiple, global).
  """
  print('Comprobando los valores predichos y reales de cada modelo...')
  # Busca el archivo de la lista de modelos
  for model in model_list:
    # Obtener el dataframe de los resultados
    model_df = ut.get_score_file(model, path)

    # Obtener el valor más grande de entre las columnas (single, multiple, global)
    max_value = model_df[['single', 'multiple', 'global']].max().max()

    # Obtener el indice del valor más grande
    max_index = model_df[['single', 'multiple', 'global']].idxmax().max()

    # Obtener el nombre de la columna donde está el valor más grande
    column_name = model_df[['single', 'multiple', 'global']].idxmax().idxmax()

    # Obtener el nombre de la fila donde está el valor más grande
    max_model = model_df.loc[max_index, 'Modelo']

    # Buscar el archivo de los datos de entrenamiento y test
    X_train, X_test, y_train, y_test, columns_X, columns_Y = ut.get_splited_data([model], 'multiple')

    # Buscar el archivo de predicciones
    pred_single_path = os.path.join(st.SINGLE_PREDICTIONS_DIR, max_model, model + '.xlsx')

    # Abrir el archivo de predicciones
    with open(pred_single_path, 'rb') as f:
      y_pred_single = pd.read_excel(f)

    if model == 'Fallo Cardiaco' or model == 'Infarto de Miocardio' or model == 'Angina' or model == 'Ictus':
      pred_multiple_path = os.path.join(st.MULTIPLE_PREDICTIONS_DIR, max_model, 'Enfermedades cardíacas' + '.xlsx')

    elif model == 'Ceguera' or model == 'Edema macular diabético' or model == 'Retinopatía de fondo' or model == 'Retinopatía proliferativa':
      pred_multiple_path = os.path.join(st.MULTIPLE_PREDICTIONS_DIR, max_model, 'Retinopatías' + '.xlsx')

    elif model == 'Neuropatía individual' or model == 'Amputación extremidades inferiores':
      pred_multiple_path = os.path.join(st.MULTIPLE_PREDICTIONS_DIR, max_model, 'Neuropatías' + '.xlsx')

    else:
      pred_multiple_path = os.path.join(st.MULTIPLE_PREDICTIONS_DIR, max_model, 'Nefropatías' + '.xlsx')

    with open(pred_multiple_path, 'rb') as f:
      y_pred_multiple = pd.read_excel(f)

    pred_global_path = os.path.join(st.GLOBAL_PREDICTIONS_DIR, max_model, 'Comorbilidades' + '.xlsx')

    with open(pred_global_path, 'rb') as f:
      y_pred_global = pd.read_excel(f)

    # Quitar las columnas HBA1C, AGE, DURATION
    y_pred_single = y_pred_single.drop(columns=['HBA1C', 'AGE', 'DURATION'])
    y_pred_multiple = y_pred_multiple.drop(columns=['HBA1C', 'AGE', 'DURATION'])
    y_pred_global = y_pred_global.drop(columns=['HBA1C', 'AGE', 'DURATION'])

    # Usar columns_Y para poner como nombre de columna a y_pred_single
    y_pred_single.columns = columns_Y

    # Crear un switcher para obtener las columnas de y_pred_multiple
    swithcer_multiple = {
      'Fallo Cardiaco': [0, 1, 2],
      'Ceguera': [0, 1, 2],
      'Neuropatía individual':[0, 1, 2],
      'Microalbuminuria': [0, 1, 2],
      'Infarto de Miocardio': [3, 4, 5],
      'Edema macular diabético': [3, 4, 5],
      'Amputación extremidades inferiores':[3, 4, 5],
      'Macroalbuminuria':[3, 4, 5],
      'Angina': [6, 7, 8],
      'Retinopatía de fondo': [6, 7, 8],
      'Enfermedad renal terminal':[6, 7, 8],
      'Ictus': [9, 10, 11],
      'Retinopatía proliferativa': [9, 10, 11],
    }

    # Obtener las columnas de y_pred_multiple a partir del switcher
    y_pred_multiple = y_pred_multiple.iloc[:, swithcer_multiple.get(model)]

    # Cambiar el nombre de la columna por el primer valor de columns_Y
    y_pred_multiple.columns = columns_Y

    # Crear un switcher para obtener las columnas de y_pred_global
    swithcer_global = {
      'Fallo Cardiaco': [0, 1, 2],
      'Infarto de Miocardio': [3, 4, 5],
      'Angina': [6, 7, 8],
      'Ictus': [9, 10, 11],
      'Ceguera': [12, 13, 14],
      'Edema macular diabético': [15, 16, 17],
      'Retinopatía de fondo': [18, 19, 20],
      'Retinopatía proliferativa': [21, 22, 23],
      'Neuropatía individual': [24, 25, 26],
      'Amputación extremidades inferiores': [27, 28, 29],
      'Microalbuminuria': [30, 31, 32],
      'Macroalbuminuria': [33, 34, 35],
      'Enfermedad renal terminal': [36, 37, 38],
    }

    # Obtener las columnas de y_pred_multiple a partir del switcher
    y_pred_global = y_pred_global.iloc[:, swithcer_global.get(model)]

    # Cambiar el nombre de la columna por el primer valor de columns_Y
    y_pred_global.columns = columns_Y

    # Crear una figura
    fig, ax = plt.subplots(figsize=(12, 12), ncols=1, nrows=3)

    ax = ax.ravel()

    # Crea una malla convenxa para los tiempo promedio
    single_avg_points = np.column_stack((y_test.iloc[:, 0], y_pred_single.iloc[:, 0])); single_avg_hull = ConvexHull(single_avg_points)
    multiple_avg_points = np.column_stack((y_test.iloc[:, 0], y_pred_multiple.iloc[:, 0])); multiple_avg_hull = ConvexHull(multiple_avg_points)
    global_avg_points = np.column_stack((y_test.iloc[:, 0], y_pred_global.iloc[:, 0]));  global_avg_hull = ConvexHull(global_avg_points)

    # Crea una malla convenxa para el intervalo inferior de tiempo promedio
    single_avg_l95ci_points = np.column_stack((y_test.iloc[:, 1], y_pred_single.iloc[:, 1])); single_avg_l95ci_hull = ConvexHull(single_avg_l95ci_points)
    multiple_avg_l95ci_points = np.column_stack((y_test.iloc[:, 1], y_pred_multiple.iloc[:, 1])); multiple_avg_l95ci_hull = ConvexHull(multiple_avg_l95ci_points)
    global_avg_l95ci_points = np.column_stack((y_test.iloc[:, 1], y_pred_global.iloc[:, 1])); global_avg_l95ci_hull = ConvexHull(global_avg_l95ci_points)

    # Crea una malla convenxa para el intervalo superior de tiempo promedio
    single_avg_u95ci_points = np.column_stack((y_test.iloc[:, 2], y_pred_single.iloc[:, 2])); single_avg_u95ci_hull = ConvexHull(single_avg_u95ci_points)
    multiple_avg_u95ci_points = np.column_stack((y_test.iloc[:, 2], y_pred_multiple.iloc[:, 2])); multiple_avg_u95ci_hull = ConvexHull(multiple_avg_u95ci_points)
    global_avg_u95ci_points = np.column_stack((y_test.iloc[:, 2], y_pred_global.iloc[:, 2])); global_avg_u95ci_hull = ConvexHull(global_avg_u95ci_points)

    # Grafica los puntos de tiempo promedio ideal
    ax[0].plot(y_test.iloc[:, 0], y_test.iloc[:, 0], color='black', label='Valor ideal tiempo promedio')

    # Grafica los puntos de tiempo promedio
    ax[0].fill(single_avg_points[single_avg_hull.vertices, 0], single_avg_points[single_avg_hull.vertices, 1], 'r', alpha=0.15, label='Area de valores predichos de tiempo promedio (single)')
    ax[0].scatter(y_test.iloc[:, 0], y_pred_single.iloc[:, 0], color='red', label='Valores predichos de tiempo promedio (single)')

    # Grafica los puntos de tiempo promedio
    ax[0].fill(multiple_avg_points[multiple_avg_hull.vertices, 0], multiple_avg_points[multiple_avg_hull.vertices, 1], 'b', alpha=0.15, label='Area de valores predichos de tiempo promedio (multiple)')
    ax[0].scatter(y_test.iloc[:, 0], y_pred_multiple.iloc[:, 0],  marker='x', color='blue', label='Valores predichos de tiempo promedio (multiple)')

    # Grafica los puntos de tiempo promedio
    ax[0].fill(global_avg_points[global_avg_hull.vertices, 0], global_avg_points[global_avg_hull.vertices, 1], 'g', alpha=0.15, label='Area de valores predichos de tiempo promedio (global)')
    ax[0].scatter(y_test.iloc[:, 0], y_pred_global.iloc[:, 0], color='green',  marker='^', label='Valores predichos de tiempo promedio (global)')

    # Grafica los puntos de tiempo promedio ideal
    ax[1].plot(y_test.iloc[:, 1], y_test.iloc[:, 1], color='black', label='Valor ideal intervalo inferior')

    # Grafica los puntos de intervalo inferior de tiempo promedio
    ax[1].fill(single_avg_l95ci_points[single_avg_l95ci_hull.vertices, 0], single_avg_l95ci_points[single_avg_l95ci_hull.vertices, 1] , 'r', alpha=0.15, label='Area de valores predichos de intervalo inferior de tiempo promedio (single)')
    ax[1].scatter(y_test.iloc[:, 1], y_pred_single.iloc[:, 1],  color='red', label='Valores predichos de intervalo inferior de tiempo promedio (single)')

    # Grafica los puntos de intervalo inferior de tiempo promedio
    ax[1].fill(multiple_avg_l95ci_points[multiple_avg_l95ci_hull.vertices, 0], multiple_avg_l95ci_points[multiple_avg_l95ci_hull.vertices, 1], 'b', alpha=0.15, label='Area de valores predichos de intervalo inferior de tiempo promedio (multiple)')
    ax[1].scatter(y_test.iloc[:, 1], y_pred_multiple.iloc[:, 1], marker='x', color='blue', label='Valores predichos de intervalo inferior de tiempo promedio (multiple)')

    # Grafica los puntos de intervalo inferior de tiempo promedio
    ax[1].fill(global_avg_l95ci_points[global_avg_l95ci_hull.vertices, 0], global_avg_l95ci_points[global_avg_l95ci_hull.vertices, 1], 'g', alpha=0.15, label='Area de valores predichos de intervalo inferior de tiempo promedio (global)')
    ax[1].scatter(y_test.iloc[:, 1], y_pred_global.iloc[:, 1], color='green',  marker='^', label='Valores predichos de intervalo inferior de tiempo promedio (global)')

    # Grafica los puntos de tiempo promedio ideal
    ax[2].plot(y_test.iloc[:, 2], y_test.iloc[:, 2], color='black', label='Valor ideal intervalo superior')

    # Grafica los puntos de intervalo superior de tiempo promedio
    ax[2].fill(single_avg_u95ci_points[single_avg_u95ci_hull.vertices, 0], single_avg_u95ci_points[single_avg_u95ci_hull.vertices, 1], 'r', alpha=0.15, label='Area de valores predichos de intervalo superior de tiempo promedio (single)')
    ax[2].scatter(y_test.iloc[:, 2], y_pred_single.iloc[:, 2], color='red', label='Valores predichos de intervalo superior de tiempo promedio (single)')

    # Grafica los puntos de intervalo superior de tiempo promedio
    ax[2].fill(multiple_avg_u95ci_points[multiple_avg_u95ci_hull.vertices, 0], multiple_avg_u95ci_points[multiple_avg_u95ci_hull.vertices, 1], 'b', alpha=0.15, label='Area de valores predichos de intervalo superior de tiempo promedio (multiple)')
    ax[2].scatter(y_test.iloc[:, 2], y_pred_multiple.iloc[:, 2], marker='x', color='blue', label='Valores predichos de intervalo superior de tiempo promedio (multiple)')

    # Grafica los puntos de intervalo superior de tiempo promedio
    ax[2].fill(global_avg_u95ci_points[global_avg_u95ci_hull.vertices, 0], global_avg_u95ci_points[global_avg_u95ci_hull.vertices, 1], 'g', alpha=0.15, label='Area de valores predichos de intervalo superior de tiempo promedio (global)')
    ax[2].scatter(y_test.iloc[:, 2], y_pred_global.iloc[:, 2], color='green',  marker='^', label='Valores predichos de intervalo superior de tiempo promedio (global)')

    # Agrega una leyenda
    ax[0].legend(fontsize='10')
    ax[1].legend(fontsize='10')
    ax[2].legend(fontsize='10')

    # Agrega etiquetas a los ejes
    for i in range(3):
      ax[i].set_xlabel('Tiempo real', fontweight='bold')
      ax[i].set_ylabel('Tiempo predicho', fontweight='bold')

    # Agrega un titulo
    fig.suptitle('Resultados de predicción para ' + model + ' con ' + max_model, fontweight='bold')

    # Configura el layout
    fig.set_layout_engine('compressed')

    # Guarda la figura
    fig.savefig(os.path.join(st.PLOTS_DIR, model + '.png'))

    # Resetear los valores de y_pred_single, y_pred_multiple y y_pred_global
    y_pred_single = pd.DataFrame()
    y_pred_multiple = pd.DataFrame()
    y_pred_global = pd.DataFrame()
    columns_Y = []

    # Cerrar la figura
    plt.close(fig)