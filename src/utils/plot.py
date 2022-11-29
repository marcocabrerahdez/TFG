''' Guarda una gráfica con los resultados. '''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import settings as st

def plot_linear_model(y_test, y_pred, filename) -> None:
  '''
  Guarda una gráfica con los resultados de la regresión lineal.

  Parámetros:
      y_test (Array): Valores reales.
      y_pred (Array): Valores predichos.
      filename (str): Nombre del archivo.
  '''
  # Displays a graph and an histogram on the same graph
  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.suptitle('Histograma y gráfica de dispersión')

  # Gráfica de dispersión
  ax1.scatter(y_test, y_test, color='blue')
  ax1.scatter(y_test, y_pred, color='orange')
  ax1.set_xlabel('Valores reales')
  ax1.set_ylabel('Valores predichos')
  ax1.set_title('Gráfica de dispersión')
  ax1.legend(['Valores reales', 'Valores predichos'])

  # Histogram with both values (real and predicted)
  ax2.hist(y_test, bins=20, color='blue', alpha=0.5)
  ax2.hist(y_pred, bins=20, color='orange', alpha=0.5)
  ax2.set_xlabel('Valores')
  ax2.set_ylabel('Frecuencia')
  ax2.set_title('Histograma')
  ax2.legend(['Valores reales', 'Valores predichos'])

  # Guarda la gráfica
  result_location = os.path.join(st.FIGURES_DIR, filename) + '.png'
  with open(result_location, "wb") as f:
    plt.savefig(result_location)



def plot_randomForestRegressor_model(y_test, y_pred, filename) -> None:
  '''
  Guarda una gráfica con los resultados del modelo RandomForestRegressor.

  Parámetros:
      y_test (Array): Valores reales.
      y_pred (Array): Valores predichos.
      filename (str): Nombre del archivo.
  '''
  # Crear dos gráficas, una para L95CI y otra para U95CI
  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.suptitle('L95CI and U95CI')

  # Gráfica L95CI
  ax1.scatter(y_test['L95CI_TIME_TO_PRET'], y_test['L95CI_TIME_TO_PRET'], color='blue')
  ax1.scatter(y_test['L95CI_TIME_TO_PRET'], y_pred[:, 0], color='orange')
  ax1.set_xlabel('Valores reales')
  ax1.set_ylabel('Valores predichos')
  ax1.set_title('L95CI')
  ax1.legend(['L95CI', 'L95CI predicho'])

  # Gráfica U95CI
  ax2.scatter(y_test['U95CI_TIME_TO_PRET'], y_test['U95CI_TIME_TO_PRET'], color='red')
  ax2.scatter(y_test['U95CI_TIME_TO_PRET'], y_pred[:, 1], color='green')
  ax2.set_xlabel('Valores reales')
  ax2.set_ylabel('Valores predichos')
  ax2.set_title('U95CI')
  ax2.legend(['U95CI', 'U95CI predicho'])

  # Guarda la gráfica
  result_location = os.path.join(st.FIGURES_DIR, filename) + '.png'
  with open(result_location, "wb") as f:
    plt.savefig(result_location)



def plot_correlations(df: pd.DataFrame, filename: str) -> None:
  '''
  Guarda una gráfica con las correlaciones entre las variables.

  Parámetros:
      df (DataFrame): DataFrame con los datos.
      filename (str): Nombre del archivo.
  '''

  # Calcula las correlaciones de las variables
  corr = df[['HbA1c', 'InitAge', 'Duration', 'AVG_TIME_TO_PRET', 'L95CI_TIME_TO_PRET', 'U95CI_TIME_TO_PRET']].corr()

  # Muestra un gráfico con las correlaciones
  fig, ax = plt.subplots(figsize=(20, 20))
  fig.suptitle('Correlaciones entre las variables')

  # Genera la matriz de correlaciones
  ax.matshow(corr)

  # Genera los nombres de las variables
  labels = df[['HbA1c', 'InitAge', 'Duration', 'AVG_TIME_TO_PRET', 'L95CI_TIME_TO_PRET', 'U95CI_TIME_TO_PRET']].columns

  # Muestra los nombres de las variables en el eje X
  ax.set_xticks(np.arange(len(labels)))
  ax.set_xticklabels(labels, rotation=90)

  # Muestra los nombres de las variables en el eje Y
  ax.set_yticks(np.arange(len(labels)))
  ax.set_yticklabels(labels)

  # Crea un colorbar
  fig.colorbar(ax.matshow(corr))

  # Recorre las dimensiones de los datos y cree anotaciones de texto.
  for i, j in np.ndindex(corr.shape):
    text = ax.text(j, i, corr.iloc[i, j], ha="center", va="center", color="black", fontsize=10)

  # Guarda la gráfica
  result_location = os.path.join(st.FIGURES_DIR, filename) + '.png'
  with open(result_location, "wb") as f:
    plt.savefig(result_location)