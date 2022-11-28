''' Guarda una gráfica con los resultados. '''

import os
import matplotlib.pyplot as plt
import settings as st

def plot_linear_model(y_test, y_pred, filename) -> None:
  '''
  Guarda una gráfica con los resultados de la regresión lineal.

  Parámetros:
      y_test (str): Valores reales.
      y_pred (str): Valores predichos.
      filename (str): Nombre del archivo.
  '''
  # Muestra una gráfica de los resultados
  plt.scatter(y_test, y_pred)

  # Valores reales en color azul
  plt.scatter(y_test, y_test, color='blue')

  # Valores predichos en color rojo
  plt.scatter(y_test, y_pred, color='red')

  # Guarda la gráfica
  result_location = os.path.join(st.FIGURES_DIR, filename) + '.png'
  with open(result_location, "wb") as f:
    plt.savefig(result_location)



def plot_randomForestRegressor_model(y_test, y_pred, filename) -> None:
  '''
  Guarda una gráfica con los resultados del modelo RandomForestRegressor.

  Parámetros:
      y_test (str): Valores reales.
      y_pred (str): Valores predichos.
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