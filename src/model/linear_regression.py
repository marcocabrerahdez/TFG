''' Modelo de regresión lineal.
    Se utiliza para predecir el tiempo promedio de inicio de tratamiento de pacientes con diabetes tipo 1.

    Parámetros:
        HbA1c (float): Nivel de hemoglobina glicosilada.
        InitAge (int): Edad al inicio del tratamiento.
        Duration (int): Duración del tratamiento.

    Retorna:
        AVG_TIME_TO_PRET (float): Tiempo promedio de inicio de tratamiento.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import plot as utils_plot
from utils import save as utils_save

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def linear_regression_model(df: pd.DataFrame) -> None:
  '''
  Entrena un modelo de regresión lineal y guarda los resultados en un archivo xlsx.

  Parámetros:
      df (pd.DataFrame): DataFrame con los datos.
  '''
  # Selecciona las columnas a utilizar
  X = df[['HbA1c', 'InitAge', 'Duration']]
  y = df['AVG_TIME_TO_PRET']

  # Divide el conjunto de datos en entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Crea el modelo de regresión lineal
  model = LinearRegression()

  # Usar validación cruzada para evaluar el modelo
  scores = cross_val_score(model, X_train, y_train, cv=10)

  # Entrena el modelo
  model.fit(X_train, y_train)

  # Evalúa el modelo
  y_pred = model.predict(X_test)

  # Calcula el error cuadrático medio
  mse = mean_squared_error(y_test, y_pred)

  # Calula la raíz del error cuadrático medio
  rmse = np.sqrt(mse)

  # Calcula el coeficiente de determinación de la predicción
  cd = model.score(X_test, y_test)

  # Guarda los resultados
  results = pd.DataFrame({
    'y_test': y_test,
    'y_pred': y_pred,
    'score': scores.std(),
    'coeficiente de determinacion': cd,
    'mse': mse,
    'rmse': rmse
  })
  utils_save.save_results(results, 'Regresión Lineal')

  # Guarda las gráficas
  utils_plot.plot_linear_model(y_test, y_pred, 'Regresión Lineal')

  # Guarda el modelo
  utils_save.save_model(model, 'Regresión Lineal')