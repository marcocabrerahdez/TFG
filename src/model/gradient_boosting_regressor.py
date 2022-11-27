''' Modelo de regresión de aumento de gradiente.
    Se utiliza para predecir los intervalos de confianza de los pacientes con diabetes tipo 1.

    Parámetros:
        HbA1c (float): Nivel de hemoglobina glicosilada.
        InitAge (int): Edad al inicio del tratamiento.
        Duration (int): Duración del tratamiento.

    Retorna:
        (float): Intervalo de confianza del paciente.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import plot as utils_plot
from utils import save as utils_save

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_pinball_loss

def gradient_boosting_regressor_model(df: pd.DataFrame) -> None:
  '''
  Entrena un modelo de Gradient Boosting Regressor y guarda los resultados en un archivo xlsx.

  Parámetros:
      df (pd.DataFrame): DataFrame con los datos.
  '''
  # Selecciona las columnas a utilizar
  X = df[['HbA1c', 'InitAge', 'Duration']]
  y = df['AVG_TIME_TO_PRET']

  # Divide el conjunto de datos en entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Parametros del modelo
  all_models = {}
  params = dict(
    learning_rate = 0.05,
    n_estimators = 100,
    max_depth = 5,
    min_samples_split = 9,
    min_samples_leaf = 9,
  )
  # Entrena el modelo usando validación cruzada en 3 intervalos [0.05, 0.5, 0.95]
  for i in [0.05, 0.5, 0.95]:
    model = GradientBoostingRegressor(loss='quantile', alpha=i, **params)
    scores = cross_val_score(model, X_train, y_train, cv=10)

    # Entrena el modelo
    all_models["q %1.2f" % i] = model.fit(X_train, y_train)

  y_lower = all_models['q 0.05'].predict(X_test)
  y_upper = all_models['q 0.95'].predict(X_test)
  y_med = all_models['q 0.50'].predict(X_test)

  # Calcula el error cuadrático medio de los intervalos
  mse_lower = mean_squared_error(y_test, y_lower)
  mse_upper = mean_squared_error(y_test, y_upper)
  mse_med = mean_squared_error(y_test, y_med)

  # Calcula el error de pinball medio de los intervalos
  mpb_lower = mean_pinball_loss(y_test, y_lower)
  mpb_upper = mean_pinball_loss(y_test, y_upper)
  mpb_med = mean_pinball_loss(y_test, y_med)

  # Calcula el coeficiente de determinación de la predicción
  r2_lower = all_models["q 0.05"].score(X_test, y_test)
  r2_upper = all_models["q 0.95"].score(X_test, y_test)
  r2_med = all_models["q 0.50"].score(X_test, y_test)

  # Guarda los resultados en un archivo xlsx
  results = pd.DataFrame({
    'mse_lower': [mse_lower],
    'mse_upper': [mse_upper],
    'mse_med': [mse_med],
    'mpb_lower': [mpb_lower],
    'mpb_upper': [mpb_upper],
    'mpb_med': [mpb_med],
    'r2_lower': [r2_lower],
    'r2_upper': [r2_upper],
    'r2_med': [r2_med],
  })
  utils_save.save_results(results, 'Regresión de Aumento de Gradiente')

  # Grafica los resultados
  utils_plot.plot_gradient_boosting_regressor(y_test, y_lower, y_upper, y_med, 'Regresión de Aumento de Gradiente')

  # Guarda el modelo
  utils_save.save_model(all_models, 'Regresión de Aumento de Gradiente')