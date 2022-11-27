''' Modelo de bosque aleatorio de regresión.
    Se utiliza para predecir los intervalos de confianza de la variable AVG_TIME_TO_PRET.

    Parámetros:
        HbA1c (float): Nivel de hemoglobina glicosilada.
        InitAge (int): Edad al inicio del tratamiento.
        Duration (int): Duración del tratamiento.

    Retorna:
        L95CI_TIME_TO_PRET (float): Límite inferior del intervalo de confianza.
        U95CI_TIME_TO_PRET (float): Límite superior del intervalo de confianza.

    Nota:
        El modelo de regresión lineal no puede predecir intervalos de confianza.
        Por lo tanto, se utiliza un modelo de Random Forest Regressor.
'''

import numpy as np
import pandas as pd

from utils import plot as utils_plot
from utils import save as utils_save

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def random_forest_regressor_model(df: pd.DataFrame) -> None:
  '''
  Entrena un modelo de Random Forest Regressor y guarda los resultados en un archivo xlsx.

  Parámetros:
      df (pd.DataFrame): DataFrame con los datos.
  '''
  # Selecciona las columnas a utilizar
  X = df[['HbA1c', 'InitAge', 'Duration']]
  y = df[['L95CI_TIME_TO_PRET', 'U95CI_TIME_TO_PRET']]

  # Divide el conjunto de datos en entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Entrena el modelo usando validación cruzada
  model = RandomForestRegressor(n_estimators=100, max_depth=5)
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
  r2 = model.score(X_test, y_test)

  # Guardar los resultados y los errores en una tabla
  results = pd.DataFrame({
    'y_L95CI_test': y_test['L95CI_TIME_TO_PRET'],
    'y_L95CI_pred': y_pred[:, 0],
    'y_U95CI_test': y_test['U95CI_TIME_TO_PRET'],
    'score': r2,
    'mse': mse,
    'rmse': rmse
  })
  utils_save.save_results(results, 'Bosque Aleatorio de Regresión')

  # Grafica los resultados
  utils_plot.plot_randomForestRegressor_model(y_test, y_pred, 'Bosque Aleatorio de Regresión')

  # Guarda el modelo
  utils_save.save_model(model, 'Bosque Aleatorio de Regresión')