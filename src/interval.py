# This model predicts confidence intervals

import sys
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Carga las funciones de utilidad
sys.path.append(os.path.abspath(os.path.join('../utils')))
from load_data import load_data

DATA_PATH = '../data/2021-07-22 datos_clusterizados_ML.xlsx'

# Predict confidence intervals from linear model
def interval_model(df):
  # Selecciona las columnas a utilizar
  X = df[['HbA1c', 'InitAge', 'Duration']]
  y = df[['L95CI_TIME_TO_PRET', 'U95CI_TIME_TO_PRET']]
  # Divide el conjunto de datos en entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Entrena el modelo usando validación cruzada
  model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
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
  results.to_excel('../out/interval_model.xlsx', index=False)

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
  plt.savefig('../figures/interval_model.png')

  # Guarda el modelo
  joblib.dump(model, '../model/interval_model.pkl')

if __name__ == '__main__':
  df = load_data(DATA_PATH)
  interval_model(df)