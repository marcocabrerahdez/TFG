import sys
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Carga las funciones de utilidad
sys.path.append(os.path.abspath(os.path.join('../utils')))
from load_data import load_data

DATA_PATH = '../data/2021-07-22 datos_clusterizados_ML.xlsx'

# Crea un modelo de regresión lineal
def linear_model(df):
  # Selecciona las columnas a utilizar
  X = df[['HbA1c', 'InitAge', 'Duration']]
  y = df['AVG_TIME_TO_PRET']
  # Divide el conjunto de datos en entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  '''
  # Normaliza los datos
  X_train_mean = X_train.mean(axis=0)
  X_train_std = X_train.std(axis=0)
  X_train = (X_train - X_train_mean) / X_train_std
  X_test = (X_test - X_train_mean) / X_train_std
  '''
  # Usar validación cruzada para evaluar el modelo
  model = LinearRegression()
  scores = cross_val_score(model, X_train, y_train, cv=10)
  print('Cross-Validation Scores: ', scores)
  print('Mean Cross-Validation Score: ', np.mean(scores))
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
  # Imprime los resultados
  print('Error cuadrático medio: ', mse)
  print('Raíz del error cuadrático medio: ', rmse)
  print('Coeficiente de determinación: ', r2)
  # Muestra una gráfica de los resultados
  plt.scatter(y_test, y_pred)
  # Valores reales en color azul
  plt.plot(y_test, y_test, color='blue')
  # Valores predichos en color rojo
  plt.plot(y_test, y_pred, color='red')
  # Guarda la gráfica
  plt.savefig('../figures/linear_model.png')
  plt.show()
  # Guarda el modelo
  joblib.dump(model, '../save/linear_model.pkl')

if __name__ == '__main__':
  # Carga los datos
  df = load_data(DATA_PATH)
  # Ejectuta el modelo
  linear_model(df)
