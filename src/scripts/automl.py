''' Clase que automatiza el proceso de entrenamiento de modelos de Machine Learning.
'''

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import settings as st

from typing import List
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

class AutoML(object):
  ''' Clase que automatiza el proceso de entrenamiento de modelos de Machine Learning.

      Atributos:
        name (str): Nombre del modelo.
        model (str): Modelo de Machine Learning.
        params (hash): Parámetros del modelo.
        df (pd.DataFrame): Datos de entrada.
        columns_X (List[str]): Columnas de entrada.
        columns_Y (List[str]): Columnas de salida.
        X_train (pd.DataFrame): Datos de entrenamiento.
        X_test (pd.DataFrame): Datos de prueba.
        y_train (pd.DataFrame): Datos de salida de entrenamiento.
        y_test (pd.DataFrame): Datos de salida de prueba.
        scoring (str): Métrica de evaluación.
  '''
  __slots__ = ['_name', '_model', '_df', '_params', '_columns_X', '_columns_Y', '_X_train', '_X_test', '_y_train', '_y_test', '_scoring', '_y_pred']

  def __init__(self, name: str, df: pd.DataFrame, model: str, params: hash, columns_X: List[str], columns_Y: List[str], scoring: str = 'neg_mean_squared_error') -> None:
    ''' Constructor de la clase.
      Parámetros:
          name (str): Nombre del modelo.
          df (pd.DataFrame): Datos de entrada.
          model (str): Modelo de Machine Learning.
          params (hash): Parámetros del modelo.
          columns_X (List[str]): Columnas de entrada.
          columns_Y (List[str]): Columnas de salida.
          scoring (str): Métrica de evaluación.
    '''
    # Asigna los atributos
    if isinstance(model, str):
      if model == 'LinearRegression':
        model = LinearRegression()
      elif model == 'RandomForestRegressor':
        model = RandomForestRegressor()
      elif model == 'SVR':
        model = SVR()
      else:
        raise Exception('Model not found')

    self._model = model
    self._name = name
    self._df = df
    self._params = params
    self._columns_X = columns_X
    self._columns_Y = columns_Y
    self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._df[self._columns_X], self._df[self._columns_Y], test_size=0.2, random_state=42)
    self._scoring = scoring
    self._y_pred = None



  def train(self) -> None:
    ''' Entrena el modelo '''
    # Crea un pipeline con el modelo y los parámetros y GridSearchCV
    pipe = Pipeline([('model', self._model)])
    grid = GridSearchCV(pipe, self._params, cv=5, scoring=self._scoring, refit=True)

    # Entrena el modelo
    grid.fit(self._X_train, self._y_train)

    # Asigna el modelo entrenado
    self._model = grid.best_estimator_



  def predict(self) -> None:
    ''' Predice con los modelos. '''
    self._y_pred = self._model.predict(self._X_test)

    # Calcula el error cuadrático medio
    mse = mean_squared_error(self._y_test, self._y_pred)

    # Calcula el coeficiente de determinación
    r2 = r2_score(self._y_test, self._y_pred)

    # Crea un dataframe con los resultados
    df_results = pd.DataFrame({'MSE': [mse], 'R2': [r2], 'PARAMS': [self._model.get_params()]})
    df_results.to_excel(os.path.join(st.PREDICTIONS_DIR, f'{self._name}.xlsx'), index=False)



  def save(self) -> None:
    ''' Guarda los modelos. '''
    model_path = os.path.join(st.MODEL_DIR, f'{self._name}.pkl')
    joblib.dump(self._model, model_path)



  def plot(self) -> None:
    ''' Grafica los modelos.
      La gráfica representa los valores reales contra los valores predichos.
    '''
    # Grafica los resultados
    plt.figure(figsize=(10, 10))

    # Valores reales en color azul y valores predichos en color rojo
    plt.plot(self._y_test, self._y_test, 'b-')
    plt.plot(self._y_test, self._y_pred, 'ro')
    plt.title('Real vs Predicho')
    plt.xlabel('Real')
    plt.ylabel('Predicho')
    plt.savefig(os.path.join(st.PLOTS_DIR, f'{self._name}.png'))
    plt.close()



  def compare(self, model_list_names, plot_name) -> None:
    ''' Muestra una gráfica de comparación de modelos. '''
    # Obtener el coeficiente de determinación de cada modelo del directorio de predicciones
    df_results = pd.DataFrame()
    for model_name in model_list_names:
      df_results = pd.concat([df_results, pd.read_excel(os.path.join(st.PREDICTIONS_DIR, f'{model_name}.xlsx'))])
    df_results.index = model_list_names

    # Grafica los resultados
    plt.figure(figsize=(10, 10))
    plt.plot(df_results['R2'], 'o-', label='R2')
    plt.title('R2')
    plt.xlabel('Modelo')
    plt.ylabel('R2')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(st.COMPARISION_DIR, f'{plot_name}.png'))
    plt.close()



  def run(self) -> None:
    ''' Ejecuta el proceso de entrenamiento. '''
    self.train()
    self.predict()
    self.save()
    self.plot()