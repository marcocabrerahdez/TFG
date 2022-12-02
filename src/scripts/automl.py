''' Clase que automatiza el proceso de entrenamiento de modelos de Machine Learning.
'''

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import settings as st

from typing import List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
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
  __slots__ = ['_name', '_model', '_df', '_params', '_columns_X', '_columns_Y', '_X_train', '_X_test', '_y_train', '_y_test', '_scoring']

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



  def train(self) -> None:
    ''' Entrena los modelos usando validación cruzada. '''
    # Crea un pipeline con el modelo y los parámetros
    pipeline = Pipeline([('model', self._model)])
    pipeline.set_params(**self._params)

    # Entrena el modelo usando validación cruzada
    scores = cross_val_score(pipeline, self._X_train, self._y_train, cv=10, scoring=self._scoring)
    pipeline.fit(self._X_train, self._y_train)



  def predict(self) -> None:
    ''' Predice con los modelos. '''
    y_pred = self._model.predict(self._X_test)

    # Calcula el error cuadrático medio
    mse = mean_squared_error(self._y_test, y_pred)

    # Calcula el coeficiente de determinación
    r2 = r2_score(self._y_test, y_pred)

    # Crea un dataframe con los resultados
    df_results = pd.DataFrame({'MSE': [mse], 'R2': [r2]})
    df_results.to_excel(os.path.join(st.PREDICTIONS_DIR, f'{self._name}_{self._model.__class__.__name__}.xlsx'), index=False)



  def save(self) -> None:
    ''' Guarda los modelos. '''
    model_path = os.path.join(st.MODEL_DIR, f'{self._name}_{self._model.__class__.__name__}.pkl')
    joblib.dump(self._model, model_path)



  def plot(self) -> None:
    ''' Grafica los modelos. '''
    y_pred = self._model.predict(self._X_test)

    # Grafica los valores reales frente a los valores predichos
    plt.figure(figsize=(10, 10))
    plt.scatter(self._y_test, y_pred, c='blue', marker='o', label='Test data')
    plt.title('Real vs Predicted')
    plt.xlabel('Real')
    plt.ylabel('Predicted')
    plt.legend(loc='upper left')
    plt.plot([self._y_test.min(), self._y_test.max()], [self._y_test.min(), self._y_test.max()], 'k--', lw=4)
    plt.savefig(os.path.join(st.PLOTS_DIR, f'{self._name}_{self._model.__class__.__name__}.png'))



  def run(self) -> None:
    ''' Ejecuta el proceso de entrenamiento. '''
    self.train()
    self.predict()
    self.save()
    self.plot()