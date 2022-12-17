''' Clase que automatiza el proceso de entrenamiento de modelos de Machine Learning.
'''

import os
import joblib
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import settings as st

from typing import List

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
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
  __slots__ = ['_name', '_class_name', '_model', '_type', '_params', '_columns_X', '_columns_Y', '_X_train', '_X_test', '_y_train', '_y_test', '_y_pred']

  def __init__(self, name: str, class_name: str, model: str, type: str, params: hash, columns_X: pd.DataFrame, _columns_Y: pd.DataFrame) -> None:
    ''' Constructor de la clase.
      Parámetros:
        name (str): Nombre del modelo.
        class_name (str): Nombre de la clase del modelo.
        model (str): Modelo de Machine Learning.
        params (hash): Parámetros del modelo.
        columns_X (List[str]): Columnas de entrada.
        columns_y (List[str]): Columnas de salida.
    '''
    self._name = name
    self._class_name = importlib.import_module(class_name)
    self._model = getattr(self._class_name , model)()
    self._type = type
    self._params = params
    self._columns_X = columns_X
    self._columns_Y = _columns_Y
    self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._columns_X, self._columns_Y, test_size=st.TEST_SIZE, random_state=st.RANDOM_STATE)
    self._y_pred = None



  def train(self) -> None:
    if self._type == 'single':
      self.train_single()
    elif self._type == 'multiple':
      self.train_multioutput()



  def train_single(self) -> None:
    ''' Entrena un conjunto de modelos '''
    # Crea un pipeline con el modelo y los parámetros
    pipe = Pipeline([('model', self._model)])

    # Crea un grid search con el pipeline y los parámetros
    grid_search = GridSearchCV(pipe, self._params, cv=5, refit=True)

    # Entrena el modelo
    self._model = grid_search.fit(self._X_train, self._y_train)



  def train_multioutput(self) -> None:
    ''' Entrena un conjunto de modelos '''
    # Crea un pipeline con el modelo y los parámetros
    pipe = Pipeline([('model', self._model)])

    # Crea un grid search con el pipeline y los parámetros
    grid_search = GridSearchCV(pipe, self._params, cv=5, refit=True)

    # Crea un Multioutput Regressor con el grid search
    self._model = MultiOutputRegressor(grid_search)

    # Entrena el modelo
    self._model.fit(self._X_train, self._y_train)



  def predict(self) -> None:
    ''' Predice con los modelos. '''
    self._y_pred = self._model.predict(self._X_test)

    # Calcula el error cuadrático medio de cada salida del modelo
    if self._type == 'single':
      mse = mean_squared_error(self._y_test, self._y_pred)
    elif self._type == 'multiple':
      mse = [mean_squared_error(self._y_test.iloc[:, i], self._y_pred[:, i]) for i in range(self._y_test.shape[1])]

    # Calcula el coeficiente de determinación de cada salida del modelo
    if self._type == 'single':
      r2 = r2_score(self._y_test, self._y_pred)
    elif self._type == 'multiple':
      r2 = [r2_score(self._y_test.iloc[:, i], self._y_pred[:, i]) for i in range(self._y_test.shape[1])]

    # Crea un dataframe con los resultados
    df_results = pd.DataFrame({
      'Enfermedad': self._y_test.columns,
      'MSE': mse,
      'R2': r2
    })
    if self._type == 'single':
      df_results.to_excel(os.path.join(st.SINGLE_PREDICTIONS_DIR, f'{self._name}.xlsx'), index=False)
    elif self._type == 'multiple':
      df_results.to_excel(os.path.join(st.MULTIPLE_PREDICTIONS_DIR, f'{self._name}.xlsx'), index=False)



  def save(self) -> None:
    ''' Guarda los modelos. '''
    if self._type == 'single':
      model_path = os.path.join(st.SINGLE_MODEL_DIR, f'{self._name}.pkl')
    elif self._type == 'multiple':
      model_path = os.path.join(st.MULTIPLE_MODEL_DIR, f'{self._name}.pkl')
    joblib.dump(self._model, model_path)



  def plot_results(self) -> None:
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
    if self._type == 'single':
      plt.savefig(os.path.join(st.SINGLE_PLOTS_DIR, f'{self._name}.png'))
    elif self._type == 'multiple':
      plt.savefig(os.path.join(st.MULTIPLE_PLOTS_DIR, f'{self._name}.png'))

    # Cierra la gráfica
    plt.close()