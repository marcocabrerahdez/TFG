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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
        y_pred (pd.DataFrame): Datos de salida predichos.
        model_list_names (List[str]): Lista de nombres de modelos.
  '''
  __slots__ = ['_name', '_class_name', '_model', '_type', '_params', '_columns_X', '_columns_Y', '_X_train', '_X_test', '_y_train', '_y_test', '_y_pred', '_model_list_names']

  def __init__(self, name: str, class_name, model, type: str, params, columns_X: pd.DataFrame, _columns_Y: pd.DataFrame) -> None:
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
    self._model_list_names = model
    self._class_name = [importlib.import_module(class_name[i]) for i in range(len(class_name))]
    self._model = [getattr(self._class_name[i], model[i])() for i in range(len(model))]
    self._type = type
    self._params = params
    self._columns_X = columns_X
    self._columns_Y = _columns_Y
    self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._columns_X, self._columns_Y, test_size=st.TEST_SIZE, random_state=st.RANDOM_STATE)
    self._y_pred = None



  def train(self) -> None:
    if self._type == 'single':
      self.train_single()
    elif self._type == 'multiple' or self._type == 'global':
      self.train_multioutput()



  def train_single(self) -> None:
    ''' Entrena un conjunto de modelos '''
    # Crea un pipeline con el modelo y los parámetros del parametro grid
    pipe = [Pipeline([('model', self._model[i])]) for i in range(len(self._model))]

    # Crea un grid search con el pipeline y los parámetros
    grid_search = [GridSearchCV(pipe[i], param_grid=self._params[i], cv=5, refit=True, scoring='neg_mean_absolute_error') for i in range(len(pipe))]

    # Entrena el modelo
    for i in range(len(grid_search)):
      grid_search[i].fit(self._X_train, self._y_train.values.ravel())

    # Asigna el mejor modelo a la variable _model
    self._model = [grid_search[i].best_estimator_ for i in range(len(grid_search))]



  def train_multioutput(self) -> None:
    ''' Entrena un conjunto de modelos '''
    # Crea un pipeline con el modelo y los parámetros del parametro grid
    pipe = [Pipeline([('model', self._model[i])]) for i in range(len(self._model))]

    # Crea un grid search con el pipeline y los parámetros
    grid_search = [GridSearchCV(pipe[i], param_grid=self._params[i], cv=5, refit=True, scoring='neg_mean_squared_error') for i in range(len(pipe))]

    # Crea un Multioutput Regressor con el grid search
    self._model = [MultiOutputRegressor(grid_search[i]) for i in range(len(grid_search))]

    # Entrena el modelo
    for i in range(len(self._model)):
      self._model[i].fit(self._X_train, self._y_train)



  def predict(self) -> None:
    ''' Predice con los modelos. '''
    self._y_pred = [self._model[i].predict(self._X_test) for i in range(len(self._model))]

    # Calcula el R2 y el error cuadrático medio de cada salida del modelo
    if self._type == 'single':
      r2 = [r2_score(self._y_test, self._y_pred[i]) for i in range(len(self._model))]
      mse = [mean_squared_error(self._y_test, self._y_pred[i]) for i in range(len(self._model))]
      mae = [mean_absolute_error(self._y_test, self._y_pred[i]) for i in range(len(self._model))]
    elif self._type == 'multiple' or self._type == 'global':
      r2 = [[r2_score(self._y_test.iloc[:, i], self._y_pred[j][:, i]) for i in range(self._y_test.shape[1])] for j in range(len(self._model))]
      mse = [[mean_squared_error(self._y_test.iloc[:, i], self._y_pred[j][:, i]) for i in range(self._y_test.shape[1])] for j in range(len(self._model))]
      mae = [[mean_absolute_error(self._y_test.iloc[:, i], self._y_pred[j][:, i]) for i in range(self._y_test.shape[1])] for j in range(len(self._model))]

    # Crea un dataframe con los resultados
    df_results = [pd.DataFrame({
      'Enfermedad': self._y_test.columns,
      'MSE': mse[i],
      'R2': r2[i],
      'MAE': mae[i],
      'Tipo': self._type,
    }) for i in range(len(self._model))]
    for i in range(len(self._model)):
      if self._type == 'single':
        if not os.path.exists(os.path.join(st.SINGLE_PREDICTIONS_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.SINGLE_PREDICTIONS_DIR, self._model_list_names[i]))
        df_results[i].to_excel(os.path.join(st.SINGLE_PREDICTIONS_DIR, self._model_list_names[i], f'{self._name}.xlsx'), index=False)
      elif self._type == 'multiple':
        if not os.path.exists(os.path.join(st.MULTIPLE_PREDICTIONS_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.MULTIPLE_PREDICTIONS_DIR, self._model_list_names[i]))
        df_results[i].to_excel(os.path.join(st.MULTIPLE_PREDICTIONS_DIR, self._model_list_names[i], f'{self._name}.xlsx'), index=False)
      elif self._type == 'global':
        if not os.path.exists(os.path.join(st.GLOBAL_PREDICTIONS_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.GLOBAL_PREDICTIONS_DIR, self._model_list_names[i]))
        df_results[i].to_excel(os.path.join(st.GLOBAL_PREDICTIONS_DIR, self._model_list_names[i], f'{self._name}.xlsx'), index=False)



  def save(self) -> None:
    ''' Guarda los modelos. '''
    model_path = []
    for i in range(len(self._model)):
      if self._type == 'single':
        if not os.path.exists(os.path.join(st.SINGLE_MODEL_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.SINGLE_MODEL_DIR, self._model_list_names[i]))
        model_path.append(os.path.join(st.SINGLE_MODEL_DIR, self._model_list_names[i], f'{self._name}.pkl'))
      elif self._type == 'multiple':
        if not os.path.exists(os.path.join(st.MULTIPLE_MODEL_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.MULTIPLE_MODEL_DIR, self._model_list_names[i]))
        model_path.append(os.path.join(st.MULTIPLE_MODEL_DIR, self._model_list_names[i], f'{self._name}.pkl'))
      elif self._type == 'global':
        if not os.path.exists(os.path.join(st.GLOBAL_MODEL_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.GLOBAL_MODEL_DIR, self._model_list_names[i]))
        model_path.append(os.path.join(st.GLOBAL_MODEL_DIR, self._model_list_names[i], f'{self._name}.pkl'))

      joblib.dump(self._model[i], model_path[i])



  def plot(self) -> None:
      ''' Grafica los modelos.
      '''
      if self._type == 'single':
        self.plot_single_results()
      elif self._type == 'multiple' or self._type == 'global':
        self.plot_multiple_results()



  def plot_single_results(self) -> None:
    ''' Grafica los modelos.
      La gráfica representa los valores reales contra los valores predichos.
    '''
    for j in range(len(self._model_list_names)):
      # Grafica los resultados
      plt.figure(figsize=(10, 10))

      # Valores reales en color azul y valores predichos en color rojo
      plt.plot(self._y_test, self._y_test, 'b-')
      plt.plot(self._y_test, self._y_pred[j], 'ro')

      # Título y etiquetas
      plt.title(f'{self._name} - {self._model_list_names[j]}')
      plt.xlabel('Real')
      plt.ylabel('Predicho')
      if self._type == 'single':
        if not os.path.exists(os.path.join(st.SINGLE_PLOTS_DIR, self._model_list_names[j])):
          os.makedirs(os.path.join(st.SINGLE_PLOTS_DIR, self._model_list_names[j]))
        plt.savefig(os.path.join(st.SINGLE_PLOTS_DIR, self._model_list_names[j], f'{self._name}.png'))

      # Cierra la gráfica
      plt.close()



  def plot_multiple_results(self) -> None:
    ''' Grafica los modelos.
      La gráfica representa los valores reales contra los valores predichos.
    '''
    for j in range(len(self._model_list_names)):
      # Grafica los resultados
      fig, ax = plt.subplots(figsize=(30, 30), nrows=self._y_test.shape[1], ncols=1)

      # Crear una gráfico de barras para cada salida
      if self._type == 'multiple' or self._type == 'global':
        for i in range(self._y_test.shape[1]):
          # Crear una gráfico de scatter para cada salida del modelo con colores azul y rojo
          ax[i].plot(self._y_test.iloc[:, i], self._y_test.iloc[:, i], 'b-')
          ax[i].plot(self._y_test.iloc[:, i], self._y_pred[j][:, i], 'ro')

          # Agrega los títulos
          ax[i].set_title(self._y_test.columns[i])

          # Agrega las etiquetas
          ax[i].set_xlabel('Real')
          ax[i].set_ylabel('Predicho')

          # Agrega la leyenda
          ax[i].legend(['Real', 'Predicho'])

      # Agrega los títulos a la gráfica
      fig.suptitle(self._name + ' - ' + self._model_list_names[j])

      # Ajustar el espacio entre subgráficas
      fig.tight_layout()

      # Ajustar el espacio entre subgráficas y el título
      fig.subplots_adjust(top=0.95)

      # Guarda la gráfica
      if self._type == 'multiple':
        if not os.path.exists(os.path.join(st.MULTIPLE_PLOTS_DIR, self._model_list_names[j])):
          os.makedirs(os.path.join(st.MULTIPLE_PLOTS_DIR, self._model_list_names[j]))
        plt.savefig(os.path.join(st.MULTIPLE_PLOTS_DIR, self._model_list_names[j], f'{self._name}.png'))
      elif self._type == 'global':
        if not os.path.exists(os.path.join(st.GLOBAL_PLOTS_DIR, self._model_list_names[j])):
          os.makedirs(os.path.join(st.GLOBAL_PLOTS_DIR, self._model_list_names[j]))
        plt.savefig(os.path.join(st.GLOBAL_PLOTS_DIR, self._model_list_names[j], f'{self._name}.png'))

      # Cierra la gráfica
      plt.close()