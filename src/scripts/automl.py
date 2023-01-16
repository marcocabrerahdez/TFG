''' Clase que automatiza el proceso de entrenamiento
    de modelos de Machine Learning.
'''

import os
import importlib
import time
import joblib
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import settings as st
import utils as ut

class AutoML:
  ''' Clase que automatiza el proceso de entrenamiento
      de modelos de Machine Learning.

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
  __slots__ = [
    '_name',
    '_class_name',
    '_model',
    '_type',
    '_params',
    '_columns_X',
    '_columns_Y',
    '_X_train',
    '_X_test',
    '_y_train',
    '_y_test',
    '_y_pred',
    '_model_list_names',
    'time',
    'cpu',
    '_trained_data_names',
    '_metrics_results'
  ]

  def __init__(self, name: str, class_name, model, type_model: str, params,
              trained_data_names = [], columns_X: pd.DataFrame = pd.DataFrame(), columns_Y: pd.DataFrame = pd.DataFrame()) -> None:
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
    self._class_name = [importlib.import_module(class_name[i])
                        for i in range(len(class_name))]
    self._model = [getattr(self._class_name[i], model[i])()
                    for i in range(len(model))]
    self._type = type_model
    self._params = params
    self._trained_data_names = trained_data_names
    if self._type == 'single':
      self._columns_X = columns_X
      self._columns_Y = columns_Y
      # pylint: disable=line-too-long
      self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._columns_X, self._columns_Y, test_size=st.TEST_SIZE, random_state=st.RANDOM_STATE)
    elif self._type == 'multiple' or self._type == 'global':
      self._X_train, self._X_test, self._y_train, self._y_test, self._columns_X, self._columns_Y = ut.get_splited_data(self._trained_data_names, self._type)
    else:
      raise Exception('El tipo de modelo no es válido.')

    # Guardar los datos de entrenamiento y prueba
    ut.save_splitted_data(self._X_train, self._X_test, self._y_train, self._y_test, self._columns_X, self._columns_Y, self._name, self._type)

    self._y_pred = None
    self.time = None
    self.cpu = None
    self._metrics_results = []



  def train(self) -> None:
    ''' Entrena un conjunto de modelos '''
    # Medir el tiempo de ejecución
    # y el uso de recursos antes de entrenar el modelo
    start_time = time.perf_counter()
    process = psutil.Process()
    cpu_before = process.cpu_percent()

    ''' Entrena un conjunto de modelos '''
    # Crea un pipeline con el modelo y los parámetros del parametro grid
    pipe = [Pipeline([('model', self._model[i])])
            for i in range(len(self._model))]

    # Crea un grid search con el pipeline y los parámetros
    grid_search = [GridSearchCV(pipe[i], param_grid=self._params[i],
                    cv=5, refit=True, scoring='r2') for i in range(len(pipe))]

    # Crea un Multioutput Regressor con el grid search
    self._model = [MultiOutputRegressor(grid_search[i])
                    for i in range(len(grid_search))]

    # Entrena el modelo
    for i in range(len(self._model)):
      self._model[i].fit(self._X_train, self._y_train)
    # Medir el tiempo de ejecución
    # y el uso de recursos después de entrenar el modelo
    elapsed_time = time.perf_counter() - start_time
    cpu_after = process.cpu_percent()

    # Calcular el tiempo de ejecución y el uso de recursos
    self.time = elapsed_time / 60
    self.cpu = (cpu_after - cpu_before) / 100



  def predict(self) -> None:
    ''' Predice con los modelos. '''
    self._y_pred = [self._model[i].predict(self._X_test)
                    for i in range(len(self._model))]



  def metrics(self) -> None:
    # Calcula el R2 y el error cuadrático medio de cada salida del modelo
    r2 = [[r2_score(self._y_test.iloc[:, i], self._y_pred[j][:, i])
            for i in range(self._y_test.shape[1])]
              for j in range(len(self._model))]
    mse = [[mean_squared_error(self._y_test.iloc[:, i], self._y_pred[j][:, i])
            for i in range(self._y_test.shape[1])]
              for j in range(len(self._model))]
    mae = [[mean_absolute_error(self._y_test.iloc[:, i],self._y_pred[j][:, i])
            for i in range(self._y_test.shape[1])]
              for j in range(len(self._model))]

    # Crea un dataframe con los resultados de cada modelo
    self._metrics_results = [pd.DataFrame({
      'Enfermedad': self._y_test.columns,
      'MSE': mse[i],
      'R2': r2[i],
      'MAE': mae[i],
      'Tipo': self._type,
      'Modelo': self._model[i],
      'Elapsed Time': self.time,
      'CPU': self.cpu
    }) for i in range(len(self._model))]




  def _save_predictions_results(self) -> None:
    # Guarda las predicciones en un archivo excel
    for i in range(len(self._model)):
      if self._type == 'single':
        if not os.path.exists(os.path.join(st.SINGLE_PREDICTIONS_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.SINGLE_PREDICTIONS_DIR, self._model_list_names[i]))
        pd.DataFrame(self._y_pred[i]).to_excel(os.path.join(st.SINGLE_PREDICTIONS_DIR, self._model_list_names[i], f'{self._name}.xlsx'), index=False)

      elif self._type == 'multiple':
        if not os.path.exists(os.path.join(st.MULTIPLE_PREDICTIONS_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.MULTIPLE_PREDICTIONS_DIR, self._model_list_names[i]))
        pd.DataFrame(self._y_pred[i]).to_excel(os.path.join(st.MULTIPLE_PREDICTIONS_DIR, self._model_list_names[i], f'{self._name}.xlsx'), index=False)

      elif self._type == 'global':
        if not os.path.exists(os.path.join(st.GLOBAL_PREDICTIONS_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.GLOBAL_PREDICTIONS_DIR, self._model_list_names[i]))
        pd.DataFrame(self._y_pred[i]).to_excel(os.path.join(st.GLOBAL_PREDICTIONS_DIR, self._model_list_names[i], f'{self._name}.xlsx'), index=False)

      else:
        raise Exception('El tipo de modelo no es válido.')



  def _save_metrics_results(self) -> None:
    print("Length of metrics results: ", len(self._metrics_results))
    # Guarda los resultados en un archivo excel
    for i in range(len(self._model)):
      print(i)
      if self._type == 'single':
        if not os.path.exists(os.path.join(st.SINGLE_METRICS_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.SINGLE_METRICS_DIR, self._model_list_names[i]))
        self._metrics_results[i].to_excel(os.path.join(st.SINGLE_METRICS_DIR, self._model_list_names[i], f'{self._name}.xlsx'), index=False)

      elif self._type == 'multiple':
        if not os.path.exists(os.path.join(st.MULTIPLE_METRICS_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.MULTIPLE_METRICS_DIR, self._model_list_names[i]))
        self._metrics_results[i].to_excel(os.path.join(st.MULTIPLE_METRICS_DIR, self._model_list_names[i], f'{self._name}.xlsx'), index=False)
      elif self._type == 'global':
        if not os.path.exists(os.path.join(st.GLOBAL_METRICS_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.GLOBAL_METRICS_DIR, self._model_list_names[i]))
        self._metrics_results[i].to_excel(os.path.join(st.GLOBAL_METRICS_DIR, self._model_list_names[i], f'{self._name}.xlsx'), index=False)



  def _save_model(self) -> None:
    ''' Guarda los modelos. '''
    model_path = []
    for i in range((len(self._model))):
      if self._type == 'single':
        if not os.path.exists(os.path.join(st.SINGLE_MODEL_DIR,  self._model_list_names[i])):
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



  def save(self) -> None:
    """ Guarda """
    self._save_model()
    self._save_metrics_results()
    self._save_predictions_results()



  def plot(self) -> None:
    ''' Grafica los modelos.
      La gráfica representa los valores reales contra los valores predichos.
    '''
    for j in range(len(self._model_list_names)):
      # Grafica los resultados
      fig, ax = plt.subplots(figsize=(30, 30),
                              nrows=self._y_test.shape[1], ncols=1)

      # Crear una gráfico para cada salida
      for i in range(self._y_test.shape[1]):
        # Crear una gráfico de scatter
        # para cada salida del modelo con colores azul y rojo
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
      if self._type == 'single':
        if not os.path.exists(os.path.join(st.SINGLE_PLOTS_DIR,
                              self._model_list_names[j])):
          os.makedirs(os.path.join(st.SINGLE_PLOTS_DIR,
                      self._model_list_names[j]))
        plt.savefig(os.path.join(st.SINGLE_PLOTS_DIR,
                    self._model_list_names[j], f'{self._name}.png'))
      elif self._type == 'multiple':
        if not os.path.exists(os.path.join(st.MULTIPLE_PLOTS_DIR,
                              self._model_list_names[j])):
          os.makedirs(os.path.join(st.MULTIPLE_PLOTS_DIR,
                      self._model_list_names[j]))
        plt.savefig(os.path.join(st.MULTIPLE_PLOTS_DIR,
                    self._model_list_names[j], f'{self._name}.png'))
      elif self._type == 'global':
        if not os.path.exists(os.path.join(st.GLOBAL_PLOTS_DIR,
                              self._model_list_names[j])):
          os.makedirs(os.path.join(st.GLOBAL_PLOTS_DIR,
                      self._model_list_names[j]))
        plt.savefig(os.path.join(st.GLOBAL_PLOTS_DIR,
                    self._model_list_names[j], f'{self._name}.png'))

      # Cierra la gráfica
      plt.close()
