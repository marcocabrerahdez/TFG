''' Clase que automatiza el proceso de entrenamiento
    de modelos de Machine Learning.
'''

import importlib
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.metrics import (mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    '_trained_data_names',
    'index_list_y_test',
  ]

  def __init__(self, name: str, class_name, model, type_model: str, params,
              trained_data_names = [], columns_X: pd.DataFrame = pd.DataFrame(), columns_Y: pd.DataFrame = pd.DataFrame()) -> None:
    """Constructor for the class.

    Parameters:
      name (str): Name of the model.
      class_name (str): Name of the Machine Learning class.
      model (str): Machine Learning model.
      type_model (str): Type of Machine Learning model.
      params (hash): Parameters of the model.
      trained_data_names (List[str]): List of names of trained data.
      columns_X (pd.DataFrame): Input columns.
      columns_y (pd.DataFrame): Output columns.
    """
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
      self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._columns_X, self._columns_Y, test_size=st.TEST_SIZE, random_state=st.RANDOM_STATE)

    elif self._type == 'multiple' or self._type == 'global':
      self._X_train, self._X_test, self._y_train, self._y_test, self._columns_X, self._columns_Y = ut.get_splited_data(self._trained_data_names, self._type)
    else:
      raise Exception('El tipo de modelo no es válido.')

    # Guardar los datos de entrenamiento y prueba
    ut.save_splitted_data(self._X_train, self._X_test, self._y_train, self._y_test, self._columns_X, self._columns_Y, self._name, self._type)

    # Inicializar los atributos
    self._y_pred = None
    self.index_list_y_test = self._y_test.index.copy()
    # Preprocesar los datos
    scaler = StandardScaler()
    scaler.fit(self._X_train)
    self._X_train = scaler.transform(self._X_train)
    self._X_test = scaler.transform(self._X_test)

    # Convertir de nuevo a dataframe
    if self._type == 'single':
      self._X_train = pd.DataFrame(self._X_train, columns=self._columns_X.columns)
      self._X_test = pd.DataFrame(self._X_test, columns=self._columns_X.columns)
    elif self._type == 'multiple' or self._type == 'global':
      self._X_train = pd.DataFrame(self._X_train, columns=self._columns_X)
      self._X_test = pd.DataFrame(self._X_test, columns=self._columns_X)



  def train(self) -> None:
    ''' Entrena un conjunto de modelos '''
    print('Entrenando modelos...')
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



  def predict(self) -> None:
    ''' Predice con los modelos. '''
    print('Prediciendo con los modelos...')
    self._y_pred = [self._model[i].predict(self._X_test)
                    for i in range(len(self._model))]



  def metrics(self, nan_pos, increment=3) -> None:
    """
    Calcula el R2, donde, para cada comorbilidad,
    cruza tipo de modelo y técnica de ML.
    """
    print('Calculando métricas...')
    if self._type == 'single':
      nan_pos = nan_pos.loc[self.index_list_y_test]
      y_test = self._y_test.copy()
      y_pred = [pd.DataFrame(self._y_pred[i], columns=y_test.columns, index=y_test.index) for i in range(len(self._model))]

      # Eliminar las filas de y_test correspondientes a las posiciones con valores True en nan_pos
      y_test = self._y_test.drop(nan_pos.index[nan_pos.any(axis=1)])
      # Eliminar las filas de y_pred correspondientes a las posiciones con valores True en nan_pos
      y_pred = [y_pred[i].drop(nan_pos.index[nan_pos.any(axis=1)]) for i in range(len(self._model))]

      r2 = [] # R2
      mape = [] # MAPE
      for i in range(len(self._model)): # Para cada modelo
        r2.append(r2_score(y_test, y_pred[i], multioutput='uniform_average'))
        mape.append(mean_absolute_percentage_error(y_test, y_pred[i], multioutput='uniform_average'))

      single_r2_df = pd.DataFrame({
        'Modelo': self._model_list_names,
        self._type: r2
      })

      single_mape_df = pd.DataFrame({
        'Modelo': self._model_list_names,
        self._type: mape
      })

      column_name = self._y_test.columns[0]

      # Guarda el dataframe en un archivo excel
      if not os.path.exists(os.path.join(st.SINGLE_R2_TABLE_DIR)):
        os.mkdir(os.path.join(st.SINGLE_R2_TABLE_DIR,))
      single_r2_df.to_excel(os.path.join(st.SINGLE_R2_TABLE_DIR, f'{column_name}.xlsx'), index=False)

      if not os.path.exists(os.path.join(st.SINGLE_MAPE_TABLE_DIR)):
        os.mkdir(st.SINGLE_MAPE_TABLE_DIR)
      single_mape_df.to_excel(os.path.join(st.SINGLE_MAPE_TABLE_DIR, f'{column_name}.xlsx'), index=False)

      # Resetea el dataframe
      single_r2_df = pd.DataFrame()
      single_mape_df = pd.DataFrame()
      column_name = ''

    elif self._type == 'multiple' or self._type == 'global':
      nan_pos = nan_pos.loc[self.index_list_y_test]
      y_test = self._y_test.copy()
      y_pred = [pd.DataFrame(self._y_pred[i], columns=y_test.columns, index=y_test.index) for i in range(len(self._model))]

      # Eliminar las filas de y_test correspondientes a las posiciones con valores True en nan_pos creando un nuevo dataframe por columna
      r2_subset = []
      mape_subset = []
      r2 = []
      mape = []
      for j in range(len(self._model)):
        for i in range(0, y_test.shape[1], increment):
          # Eliminar las filas de y_test correspondientes a las posiciones con valores True en nan_pos
          y_test_increment = y_test.iloc[:, i:i+increment].drop(nan_pos.index[nan_pos.any(axis=1)])

          # Eliminar las filas de y_pred correspondientes a las posiciones con valores True en nan_pos
          y_pred_increment = y_pred[j].iloc[:, i:i+increment].drop(nan_pos.index[nan_pos.any(axis=1)])

          # Calcula el R2 y MAPE
          r2_subset.append(r2_score(y_test_increment, y_pred_increment, multioutput='uniform_average'))
          mape_subset.append(mean_absolute_percentage_error(y_test_increment, y_pred_increment, multioutput='uniform_average'))

          # Resetea el dataframe
          y_test_increment = pd.DataFrame()
          y_pred_increment = pd.DataFrame()

        # Guarda el valor de R2 y MAPE en una lista
        r2.append(r2_subset)
        mape.append(mape_subset)

        # Limpiar las listas
        r2_subset = []
        mape_subset = []

      colum_names = [[y_test.columns[i] for i in range(i, i+increment)][0] for i in range(0, y_test.shape[1], increment)]

      # Trasponer la matriz de R2 y MAPE
      r2 = np.transpose(r2)
      mape = np.transpose(mape)

      for i in range(len(r2)):
        multiple_global_r2_df = pd.DataFrame({
          'Modelo': self._model_list_names,
          self._type: r2[i]
        })

        multiple_global_mape_df = pd.DataFrame({
          'Modelo': self._model_list_names,
          self._type: mape[i]
        })

        if self._type == 'multiple':
          if not os.path.exists(os.path.join(st.MULTIPLE_R2_TABLE_DIR)):
            os.mkdir(st.MULTIPLE_R2_TABLE_DIR)
          multiple_global_r2_df.to_excel(os.path.join(st.MULTIPLE_R2_TABLE_DIR, f'{colum_names[i]}.xlsx'), index=False)

          if not os.path.exists(os.path.join(st.MULTIPLE_MAPE_TABLE_DIR)):
            os.mkdir(st.MULTIPLE_MAPE_TABLE_DIR)
          multiple_global_mape_df.to_excel(os.path.join(st.MULTIPLE_MAPE_TABLE_DIR, f'{colum_names[i]}.xlsx'), index=False)


        elif self._type == 'global':
          if not os.path.exists(os.path.join(st.GLOBAL_R2_TABLE_DIR)):
            os.mkdir(st.GLOBAL_R2_TABLE_DIR)
          multiple_global_r2_df.to_excel(os.path.join(st.GLOBAL_R2_TABLE_DIR, f'{colum_names[i]}.xlsx'), index=False)

          if not os.path.exists(os.path.join(st.GLOBAL_MAPE_TABLE_DIR)):
            os.mkdir(st.GLOBAL_MAPE_TABLE_DIR)
          multiple_global_mape_df.to_excel(os.path.join(st.GLOBAL_MAPE_TABLE_DIR, f'{colum_names[i]}.xlsx'), index=False)

        # Resetear el dataframe
        multiple_global_r2_df = pd.DataFrame()
        multiple_global_mape_df = pd.DataFrame()

      # Resetear el column_names
      colum_names = []



  def _save_predictions_results(self) -> None:
    '''Guarda las predicciones en un archivo excel.'''
    if self._type not in ['single', 'multiple', 'global']:
      raise ValueError('El tipo de modelo no es válido.')

    for i, y_pred_i in enumerate(self._y_pred):
      model_dir = ''
      if self._type == 'single':
        model_dir = st.SINGLE_PREDICTIONS_DIR
      elif self._type == 'multiple':
        model_dir = st.MULTIPLE_PREDICTIONS_DIR
      elif self._type == 'global':
        model_dir = st.GLOBAL_PREDICTIONS_DIR

      model_list_dir = os.path.join(model_dir, self._model_list_names[i])
      if not os.path.exists(model_list_dir):
        os.makedirs(model_list_dir)

      y_pred_df = pd.DataFrame(y_pred_i, columns=self._y_test.columns)
      file_path = os.path.join(model_list_dir, f'{self._name}.xlsx')
      y_pred_df.to_excel(file_path, index=False)



  def _save_model(self) -> None:
    '''Guarda los modelos.'''
    model_path = []
    model_dir = ''
    if self._type == 'single':
      model_dir = st.SINGLE_MODEL_DIR
    elif self._type == 'multiple':
      model_dir = st.MULTIPLE_MODEL_DIR
    elif self._type == 'global':
      model_dir = st.GLOBAL_MODEL_DIR

    for i, model in enumerate(self._model):
      model_name = f'{self._name}.pkl'
      model_list_dir = os.path.join(model_dir, self._model_list_names[i])
      model_path.append(os.path.join(model_list_dir, model_name))
      if not os.path.exists(model_list_dir):
        os.makedirs(model_list_dir)
      joblib.dump(model, model_path[i])



  def save(self) -> None:
    """ Guarda """
    print('Guardando resultados...')
    self._save_model()
    self._save_predictions_results()



  def plot_avg_time(self, nan_pos) -> None:
    ''' Grafica los modelos.
      La gráfica representa los valores reales contra los valores predichos.
    '''
    print('Graficando...')
    nan_pos = nan_pos.loc[self.index_list_y_test]
    y_test = self._y_test.copy()
    y_pred = [pd.DataFrame(self._y_pred[i], columns=y_test.columns, index=y_test.index) for i in range(len(self._model))]

    if self._type == 'single':
      # Eliminar las filas de y_test correspondientes a las posiciones con valores True en nan_pos
      y_test = self._y_test.drop(nan_pos.index[nan_pos.any(axis=1)])
      # Eliminar las filas de y_pred correspondientes a las posiciones con valores True en nan_pos
      y_pred = [y_pred[i].drop(nan_pos.index[nan_pos.any(axis=1)]) for i in range(len(self._model))]

      for j in range(len(self._model_list_names)):
        # Filtra las columnas que empiezan por AVG y las asigna a un dataframe
        y_test_l95ci_df = y_test.filter(regex='^(?!(AVG|U95CI)).*')
        y_test_u95ci_df = y_test.filter(regex='^(?!(AVG|L95CI)).*')
        y_test_df = y_test.filter(regex='^(?!(L95CI|U95CI)).*')

        y_pred_l95ci_df = y_pred[j].filter(regex='^(?!(AVG|U95CI)).*')
        y_pred_u95ci_df = y_pred[j].filter(regex='^(?!(AVG|L95CI)).*')
        y_pred_df = y_pred[j].filter(regex='^(?!(L95CI|U95CI)).*')

        # Graficar los resultados
        fig, ax = plt.subplots(figsize=(10, 10))

        # Normalizar los ejes x e y
        ax.set_xlim([0, max(y_test_df.values.max(), y_pred_df.values.max(), y_test_l95ci_df.values.max(), y_pred_l95ci_df.values.max(), y_test_u95ci_df.values.max(), y_pred_u95ci_df.values.max()) + 1])
        ax.set_ylim([0, max(y_test_df.values.max(), y_pred_df.values.max(), y_test_l95ci_df.values.max(), y_pred_l95ci_df.values.max(), y_test_u95ci_df.values.max(), y_pred_u95ci_df.values.max()) + 1])

        # Grafica la diagonal
        ax.plot([0, max(y_test_df.values.max(), y_pred_df.values.max(), y_test_l95ci_df.values.max(), y_pred_l95ci_df.values.max(), y_test_u95ci_df.values.max(), y_pred_u95ci_df.values.max()) + 1], [0, max(y_test_df.values.max(), y_pred_df.values.max(), y_test_l95ci_df.values.max(), y_pred_l95ci_df.values.max(), y_test_u95ci_df.values.max(), y_pred_u95ci_df.values.max()) + 1], color='black', linestyle='-', label='Valor Ideal')

        # Crear la malla convexa
        avg_points = np.column_stack((y_test_df, y_pred_df))
        avg_hull = ConvexHull(avg_points)

        # Grafica los valores reales vs predichos como puntos de AVG
        ax.fill(avg_points[avg_hull.vertices,0], avg_points[avg_hull.vertices,1], 'r', alpha=0.15, label='Área de valores predichos de tiempo promedio')
        ax.scatter(y_test_df, y_pred_df, color='red', alpha=0.85, label='Valor predicho de tiempo promedio')

        # Grafica los valores reales vs predichos como puntos de L95CI
        l95ci_points = np.column_stack((y_test_l95ci_df, y_pred_l95ci_df))
        l95ci_hull = ConvexHull(l95ci_points)

        ax.fill(l95ci_points[l95ci_hull.vertices,0], l95ci_points[l95ci_hull.vertices,1], color='blue', alpha=0.15, label='Área de valores predichos del intervalo de confianza inferior')
        ax.scatter(y_test_l95ci_df, y_pred_l95ci_df, color='blue', marker='x', alpha=0.85, label='Valor predicho del intervalo de confianza inferior')

        # Grafica los valores reales vs predichos como puntos de U95CI
        u95ci_points = np.column_stack((y_test_u95ci_df, y_pred_u95ci_df))
        u95ci_hull = ConvexHull(u95ci_points)

        ax.fill(u95ci_points[u95ci_hull.vertices,0], u95ci_points[u95ci_hull.vertices,1], color='green', alpha=0.15, label='Área de valores predichos de intervalo de confianza superior')
        ax.scatter(y_test_u95ci_df, y_pred_u95ci_df, color='green', marker='^', alpha=0.85, label='Valor predicho del intervalo de confianza superior')

        # Agrega los ejes
        ax.set_xlabel('Valor real de tiempo promedio', fontsize=10, fontweight='bold')
        ax.set_ylabel('Valor predicho de tiempo promedio', fontsize=10, fontweight='bold')

        # Agrega la leyenda
        ax.legend()

        ax.set_title(f'Tiempo promedio hasta aparición de {self._name}', fontweight='bold', fontsize=11)

        # Configura el layout
        fig.set_layout_engine('compressed')

        # Guarda la gráfica
        if not os.path.exists(os.path.join(st.SINGLE_PLOTS_DIR,
                              self._model_list_names[j], 'average time')):
          os.makedirs(os.path.join(st.SINGLE_PLOTS_DIR,
                      self._model_list_names[j], 'average time'))
        plt.savefig(os.path.join(st.SINGLE_PLOTS_DIR,
                    self._model_list_names[j], 'average time', f'{self._name}.png'))

        # Cierra la gráfica
        plt.close()

        # Borrar las variables
        avg_points = None
        avg_hull = None
        l95ci_points = None
        l95ci_hull = None
        u95ci_points = None
        u95ci_hull = None

    elif self._type == 'multiple' or self._type == 'global':
      for j in range(len(self._model_list_names)):
        for i in range(0, len(y_test.columns), 3):
          # Eliminar las filas de y_test correspondientes a las posiciones con valores True en nan_pos
          y_test_col = y_test.iloc[:, i:i+3].drop(nan_pos.index[nan_pos.any(axis=1)])

          # Eliminar las filas de y_pred correspondientes a las posiciones con valores True en nan_pos
          y_pred_col = y_pred[j].iloc[:, i:i+3].drop(nan_pos.index[nan_pos.any(axis=1)])

          # Filtra las columnas que empiezan por AVG y las asigna a un dataframe
          y_test_l95ci_df = y_test_col.filter(regex='^(?!(AVG|U95CI)).*')
          y_test_u95ci_df = y_test_col.filter(regex='^(?!(AVG|L95CI)).*')
          y_test_df = y_test_col.filter(regex='^(?!(L95CI|U95CI)).*')

          y_pred_l95ci_df = y_pred_col.filter(regex='^(?!(AVG|U95CI)).*')
          y_pred_u95ci_df = y_pred_col.filter(regex='^(?!(AVG|L95CI)).*')
          y_pred_df = y_pred_col.filter(regex='^(?!(L95CI|U95CI)).*')

          # Graficar los resultados
          fig, ax = plt.subplots(figsize=(10, 10))

          # Normalizar los ejes x e y
          ax.set_xlim([0, max(y_test_df.values.max(), y_pred_df.values.max(), y_test_l95ci_df.values.max(), y_pred_l95ci_df.values.max(), y_test_u95ci_df.values.max(), y_pred_u95ci_df.values.max()) + 1])
          ax.set_ylim([0, max(y_test_df.values.max(), y_pred_df.values.max(), y_test_l95ci_df.values.max(), y_pred_l95ci_df.values.max(), y_test_u95ci_df.values.max(), y_pred_u95ci_df.values.max()) + 1])

          # Grafica la diagonal
          ax.plot([0, max(y_test_df.values.max(), y_pred_df.values.max(), y_test_l95ci_df.values.max(), y_pred_l95ci_df.values.max(), y_test_u95ci_df.values.max(), y_pred_u95ci_df.values.max()) + 1], [0, max(y_test_df.values.max(), y_pred_df.values.max(), y_test_l95ci_df.values.max(), y_pred_l95ci_df.values.max(), y_test_u95ci_df.values.max(), y_pred_u95ci_df.values.max()) + 1], color='black', linestyle='-', label='Valor Ideal')

          # Crear la malla convexa
          avg_points = np.column_stack((y_test_df, y_pred_df))
          avg_hull = ConvexHull(avg_points)

          # Grafica los valores reales vs predichos como puntos de AVG
          ax.fill(avg_points[avg_hull.vertices,0], avg_points[avg_hull.vertices,1], 'r', alpha=0.15, label='Área de valores predichos de incidencia promedio')
          ax.scatter(y_test_df, y_pred_df, color='red', alpha=0.85, label='Valor predicho de incidencia promedio')

          # Grafica los valores reales vs predichos como puntos de L95CI
          l95ci_points = np.column_stack((y_test_l95ci_df, y_pred_l95ci_df))
          l95ci_hull = ConvexHull(l95ci_points)

          ax.fill(l95ci_points[l95ci_hull.vertices,0], l95ci_points[l95ci_hull.vertices,1], color='blue', alpha=0.15, label='Área de valores predichos del intervalo de confianza inferior')
          ax.scatter(y_test_l95ci_df, y_pred_l95ci_df, color='blue', marker='x', alpha=0.85, label='Valor predicho del intervalo de confianza inferior')

          # Grafica los valores reales vs predichos como puntos de U95CI
          u95ci_points = np.column_stack((y_test_u95ci_df, y_pred_u95ci_df))
          u95ci_hull = ConvexHull(u95ci_points)

          ax.fill(u95ci_points[u95ci_hull.vertices,0], u95ci_points[u95ci_hull.vertices,1], color='green', alpha=0.15, label='Área de valores predichos de intervalo de confianza superior')
          ax.scatter(y_test_u95ci_df, y_pred_u95ci_df, color='green', marker='^', alpha=0.85, label='Valor predicho del intervalo de confianza superior')

          # Agrega los ejes
          ax.set_xlabel('Valor real de incidencia promedio', fontsize=10, fontweight='bold')
          ax.set_ylabel('Valor predicho de incidencia promedio', fontsize=10, fontweight='bold')

          # Agrega la leyenda
          ax.legend()

          name = ''
          if 'AVG_TIMETO_HF' == y_test_df.columns[0]:
            name = 'Fallo Cardiaco'

          elif 'AVG_TIMETO_MI' == y_test_df.columns[0]:
            name = 'Infarto de Miocardio'

          elif 'AVG_TIMETO_ANGINA' == y_test_df.columns[0]:
            name = 'Angina'

          elif 'AVG_TIMETO_STROKE' == y_test_df.columns[0]:
            name = 'Ictus'

          elif 'AVG_TIMETO_BLI' == y_test_df.columns[0]:
            name = 'Ceguera'

          elif 'AVG_TIMETO_ME' == y_test_df.columns[0]:
            name = 'Edema macular diabético'

          elif 'AVG_TIMETO_BGRET' == y_test_df.columns[0]:
            name = 'Retinopatía de fondo'

          elif 'AVG_TIMETO_PRET' == y_test_df.columns[0]:
            name = 'Retinopatía proliferativa'

          elif 'AVG_TIMETO_NEU' == y_test_df.columns[0]:
            name = 'Neuropatía individual'

          elif 'AVG_TIMETO_LEA' == y_test_df.columns[0]:
            name = 'Amputación extremidades inferiores'

          elif 'AVG_TIMETO_ALB1' == y_test_df.columns[0]:
            name = 'Microalbuminuria'

          elif 'AVG_TIMETO_ALB2' == y_test_df.columns[0]:
            name = 'Macroalbuminuria'

          elif 'AVG_TIMETO_ESRD' == y_test_df.columns[0]:
            name = 'Enfermedad renal terminal'

          else:
            name = y_test_df.columns[0]

          # Agrega el título
          ax.set_title(f'Incidencia promedio de {name} en relación con la población total', fontweight='bold', fontsize=12)

          # Configura el layout
          fig.set_layout_engine('compressed')

          if self._type == 'multiple':
            if not os.path.exists(os.path.join(st.MULTIPLE_PLOTS_DIR, self._model_list_names[j], 'incidence')):
              os.makedirs(os.path.join(st.MULTIPLE_PLOTS_DIR, self._model_list_names[j], 'incidence'))
            plt.savefig(os.path.join(st.MULTIPLE_PLOTS_DIR, self._model_list_names[j], 'incidence', f'{name}.png'))
          elif self._type == 'global':
            if not os.path.exists(os.path.join(st.GLOBAL_PLOTS_DIR, self._model_list_names[j], 'incidence')):
              os.makedirs(os.path.join(st.GLOBAL_PLOTS_DIR, self._model_list_names[j], 'incidence'))
            plt.savefig(os.path.join(st.GLOBAL_PLOTS_DIR, self._model_list_names[j], 'incidence', f'{name}.png'))

          # Cierra la gráfica
          plt.close()

          # Borrar las variables
          avg_points = None
          avg_hull = None
          l95ci_points = None
          l95ci_hull = None
          u95ci_points = None
          u95ci_hull = None
          y_test_df = None
          y_pred_df = None
          y_test_l95ci_df = None
          y_pred_l95ci_df = None
          y_test_u95ci_df = None
          y_pred_u95ci_df = None
          y_test_col = None
          y_pred_col = None
          name = ''



  def plot_upto_time(self) -> None:
    '''
      Gráfica los modelos de la supervivenia hasta cierta edad.
    '''
    print('Graficando...')
    y_test = self._y_test.copy()
    y_pred = [pd.DataFrame(self._y_pred[i], columns=y_test.columns, index=y_test.index) for i in range(len(self._model))]

    if self._type == 'single':
      # Para cada modelo
      for j in range(len(self._model_list_names)):
        # Filtra las columnas que empiezan por AVG y las asigna a un dataframe
        y_test_l95ci_df = y_test.filter(regex='^(?!(AVG|U95CI)).*')
        y_test_u95ci_df = y_test.filter(regex='^(?!(AVG|L95CI)).*')
        y_test_df = y_test.filter(regex='^(?!(L95CI|U95CI)).*')

        y_pred_l95ci_df = y_pred[j].filter(regex='^(?!(AVG|U95CI)).*')
        y_pred_u95ci_df = y_pred[j].filter(regex='^(?!(AVG|L95CI)).*')
        y_pred_df = y_pred[j].filter(regex='^(?!(L95CI|U95CI)).*')

        # Graficar los resultados
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(y_test_df.columns, y_test_df.cumsum(axis=1).iloc[1], 'o-b', label='Valor ideal')
        ax.plot(y_pred_df.columns, y_pred_df.cumsum(axis=1).iloc[1], 'o-r', label='Valor predicho')

        # Grafica los intervalos de confianza
        ax.fill_between(y_test_df.columns,
                            y_test_l95ci_df.cumsum(axis=1).values[1],
                            y_test_u95ci_df.cumsum(axis=1).values[1],
                            alpha=0.2, color='b', label='Área de confianza real')
        ax.fill_between(y_pred_df.columns,
                            y_pred_l95ci_df.cumsum(axis=1).values[1],
                            y_pred_u95ci_df.cumsum(axis=1).values[1],
                            alpha=0.2, color='r', label='Área de confianza predicha')

        # Agregar xticks
        ax.set_xticks(y_test_df.columns)
        ax.set_xticklabels([i.split("UPTO_")[-1] for i in y_test_df.columns])

        # Agrega los ejes
        ax.set_xlabel('Edad')
        ax.set_ylabel('Porcentaje de afectación')

        name = ''
        if 'AVG_HF_UPTO_20' == y_test_df.columns[0]:
          name = 'Fallo Cardiaco'

        elif 'AVG_MI_UPTO_20' == y_test_df.columns[0]:
          name = 'Infarto de Miocardio'

        elif 'AVG_ANGINA_UPTO_20' == y_test_df.columns[0]:
          name = 'Angina'

        elif 'AVG_STROKE_UPTO_20' == y_test_df.columns[0]:
          name = 'Ictus'

        elif 'AVG_BLI_UPTO_20' == y_test_df.columns[0]:
          name = 'Ceguera'

        elif 'AVG_ME_UPTO_20' == y_test_df.columns[0]:
          name = 'Edema macular diabético'

        elif 'AVG_BGRET_UPTO_20' == y_test_df.columns[0]:
          name = 'Retinopatía de fondo'

        elif 'AVG_PRET_UPTO_20' == y_test_df.columns[0]:
          name = 'Retinopatía proliferativa'

        elif 'AVG_NEU_UPTO_20' == y_test_df.columns[0]:
          name = 'Neuropatía individual'

        elif 'AVG_LEA_UPTO_20' == y_test_df.columns[0]:
          name = 'Amputación extremidades inferiores'

        elif 'AVG_ALB1_UPTO_20' == y_test_df.columns[0]:
          name = 'Microalbuminuria'

        elif 'AVG_ALB2_UPTO_20' == y_test_df.columns[0]:
          name = 'Macroalbuminuria'

        elif 'AVG_ESRD_UPTO_20' == y_test_df.columns[0]:
          name = 'Enfermedad renal terminal'

        else:
          name = y_test_df.columns[0]

        ax.legend()

        # Agrega el título
        fig.suptitle(f'Relación de afectación de {name} por grupos de edad', fontweight='bold', fontsize=12)

        # Configura el layout
        fig.set_layout_engine('compressed')

        if not os.path.exists(os.path.join(st.SINGLE_PLOTS_DIR, self._model_list_names[j], 'upto time')):
          os.makedirs(os.path.join(st.SINGLE_PLOTS_DIR, self._model_list_names[j], 'upto time'))
        plt.savefig(os.path.join(st.SINGLE_PLOTS_DIR, self._model_list_names[j], 'upto time', f'{name}.png'))

        # Cierra la gráfica
        plt.close()

    elif self._type == 'multiple' or self._type == 'global':
      for j in range(len(self._model)):
        for i in range(0, len(y_test.columns), 27):
          # Eliminar las filas de y_test correspondientes a las posiciones con valores True en nan_pos
          y_test_col = y_test.iloc[:, i:i+27]

          # Eliminar las filas de y_pred correspondientes a las posiciones con valores True en nan_pos
          y_pred_col = y_pred[j].iloc[:, i:i+27]

          # Filtra las columnas que empiezan por AVG y las asigna a un dataframe
          y_test_l95ci_df = y_test_col.filter(regex='^(?!(AVG|U95CI)).*')
          y_test_u95ci_df = y_test_col.filter(regex='^(?!(AVG|L95CI)).*')
          y_test_df = y_test_col.filter(regex='^(?!(L95CI|U95CI)).*')
          y_pred_l95ci_df = y_pred_col.filter(regex='^(?!(AVG|U95CI)).*')
          y_pred_u95ci_df = y_pred_col.filter(regex='^(?!(AVG|L95CI)).*')
          y_pred_df = y_pred_col.filter(regex='^(?!(L95CI|U95CI)).*')

          # Graficar los resultados
          fig, ax = plt.subplots(figsize=(10, 6))

          # Grafica los datos de suma acumulada
          ax.plot(y_test_df.columns, y_test_df.cumsum(axis=1).iloc[1], 'o-b', label='Valor real')
          ax.plot(y_pred_df.columns, y_pred_df.cumsum(axis=1).iloc[1], 'o-r', label='Valor predicho')

          # Grafica los intervalos de confianza
          ax.fill_between(y_test_df.columns,
                              y_test_l95ci_df.cumsum(axis=1).values[1],
                              y_test_u95ci_df.cumsum(axis=1).values[1],
                              alpha=0.2, color='b', label='Área de confianza real')

          ax.fill_between(y_pred_df.columns,
                              y_pred_l95ci_df.cumsum(axis=1).values[1],
                              y_pred_u95ci_df.cumsum(axis=1).values[1],
                              alpha=0.2, color='r', label='Área de confianza predicha')

          # Agregar xticks
          ax.set_xticks(y_test_df.columns)
          ax.set_xticklabels([k.split("UPTO_")[-1] for k in y_test_df.columns])

          # Agrega los ejes
          ax.set_xlabel('Edad')
          ax.set_ylabel('Porcentaje de afectación')

          name = ''
          if 'AVG_HF_UPTO_20' == y_test_df.columns[0]:
            name = 'Fallo Cardiaco'

          elif 'AVG_MI_UPTO_20' == y_test_df.columns[0]:
            name = 'Infarto de Miocardio'

          elif 'AVG_ANGINA_UPTO_20' == y_test_df.columns[0]:
            name = 'Angina'

          elif 'AVG_STROKE_UPTO_20' == y_test_df.columns[0]:
            name = 'Ictus'

          elif 'AVG_BLI_UPTO_20' == y_test_df.columns[0]:
            name = 'Ceguera'

          elif 'AVG_ME_UPTO_20' == y_test_df.columns[0]:
            name = 'Edema macular diabético'

          elif 'AVG_BGRET_UPTO_20' == y_test_df.columns[0]:
            name = 'Retinopatía de fondo'

          elif 'AVG_PRET_UPTO_20' == y_test_df.columns[0]:
            name = 'Retinopatía proliferativa'

          elif 'AVG_NEU_UPTO_20' == y_test_df.columns[0]:
            name = 'Neuropatía individual'

          elif 'AVG_LEA_UPTO_20' == y_test_df.columns[0]:
            name = 'Amputación extremidades inferiores'

          elif 'AVG_ALB1_UPTO_20' == y_test_df.columns[0]:
            name = 'Microalbuminuria'

          elif 'AVG_ALB2_UPTO_20' == y_test_df.columns[0]:
            name = 'Macroalbuminuria'

          elif 'AVG_ESRD_UPTO_20' == y_test_df.columns[0]:
            name = 'Enfermedad renal terminal'

          else:
            name = y_test_df.columns[0]

          ax.legend()

          # Agrega el título
          fig.suptitle(f'Relación de afectación de {name} por grupos de edad', fontweight='bold', fontsize=12)

          # Configura el layout
          fig.set_layout_engine('compressed')

          if self._type == 'multiple':
            if not os.path.exists(os.path.join(st.MULTIPLE_PLOTS_DIR, self._model_list_names[j], 'upto time')):
              os.makedirs(os.path.join(st.MULTIPLE_PLOTS_DIR, self._model_list_names[j], 'upto time'))
            plt.savefig(os.path.join(st.MULTIPLE_PLOTS_DIR, self._model_list_names[j], 'upto time', f'{name}.png'))

          elif self._type == 'global':
            if not os.path.exists(os.path.join(st.GLOBAL_PLOTS_DIR, self._model_list_names[j], 'upto time')):
              os.makedirs(os.path.join(st.GLOBAL_PLOTS_DIR, self._model_list_names[j], 'upto time'))
            plt.savefig(os.path.join(st.GLOBAL_PLOTS_DIR, self._model_list_names[j], 'upto time', f'{name}.png'))

          # Cierra la gráfica
          plt.close()