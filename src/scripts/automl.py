''' Clase que automatiza el proceso de entrenamiento
    de modelos de Machine Learning.
'''

import os
import importlib
import time
import joblib
import psutil
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        # Guardar las predicciones con sus datos de entrada
        pd.concat([self._X_test, pd.DataFrame(self._y_pred[i])], axis=1).to_excel(os.path.join(st.SINGLE_PREDICTIONS_DIR, self._model_list_names[i], f'{self._name}.xlsx'), index=False)

      elif self._type == 'multiple':
        if not os.path.exists(os.path.join(st.MULTIPLE_PREDICTIONS_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.MULTIPLE_PREDICTIONS_DIR, self._model_list_names[i]))
        # Guardar las predicciones con sus datos de entrada
        pd.concat([self._X_test, pd.DataFrame(self._y_pred[i])], axis=1).to_excel(os.path.join(st.MULTIPLE_PREDICTIONS_DIR, self._model_list_names[i], f'{self._name}.xlsx'), index=False)

      elif self._type == 'global':
        if not os.path.exists(os.path.join(st.GLOBAL_PREDICTIONS_DIR, self._model_list_names[i])):
          os.makedirs(os.path.join(st.GLOBAL_PREDICTIONS_DIR, self._model_list_names[i]))
         # Guardar las predicciones con sus datos de entrada
        pd.concat([self._X_test, pd.DataFrame(self._y_pred[i])], axis=1).to_excel(os.path.join(st.GLOBAL_PREDICTIONS_DIR, self._model_list_names[i], f'{self._name}.xlsx'), index=False)

      else:
        raise Exception('El tipo de modelo no es válido.')



  def _save_metrics_results(self) -> None:
    # Guarda los resultados en un archivo excel
    for i in range(len(self._model)):
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



  def plot_avg_time(self) -> None:
    ''' Grafica los modelos.
      La gráfica representa los valores reales contra los valores predichos.
    '''
    # Para cada modelo
    for j in range(len(self._model_list_names)):
      # Convertir los valores predichos  a un dataframe
      y_pred_df = pd.DataFrame(self._y_pred[j], columns=self._y_test.columns)
      y_pred_df.reset_index(drop=True, inplace=True)

      # Unir X_test con y_test
      Xy_test_df = pd.concat([self._X_test, self._y_test], axis=1)
      Xy_test_df.reset_index(drop=True, inplace=True)
      Xy_test_df = pd.concat([Xy_test_df, y_pred_df], axis=1)
      Xy_test_df.reset_index(drop=True, inplace=True)

      # Filtra las columnas que empiezan por AVG y las asigna a un dataframe
      Xy_test_l95ci_df = Xy_test_df.filter(regex='^(?!(AVG|U95CI)).*')
      Xy_test_u95ci_df = Xy_test_df.filter(regex='^(?!(AVG|L95CI)).*')
      Xy_test_df = Xy_test_df.filter(regex='^(?!(L95CI|U95CI)).*')

      if self._type == 'single':
        # Graficar los resultados
        fig, ax = plt.subplots(figsize=(10, 10))

        # Grafica los valores reales vs predichos como puntos
        ax.plot(Xy_test_df.iloc[:, 3], Xy_test_df.iloc[:, 3], color=(0.3, 0.8, 0.5), label='Valor ideal')
        ax.scatter(Xy_test_df.iloc[:, 3], Xy_test_df.iloc[:, 4], color=(0.8, 0.7, 0.2), label='Valor predicho')

        # Grafica los intervalos de confianza reales y predichos como líneas
        ax.plot(Xy_test_df.iloc[:, 3], Xy_test_l95ci_df.iloc[:, 3], color=(0.2, 0.5, 0.5), linestyle='dashed', label='Intervalo de confianza ideal')
        ax.plot(Xy_test_df.iloc[:, 3], Xy_test_u95ci_df.iloc[:, 3], color=(0.2, 0.5, 0.5), linestyle='dashed')
        ax.scatter(Xy_test_df.iloc[:, 3], Xy_test_u95ci_df.iloc[:, 4], color=(0.7, 0.8, 0.2), label='Intervalo de confianza predicho')
        ax.scatter(Xy_test_df.iloc[:, 3], Xy_test_l95ci_df.iloc[:, 4], color=(0.7, 0.8, 0.2))

        # Agrega los ejes
        ax.set_xlabel('Valor ideal de tiempo promedio hasta aparición', fontsize=10, fontweight='bold')
        ax.set_ylabel('Valor predicho de tiempo promedio hasta aparición', fontsize=10, fontweight='bold')

        # Agrega la leyenda
        ax.legend()

        # Agrega el título
        ax.set_title(f'Gráfica de tiempo promedio hasta aparición para {self._name}', fontweight='bold', fontsize=12)

        # Configura el layout
        fig.set_layout_engine('compressed')

      elif self._type == 'multiple' or self._type == 'global':
        # Quitar las 3 primeras columnas
        Xy_test_df = Xy_test_df.drop(Xy_test_df.columns[:3], axis=1)
        Xy_test_l95ci_df = Xy_test_l95ci_df.drop(Xy_test_l95ci_df.columns[:3], axis=1)
        Xy_test_u95ci_df = Xy_test_u95ci_df.drop(Xy_test_u95ci_df.columns[:3], axis=1)

        # Dividir el dataframe en 2 dataframes
        y_test_df = Xy_test_df.iloc[:, :int(Xy_test_df.shape[1] / 2)]
        y_pred_df = Xy_test_df.iloc[:, int(Xy_test_df.shape[1] / 2):]
        y_test_l95ci_df = Xy_test_l95ci_df.iloc[:, :int(Xy_test_l95ci_df.shape[1] / 2)]
        y_test_u95ci_df = Xy_test_u95ci_df.iloc[:, :int(Xy_test_u95ci_df.shape[1] / 2)]
        y_pred_l95ci_df = Xy_test_l95ci_df.iloc[:, int(Xy_test_l95ci_df.shape[1] / 2):]
        y_pred_u95ci_df = Xy_test_u95ci_df.iloc[:, int(Xy_test_u95ci_df.shape[1] / 2):]

        # Crear un subplot para cada columna
        fig, ax = plt.subplots(figsize=(20, 20) if self._type == 'global' else (10, 10),
                                nrows=(math.ceil(y_test_df.shape[1] / 2)) if (math.ceil(y_test_df.shape[1] / 2)) >= 2  else y_test_df.shape[1],
                                ncols=2 if (math.ceil(y_test_df.shape[1] / 2)) >= 2 else 1)

        # Aplanar el arreglo
        ax = ax.ravel()

        # Graficar cada columna
        for i in range(y_test_df.shape[1]):
          # Grafica los valores reales vs predichos como puntos
          ax[i].plot(y_test_df.iloc[:, i], y_test_df.iloc[:, i], color=(0.3, 0.8, 0.5), label='Valor ideal')
          ax[i].scatter(y_test_df.iloc[:, i], y_pred_df.iloc[:, i], color=(0.8, 0.7, 0.2), label='Valor predicho')

          # Grafica los valores reales vs predichos como líneas
          ax[i].plot(y_test_df.iloc[:, i], y_test_l95ci_df.iloc[:, i], color=(0.2, 0.5, 0.5), linestyle='dashed', label='Intervalo de confianza ideal')
          ax[i].scatter(y_test_df.iloc[:, i], y_pred_l95ci_df.iloc[:, i], color=(0.7, 0.8, 0.2), label='Intervalo de confianza predicho')
          ax[i].plot(y_test_df.iloc[:, i], y_test_u95ci_df.iloc[:, i], color=(0.2, 0.5, 0.5), linestyle='dashed')
          ax[i].scatter(y_test_df.iloc[:, i], y_pred_u95ci_df.iloc[:, i], color=(0.7, 0.8, 0.2))

          # Agrega los ejes
          ax[i].set_xlabel('Valor ideal', fontsize=10, fontweight='bold')
          ax[i].set_ylabel('Valor predicho', fontsize=10, fontweight='bold')

          # añadir titulo al subplot
          ax[i].set_title(f'{y_test_df.columns[i]}', fontweight='bold', fontsize=10)

        # Agrega la leyenda
        fig.legend(['Valor ideal', 'Valor predicho', 'Intervalo de confianza ideal', 'Intervalo de confianza predicho'], fontsize=10)

        # Agrega el título
        fig.suptitle(f'Gráfica de tiempo promedio hasta aparición para {self._name}', fontweight='bold', fontsize=15)

        # Configura el layout
        fig.set_layout_engine('compressed')

      # Limpiar la memoria
      Xy_test_df = pd.DataFrame()
      Xy_test_l95ci_df = pd.DataFrame()
      Xy_test_u95ci_df = pd.DataFrame()
      y_pred_df = pd.DataFrame()

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


  def plot_upto_time(self) -> None:
    '''
      Gráfica los modelos de la supervivenia hasta cierta edad.
    '''
    # Para cada modelo
    for j in range(len(self._model_list_names)):
      # Convertir los valores predichos  a un dataframe
      y_pred_df = pd.DataFrame(self._y_pred[j], columns=self._y_test.columns)

      # Filtra las columnas que empiezan por AVG y las asigna a un dataframe
      y_pred_df_avg = y_pred_df.filter(regex='^AVG')
      y_test_df_avg = self._y_test.filter(regex='^AVG')

      # Filtra las columnas que empiezan por L95CI y las asigna a un dataframe
      y_pred_df_l95ci = y_pred_df.filter(regex='^L95CI')
      y_test_df_l95ci = self._y_test.filter(regex='^L95CI')

      # Filtra las columnas que empiezan por U95CI y las asigna a un dataframe
      y_pred_df_u95ci = y_pred_df.filter(regex='^U95CI')
      y_test_df_u95ci = self._y_test.filter(regex='^U95CI')

      # Unir en un dataframe X_test con y_test
      Xy_test_df = pd.concat([self._X_test, y_test_df_avg], axis=1)

      # Resetea los indices
      y_pred_df_avg.reset_index(drop=True, inplace=True)
      y_test_df_avg.reset_index(drop=True, inplace=True)
      y_pred_df_l95ci.reset_index(drop=True, inplace=True)
      y_test_df_l95ci.reset_index(drop=True, inplace=True)
      y_pred_df_u95ci.reset_index(drop=True, inplace=True)
      y_test_df_u95ci.reset_index(drop=True, inplace=True)

      if self._type == 'single':
        # Graficar los resultados
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), subplot_kw={'projection': '3d'})

        # Renombrar las columnas
        cols = y_test_df_avg.columns
        new_cols = [int(col.split("_UPTO_")[1]) for col in cols] # Extrae el número de la columna
        y_test_df_avg.columns = new_cols
        y_pred_df_avg.columns = new_cols
        y_test_df_l95ci.columns = new_cols
        y_pred_df_l95ci.columns = new_cols
        y_test_df_u95ci.columns = new_cols
        y_pred_df_u95ci.columns = new_cols

        # Caluclar numero de columnas
        for col in y_test_df_avg.columns:
          # Graficar los valores
          ax[0].scatter(Xy_test_df['HBA1C'], y_test_df_avg[col], y_test_df_l95ci[col].values, color=(0.3, 0.8, 0.5))
          ax[0].scatter(Xy_test_df['HBA1C'], y_test_df_avg[col], y_pred_df_l95ci[col].values, color=(0.8, 0.7, 0.2))

        for col in y_test_df_avg.columns:
          # Graficar los valores
          ax[1].scatter(Xy_test_df['HBA1C'], y_test_df_avg[col], y_test_df_avg[col].values, color=(0.3, 0.8, 0.5), label='Valor generado (modelo simulación)')
          ax[1].scatter(Xy_test_df['HBA1C'], y_test_df_avg[col], y_pred_df_avg[col].values, color=(0.8, 0.7, 0.2), label='Valor predicho (modelo subrogado)')

        for col in y_test_df_avg.columns:
          # Graficar los valores
          ax[2].scatter(Xy_test_df['HBA1C'], y_test_df_avg[col], y_test_df_u95ci[col].values, color=(0.3, 0.8, 0.5))
          ax[2].scatter(Xy_test_df['HBA1C'], y_test_df_avg[col], y_pred_df_u95ci[col].values, color=(0.8, 0.7, 0.2))

        # Set yticks to the columns of the dataframe
        for i in range(3):
          ax[i].set_yticks(y_test_df_avg.columns)

          # Configurar la gráfica
          ax[i].set_xlabel('Nivel de HBA1C', fontsize=8, fontweight='bold')
          ax[i].set_ylabel('Edad (hasta)', fontsize=8, fontweight='bold')
          ax[i].set_zlabel('Porcentaje de personas afectadas', fontsize=8, fontweight='bold')

        # Añadir titulo a cada gráfica
        ax[0].set_title('Intervalo de confianza inferior', fontweight='bold', fontsize=10)
        ax[1].set_title('Valor medio', fontweight='bold', fontsize=10)
        ax[2].set_title('Intervalo de confianza superior', fontweight='bold', fontsize=10)


      else:
        # Dividir el dataframe en grupos
        y_pred_df_avg = [y_pred_df_avg[y_pred_df_avg.columns[i:i+9]] for i in range(0, len(y_pred_df_avg.columns), 9)]
        y_test_df_avg = [y_test_df_avg[y_test_df_avg.columns[i:i+9]] for i in range(0, len(y_test_df_avg.columns), 9)]

        y_pred_df_l95ci = [y_pred_df_l95ci[y_pred_df_l95ci.columns[i:i+9]] for i in range(0, len(y_pred_df_l95ci.columns), 9)]
        y_test_df_l95ci = [y_test_df_l95ci[y_test_df_l95ci.columns[i:i+9]] for i in range(0, len(y_test_df_l95ci.columns), 9)]

        y_pred_df_u95ci = [y_pred_df_u95ci[y_pred_df_u95ci.columns[i:i+9]] for i in range(0, len(y_pred_df_u95ci.columns), 9)]
        y_test_df_u95ci = [y_test_df_u95ci[y_test_df_u95ci.columns[i:i+9]] for i in range(0, len(y_test_df_u95ci.columns), 9)]

        # Eliminar columnas duplicadas en los grupos
        for i in range(len(y_pred_df_avg)):
          y_pred_df_avg[i] = y_pred_df_avg[i].loc[:,~y_pred_df_avg[i].columns.duplicated()]
          y_test_df_avg[i] = y_test_df_avg[i].loc[:,~y_test_df_avg[i].columns.duplicated()]

          y_pred_df_l95ci[i] = y_pred_df_l95ci[i].loc[:,~y_pred_df_l95ci[i].columns.duplicated()]
          y_test_df_l95ci[i] = y_test_df_l95ci[i].loc[:,~y_test_df_l95ci[i].columns.duplicated()]

          y_pred_df_u95ci[i] = y_pred_df_u95ci[i].loc[:,~y_pred_df_u95ci[i].columns.duplicated()]
          y_test_df_u95ci[i] = y_test_df_u95ci[i].loc[:,~y_test_df_u95ci[i].columns.duplicated()]

        # Grafica los resultados
        fig, ax = plt.subplots(figsize=(19, len(y_test_df_avg) * 6) if self._type == 'multiple' else (25, len(y_test_df_avg) * 6),
                              subplot_kw={'projection': '3d'},
                              nrows=len(y_test_df_avg),
                              ncols=3
                              )
        # BUG: Recorrer las columnas de los grupos porque se cambiaron los nombres
        for m in range(len(y_test_df_avg)):
          # Renombrar las columnas que acaban en .1
          y_test_df_avg[m].rename(columns=lambda x: x.split(".")[0], inplace=True)
          y_pred_df_avg[m].rename(columns=lambda x: x.split(".")[0], inplace=True)

          y_test_df_l95ci[m].rename(columns=lambda x: x.split(".")[0], inplace=True)
          y_pred_df_l95ci[m].rename(columns=lambda x: x.split(".")[0], inplace=True)

          y_test_df_u95ci[m].rename(columns=lambda x: x.split(".")[0], inplace=True)
          y_pred_df_u95ci[m].rename(columns=lambda x: x.split(".")[0], inplace=True)

        # Coger el primer nombre de columna de cada grupo
        colum_names = [y_test_df_avg[i].columns[0] for i in range(len(y_test_df_avg))]

        # Graficar los valores de cada grupo de columnas
        for i in range(len(y_test_df_avg)):
          # Renombrar las columnas
          cols = y_test_df_avg[i].columns
          new_cols = [int(col.split("_UPTO_")[1]) for col in cols] # Extrae el número de la columna
          y_test_df_avg[i].columns = new_cols
          y_pred_df_avg[i].columns = new_cols
          y_test_df_l95ci[i].columns = new_cols
          y_pred_df_l95ci[i].columns = new_cols
          y_test_df_u95ci[i].columns = new_cols
          y_pred_df_u95ci[i].columns = new_cols

          for col in y_test_df_avg[i].columns:
            # Graficar los valores
            ax[i][0].scatter(Xy_test_df['HBA1C'], y_test_df_avg[i][col], y_test_df_l95ci[i][col].values, color=(0.3, 0.8, 0.5))
            ax[i][0].scatter(Xy_test_df['HBA1C'], y_test_df_avg[i][col], y_pred_df_l95ci[i][col].values, color=(0.8, 0.7, 0.2))

            ax[i][1].scatter(Xy_test_df['HBA1C'], y_test_df_avg[i][col], y_test_df_avg[i][col].values, color=(0.3, 0.8, 0.5))
            ax[i][1].scatter(Xy_test_df['HBA1C'], y_test_df_avg[i][col], y_pred_df_avg[i][col].values, color=(0.8, 0.7, 0.2))

            ax[i][2].scatter(Xy_test_df['HBA1C'], y_test_df_avg[i][col], y_test_df_u95ci[i][col].values, color=(0.3, 0.8, 0.5))
            ax[i][2].scatter(Xy_test_df['HBA1C'], y_test_df_avg[i][col], y_pred_df_u95ci[i][col].values, color=(0.8, 0.7, 0.2))

          for k in range(3):
            ax[i][k].set_yticks(y_test_df_avg[i].columns)

            # Configurar la gráfica
            ax[i][k].set_xlabel('Nivel de HBA1C', fontsize=8, fontweight='bold')
            ax[i][k].set_ylabel('Edad (hasta)', fontsize=8, fontweight='bold')
            ax[i][k].set_zlabel('Porcentaje de personas afectadas', fontsize=8, fontweight='bold')

          # Añadir titulo a cada gráfica
          ax[i][0].set_title('Intervalo de confianza inferior', fontweight='bold', fontsize=10)
          ax[i][1].set_title('Valor medio', fontweight='bold', fontsize=10)
          ax[i][2].set_title('Intervalo de confianza superior', fontweight='bold', fontsize=10)

          # Añadir titulo a cada fila en el medio de la gráfica
          if 'HF' in colum_names[i]:
            ax[i][1].set_title('Fallo Cardiaco (UPTO)', fontsize=10, fontweight='bold')
          elif 'MI' in colum_names[i]:
            ax[i][1].set_title('Infarto de Miocardio (UPTO)', fontsize=10, fontweight='bold')

          elif 'ANGINA' in colum_names[i]:
            ax[i][1].set_title('Angina (UPTO)', fontsize=10, fontweight='bold')

          elif 'STROKE' in colum_names[i]:
            ax[i][1].set_title('Ictus (UPTO)', fontsize=10, fontweight='bold')

          elif 'BLI' in colum_names[i]:
            ax[i][1].set_title('Ceguera (UPTO)', fontsize=10, fontweight='bold')

          elif 'ME' in colum_names[i]:
            ax[i][1].set_title('Edema macular diabético (UPTO)', fontsize=10, fontweight='bold')

          elif 'BGRET' in colum_names[i]:
            ax[i][1].set_title('Retinopatía de fondo (UPTO)', fontsize=10, fontweight='bold')

          elif 'PRET' in colum_names[i]:
            ax[i][1].set_title('Retinopatía proliferativa (UPTO)', fontsize=10, fontweight='bold')

          elif 'NEU' in colum_names[i]:
            ax[i][1].set_title('Neuropatía individual (UPTO)', fontsize=10, fontweight='bold')

          elif 'LEA' in colum_names[i]:
            ax[i][1].set_title('Amputación extremidades inferiores (UPTO)', fontsize=10, fontweight='bold')

          elif 'ALB1' in colum_names[i]:
            ax[i][1].set_title('Microalbuminuria (UPTO)', fontsize=10, fontweight='bold')

          elif 'ALB2' in colum_names[i]:
            ax[i][1].set_title('Macroalbuminuria (UPTO)')

      # Agrega el título general
      fig.suptitle(f'{self._name}', fontweight='bold', fontsize=15 if self._type == 'multiple' or self._type == 'global' else 12)

      # Agrega la leyenda
      fig.legend(['Valor generado (modelo simulación)', 'Valor predicho (modelo subrogado)'], fontsize=10, ncol=2)

      # Configura el layout
      fig.set_layout_engine('constrained')

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
