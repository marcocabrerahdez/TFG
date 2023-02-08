""" Funciones de utilidad """

import os
import glob
import shutil

import pandas as pd
import settings as st

def search_model_file(model_name: str, directory_name: str) -> str:
  ''' Busca el archivo de un modelo.
  '''
  # directory_name es el directori padre de model_name
  # Buscamos el archivo del modelo
  model_file = glob.glob(os.path.join(st.METRICS_DIR, '**',
                          directory_name, model_name + '*.xlsx'))

  # Si no encontramos el archivo, devolvemos error
  if not model_file:
    return print('No se encontró el archivo del modelo.', model_name)

  # Devolvemos el archivo
  return model_file[0]



def get_score_file(filename: str, path: str) -> pd.DataFrame:
  # Añadir a filename el .xlsx
  filename = filename + '.xlsx'

  results = pd.DataFrame()

  for root, dirs, files in os.walk(path):
    if filename in files:
      results = pd.concat([results, pd.read_excel(os.path.join(root, filename))], axis=1)

  # Eliminar duplicados
  results = results.loc[:,~results.columns.duplicated()]
  # Las columnas van en orden 'single', 'multiple', 'global'
  results = results.reindex(columns=['Modelo', 'single', 'multiple', 'global'])

  return results



def get_model_results(model_name: str, directory_name: str) -> pd.DataFrame:
  ''' Obtiene los resultados de un modelo.
  '''
  # Buscamos el archivo del modelo
  model_file = search_model_file(model_name, directory_name)
  # Si no encontramos el archivo, devolvemos error
  if not model_file or not directory_name:
    return print('No se encontró el archivo del modelo o directorio.',
                  model_name, directory_name)

  # Leemos el archivo y lo devolvemos
  return pd.read_excel(model_file)



def save_splitted_data(_X_train, _X_test, _y_train, _y_test, _columns_X, _columns_Y, _name, _type) -> None:
  # Concatenenar dataframes
  if _type == 'single':
    df_X_train = pd.DataFrame(_X_train, columns=_columns_X.columns)
    df_X_test = pd.DataFrame(_X_test, columns=_columns_X.columns)
    df_y_train = pd.DataFrame(_y_train, columns=_columns_Y.columns)
    df_y_test = pd.DataFrame(_y_test, columns=_columns_Y.columns)
  else:
    df_X_train = pd.DataFrame(_X_train, columns=_columns_X)
    df_X_test = pd.DataFrame(_X_test, columns=_columns_X)
    df_y_train = pd.DataFrame(_y_train, columns=_columns_Y)
    df_y_test = pd.DataFrame(_y_test, columns=_columns_Y)

  if _type == 'single':
    if not os.path.exists(os.path.join(st.SPLITED_DATA_SINGLE, _name)):
      os.makedirs(os.path.join(st.SPLITED_DATA_SINGLE, _name))
    df_X_train.to_excel(os.path.join(st.SPLITED_DATA_SINGLE, _name, 'X_train.xlsx'), index=False)
    df_X_test.to_excel(os.path.join(st.SPLITED_DATA_SINGLE, _name, 'X_test.xlsx'), index=False)
    df_y_train.to_excel(os.path.join(st.SPLITED_DATA_SINGLE, _name, 'y_train.xlsx'), index=False)
    df_y_test.to_excel(os.path.join(st.SPLITED_DATA_SINGLE, _name, 'y_test.xlsx'), index=False)
  elif _type == 'multiple':
    if not os.path.exists(os.path.join(st.SPLITED_DATA_MULTIPLE, _name)):
      os.makedirs(os.path.join(st.SPLITED_DATA_MULTIPLE, _name))
    df_X_train.to_excel(os.path.join(st.SPLITED_DATA_MULTIPLE, _name, 'X_train.xlsx'), index=False)
    df_X_test.to_excel(os.path.join(st.SPLITED_DATA_MULTIPLE, _name, 'X_test.xlsx'), index=False)
    df_y_train.to_excel(os.path.join(st.SPLITED_DATA_MULTIPLE, _name, 'y_train.xlsx'), index=False)
    df_y_test.to_excel(os.path.join(st.SPLITED_DATA_MULTIPLE, _name, 'y_test.xlsx'), index=False)
  elif _type == 'global':
    if not os.path.exists(os.path.join(st.SPLITED_DATA_GLOBAL, _name)):
      os.makedirs(os.path.join(st.SPLITED_DATA_GLOBAL, _name))
    df_X_train.to_excel(os.path.join(st.SPLITED_DATA_GLOBAL, _name, 'X_train.xlsx'), index=False)
    df_X_test.to_excel(os.path.join(st.SPLITED_DATA_GLOBAL, _name, 'X_test.xlsx'), index=False)
    df_y_train.to_excel(os.path.join(st.SPLITED_DATA_GLOBAL, _name, 'y_train.xlsx'), index=False)
    df_y_test.to_excel(os.path.join(st.SPLITED_DATA_GLOBAL, _name, 'y_test.xlsx'), index=False)
  else:
    raise ValueError('El tipo de modelo no es válido')



def get_splited_data(_trained_data_names, _type):
  """ Obtiene los datos de entrenamiento y testeo de los modelos entrenados """
  _X_train = pd.DataFrame()
  _X_test = pd.DataFrame()
  _y_train = pd.DataFrame()
  _y_test = pd.DataFrame()
  for name in _trained_data_names:
    # Concatenar dataframes
    if _type == 'multiple':
      _X_train = pd.concat([_X_train, pd.read_excel(os.path.join(st.SPLITED_DATA_SINGLE, name, 'X_train.xlsx'))], axis=0)
      _X_test = pd.concat([_X_test, pd.read_excel(os.path.join(st.SPLITED_DATA_SINGLE, name, 'X_test.xlsx'))], axis=0)
      _y_train = pd.concat([_y_train, pd.read_excel(os.path.join(st.SPLITED_DATA_SINGLE, name, 'y_train.xlsx'))], axis=1)
      _y_test = pd.concat([_y_test, pd.read_excel(os.path.join(st.SPLITED_DATA_SINGLE, name, 'y_test.xlsx'))], axis=1)
    elif _type == 'global':
      _X_train = pd.concat([_X_train, pd.read_excel(os.path.join(st.SPLITED_DATA_MULTIPLE, name, 'X_train.xlsx'))], axis=0)
      _X_test = pd.concat([_X_test, pd.read_excel(os.path.join(st.SPLITED_DATA_MULTIPLE, name, 'X_test.xlsx'))], axis=0)
      _y_train = pd.concat([_y_train, pd.read_excel(os.path.join(st.SPLITED_DATA_MULTIPLE, name, 'y_train.xlsx'))], axis=1)
      _y_test = pd.concat([_y_test, pd.read_excel(os.path.join(st.SPLITED_DATA_MULTIPLE, name, 'y_test.xlsx'))], axis=1)

  # Eliminar duplicados
  _X_train.drop_duplicates(inplace=True)
  _X_test.drop_duplicates(inplace=True)
  _y_train.drop_duplicates(inplace=True)
  _y_test.drop_duplicates(inplace=True)

  # Reiniciar los índices
  _X_train.reset_index(drop=True, inplace=True)
  _X_test.reset_index(drop=True, inplace=True)
  _y_train.reset_index(drop=True, inplace=True)
  _y_test.reset_index(drop=True, inplace=True)

  _X_columns = _X_train.columns
  _y_columns = _y_train.columns

  return _X_train, _X_test, _y_train, _y_test, _X_columns, _y_columns
