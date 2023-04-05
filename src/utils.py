""" Funciones de utilidad """

import os
import glob
import shutil

import pandas as pd
import numpy as np
import settings as st

def search_model_file(model_name: str, directory_name: str) -> str:
  '''Searches for the file of a given model.

  Args:
    model_name (str): Name of the model.
    directory_name (str): Name of the directory where the model is located.

  Returns:
    str: Path of the model file.

  '''
  # directory_name is the parent directory of model_name
  # Search for the model file
  model_file = glob.glob(os.path.join(st.METRICS_DIR, '**', directory_name, model_name + '*.xlsx'))

  # If the file is not found, return an error
  if not model_file:
    return print('Model file not found:', model_name)

  # Return the file path
  return model_file[0]



def get_score_file(filename: str, path: str) -> pd.DataFrame:
  """
  Returns a DataFrame containing scores for a given filename and path.

  Args:
    filename (str): The filename to search for.
    path (str): The path to search in.

  Returns:
    pd.DataFrame: A DataFrame containing the scores for the given file.
  """
  # Add .xlsx extension to filename
  filename = f"{filename}.xlsx"

  results = pd.DataFrame()

  # Search for filename in path
  for root, dirs, files in os.walk(path):
    if filename in files:
      results = pd.concat([results, pd.read_excel(os.path.join(root, filename))], axis=1)

  # Remove duplicates and reorder columns
  results = results.loc[:, ~results.columns.duplicated()]
  results = results.reindex(columns=["Modelo", "single", "multiple", "global"])

  return results



def get_model_results(model_name: str, directory_name: str) -> pd.DataFrame:
  """Obtains the results of a model.

  Args:
    model_name (str): The name of the model.
    directory_name (str): The directory name where the model file is located.

  Returns:
    pd.DataFrame: The results of the model in a pandas DataFrame.
  """
  # Search for the model file
  model_file = search_model_file(model_name, directory_name)

  # Raise an error if the model file or directory is not found
  if not model_file or not directory_name:
    raise FileNotFoundError('Model file or directory not found: ' + model_name + ', ' + directory_name)

  # Read the file and return it
  return pd.read_excel(model_file, index_col=0)



def save_splitted_data(_X_train, _X_test, _y_train, _y_test, _columns_X, _columns_Y, _name, _type,) -> None:
  """Saves the splitted data into different directories based on the type of model.

  Args:
    _X_train (pd.DataFrame): The training data input features.
    _X_test (pd.DataFrame): The test data input features.
    _y_train (pd.DataFrame): The training data target feature.
    _y_test (pd.DataFrame): The test data target feature.
    _columns_X (pd.DataFrame): The column names of the input features.
    _columns_Y (pd.DataFrame): The column names of the target feature.
    _name (str): The name of the model.
    _type (str): The type of the model. Can be 'single', 'multiple', or 'global'.

  Returns:
    None
  """
  # Concatenate dataframes
  if _type == 'single':
    df_X_train = pd.DataFrame(_X_train, columns=_columns_X.columns)
    df_X_test = pd.DataFrame(_X_test, columns=_columns_X.columns)
    df_y_train = pd.DataFrame(_y_train, columns=_columns_Y.columns, index=_y_train.index)
    df_y_test = pd.DataFrame(_y_test, columns=_columns_Y.columns, index=_y_test.index)
  else:
    df_X_train = pd.DataFrame(_X_train, columns=_columns_X)
    df_X_test = pd.DataFrame(_X_test, columns=_columns_X)
    df_y_train = pd.DataFrame(_y_train, columns=_columns_Y, index=_y_train.index)
    df_y_test = pd.DataFrame(_y_test, columns=_columns_Y, index=_y_test.index)

  # Save data based on the model type
  if _type == 'single':
    path = os.path.join(st.SPLITED_DATA_SINGLE, _name)
  elif _type == 'multiple':
    path = os.path.join(st.SPLITED_DATA_MULTIPLE, _name)
  elif _type == 'global':
    path = os.path.join(st.SPLITED_DATA_GLOBAL, _name)
  else:
    raise ValueError('El tipo de modelo no es válido')

  if not os.path.exists(path):
      os.makedirs(path)

  df_X_train.to_excel(os.path.join(path, 'X_train.xlsx'), index=False)
  df_X_test.to_excel(os.path.join(path, 'X_test.xlsx'), index=False)
  df_y_train.to_excel(os.path.join(path, 'y_train.xlsx'), index=True)
  df_y_test.to_excel(os.path.join(path, 'y_test.xlsx'), index=True)



def get_splited_data(_trained_data_names, _type):
  """
  Returns the training and testing data for the trained models.

  Args:
    trained_data_names (list): List of names of trained data.
    data_type (str): Type of data: 'multiple' or 'global'.

  Returns:
    tuple: Tuple containing the training and testing data and corresponding columns.
  """
  X_train = pd.DataFrame()
  X_test = pd.DataFrame()
  y_train = pd.DataFrame()
  y_test = pd.DataFrame()

  for name in _trained_data_names:
    path = st.SPLITED_DATA_SINGLE if _type == 'multiple' else st.SPLITED_DATA_MULTIPLE
    X_train = pd.concat([X_train, pd.read_excel(os.path.join(path, name, 'X_train.xlsx'))], axis=0)
    X_test = pd.concat([X_test, pd.read_excel(os.path.join(path, name, 'X_test.xlsx'))], axis=0)
    y_train = pd.concat([y_train, pd.read_excel(os.path.join(path, name, 'y_train.xlsx'))], axis=1)
    y_test = pd.concat([y_test, pd.read_excel(os.path.join(path, name, 'y_test.xlsx'))], axis=1)

  # Remove duplicates
  X_train.drop_duplicates(inplace=True)
  X_test.drop_duplicates(inplace=True)

  # Set index to the first column of y_test
  y_test.index = y_test.iloc[:, 0]
  y_train.index = y_train.iloc[:, 0]

  # Remove columns named 'Unnamed: 0'
  y_test = y_test.loc[:, ~y_test.columns.str.contains('^Unnamed')]
  y_train = y_train.loc[:, ~y_train.columns.str.contains('^Unnamed')]

  # Remove the name of the index
  y_test.index.name = None
  y_train.index.name = None
  """
  if _type == 'global':
    y_test.drop(y_test.columns.duplicated(), axis=1, inplace=True)
    y_train.drop(y_train.columns.duplicated(), axis=1, inplace=True)
  """

  X_columns = X_train.columns
  y_columns = y_train.columns

  return X_train, X_test, y_train, y_test, X_columns, y_columns
