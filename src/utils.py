"""Utility functions for the project."""

import os
from typing import List

import pandas as pd

import settings as st


def save_splitted_data(
    _X_train: pd.DataFrame,
    _X_test: pd.DataFrame,
    _y_train: pd.DataFrame,
    _y_test: pd.DataFrame,
    _name: str,
    _type: str,
) -> None:
    """Saves the splitted data into the corresponding directory.

    Args:
        _X_train (pd.DataFrame): The training data.
        _X_test (pd.DataFrame): The testing data.
        _y_train (pd.DataFrame): The training labels.
        _y_test (pd.DataFrame): The testing labels.
        _name (str): The name of the model.
        _type (str): The type of data: 'multiple' or 'global'.
    """
    # Directory where the data will be saved, depending on the type (single, multiple or global)
    if (_type == 'single'):
        directory = st.SPLITED_DATA_SINGLE
    elif (_type == 'multiple'):
        directory = st.SPLITED_DATA_MULTIPLE
    elif (_type == 'global'):
        directory = st.SPLITED_DATA_GLOBAL
    else:
        raise ValueError('Invalid type:', _type)

    #  Check if the directory exists, if not, create it
    if not os.path.exists(os.path.join(directory, _name)):
        os.makedirs(os.path.join(directory, _name))

    #  Save each DataFrame into a different file
    _X_train.to_excel(os.path.join(
        directory, _name, 'X_train.xlsx'), index=True)
    _X_test.to_excel(os.path.join(
        directory, _name, 'X_test.xlsx'), index=True)
    _y_train.to_excel(os.path.join(
        directory, _name, 'y_train.xlsx'), index=True)
    _y_test.to_excel(os.path.join(
        directory, _name, 'y_test.xlsx'), index=True)


def get_splited_data(_trained_data_names: List[str], _type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns the splitted data for the given model.

    Args:
      _trained_data_names (list): The names of the trained data.
      _type (str): The type of data: 'multiple' or 'global'.

    Returns:
        x_train (pd.DataFrame): The training data.
        x_test (pd.DataFrame): The testing data.
        y_train (pd.DataFrame): The training labels.
        y_test (pd.DataFrame): The testing labels.
    """
    # Initialize the DataFrames
    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    directory = 'single' if _type == 'multiple' else 'multiple'

    # Iterate over the trained data names
    for name in _trained_data_names:
        # Search for file
        x_train_file = os.path.join(
            st.SPLITED_DATA, directory, name, 'X_train.xlsx')
        x_test_file = os.path.join(
            st.SPLITED_DATA, directory, name, 'X_test.xlsx')
        y_train_file = os.path.join(
            st.SPLITED_DATA, directory, name, 'y_train.xlsx')
        y_test_file = os.path.join(
            st.SPLITED_DATA, directory, name, 'y_test.xlsx')

        # Raise an error if the file is not found
        if not x_train_file or not x_test_file or not y_train_file or not y_test_file:
            raise FileNotFoundError('File not found: ' + name)

        # Read the file and append it to the DataFrame
        x_train = pd.concat([x_train, pd.read_excel(
            x_train_file, index_col=0)], axis=0)
        x_test = pd.concat([x_test, pd.read_excel(
            x_test_file, index_col=0)], axis=0)
        y_train = pd.concat([y_train, pd.read_excel(
            y_train_file, index_col=0)], axis=1)
        y_test = pd.concat([y_test, pd.read_excel(
            y_test_file, index_col=0)], axis=1)

    # Remove duplicates
    x_train.drop_duplicates(inplace=True)
    x_test.drop_duplicates(inplace=True)

    return x_train, x_test, y_train, y_test


def get_test_file(name: str, flag: str) -> pd.DataFrame:
    """Returns the test file for the given model.

    Args:
        name (str): The name of the test file.
        flag (str): The flag of the test file. Can be 'x' or 'y'.

    Returns:
        pd.DataFrame: The test file.
    """
    # Search for the test file
    if (flag == 'y'):
        test_file = os.path.join(
            st.SPLITED_DATA, 'single', name, 'y_test.xlsx')
    elif (flag == 'x'):
        test_file = os.path.join(
            st.SPLITED_DATA, 'single', name, 'X_test.xlsx')
    else:
        raise ValueError('Invalid flag:', flag)

    # Raise an error if the test file is not found
    if not test_file:
        raise FileNotFoundError('Test file not found: ' + name)

    # Read the file and return it
    return pd.read_excel(test_file, index_col=0)


def get_prediction_file(model: str, folder_prediction: str, type: str, name: str) -> pd.DataFrame:
    """Returns the prediction file for the given model.

    Args:
        model (str): The name of the model.
        folder_prediction (str): The name of the folder where the prediction file is stored.
        type (str): The type of the model. Can be 'single', 'multiple', or 'global'.
        name (str): The name of the prediction file.

    Returns:
        pd.DataFrame: The prediction file.
    """
    # Search for the prediction file
    prediction_file = os.path.join(st.PREDICTIONS_DIR, type, model,
                                   folder_prediction, name + '.xlsx')

    # Raise an error if the prediction file is not found
    if not prediction_file:
        raise FileNotFoundError('Prediction file not found: ' + name)

    # Read the file and return it
    return pd.read_excel(prediction_file)


def delete_nan_values(y_test: pd.DataFrame, x_test: pd.DataFrame, name: str) -> pd.DataFrame:
    """Deletes the NaN values from the given DataFrame.

    Args:
        y_test (pd.DataFrame): The DataFrame to delete the NaN values from.
        x_test (pd.DataFrame): The DataFrame to delete the NaN values from.
        name (str): The name of the model.

    Returns:
        pd.DataFrame: The DataFrame without NaN values.
    """
    if (name == 'Fallo Cardiaco'):
        comorbidity = 'HF'
    elif (name == 'Infarto de Miocardio'):
        comorbidity = 'MI'
    elif (name == 'Angina'):
        comorbidity = 'ANGINA'
    elif (name == 'Ictus'):
        comorbidity = 'STROKE'
    elif (name == 'Ceguera'):
        comorbidity = 'BLI'
    elif (name == 'Edema macular diabético'):
        comorbidity = 'ME'
    elif (name == 'Retinopatía de fondo'):
        comorbidity = 'BGRET'
    elif (name == 'Retinopatía proliferativa'):
        comorbidity = 'PRET'
    elif (name == 'Neuropatía'):
        comorbidity = 'NEU'
    elif (name == 'Amputación extremidades inferiores'):
        comorbidity = 'LEA'
    elif (name == 'Microalbuminuria'):
        comorbidity = 'ALB1'
    elif (name == 'Macroalbuminuria'):
        comorbidity = 'ALB2'
    elif (name == 'Enfermedad renal terminal'):
        comorbidity = 'ESRD'
    else:
        raise ValueError('Invalid name:', name)

    y_test = y_test.loc[x_test[comorbidity] != 1]
    return y_test
