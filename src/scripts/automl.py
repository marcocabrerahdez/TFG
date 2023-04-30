"""Module that contains the AutoML class.

This class automates the training process of Machine Learning models.
"""

import importlib
import os

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import settings as st
import utils as ut


class AutoML:
    """Class that automates the training process of Machine Learning models.

    Attributes:
        name (str): Name of the model.
        model (str): Machine Learning model.
        params (hash): Parameters of the model.
        df (pd.DataFrame): Input data.
        columns_X (List[str]): Input columns.
        columns_Y (List[str]): Output columns.
        X_train (pd.DataFrame): Training data.
        X_test (pd.DataFrame): Test data.
        y_train (pd.DataFrame): Training output data.
        y_test (pd.DataFrame): Test output data.
        y_pred (pd.DataFrame): Predicted output data.
        model_list_names (List[str]): List of model names.
        trained_data_names (List[str]): List of trained data names.
        index_list_y_test (List[int]): List of indexes of the test data.
    """
    __slots__ = [
        '_name',
        '_class_name',
        '_model',
        '_type',
        '_params',
        '_X_train',
        '_X_test',
        '_y_train',
        '_y_test',
        '_y_pred',
        '_model_list_names',
        '_trained_data_names'
    ]

    def __init__(
            self,
            name: str,
            class_name,
            model,
            type_model: str,
            params,
            trained_data_names=[],
            columns_X: pd.DataFrame = pd.DataFrame(),
            columns_Y: pd.DataFrame = pd.DataFrame()) -> None:
        """Constructor for the class.

        Args:
            name (str): Name of the model.
            class_name (str): Name of the Machine Learning class.
            model (str): Machine Learning model.
            type_model (str): Type of Machine Learning model.
            params (hash): Parameters of the model.
            trained_data_names (List[str]): List of names of trained data.
            columns_X (pd.DataFrame): Input columns.
            columns_y (pd.DataFrame): Output columns.

        Returns:
            None
        """
        # Initialize the attributes
        self._name = name
        self._model_list_names = model
        self._class_name = [importlib.import_module(class_name[i])
                            for i in range(len(class_name))]

        self._type = type_model
        self._params = params
        self._trained_data_names = trained_data_names
        self._y_pred = None

        # Create the model
        self._model = []
        for i in range(len(model)):
            if (model[i] == 'LinearRegression'):
                self._model.append(getattr(self._class_name[i], model[i])())
            else:
                self._model.append(getattr(self._class_name[i], model[i])(
                    random_state=st.RANDOM_STATE))
        if self._type == 'single':
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
                columns_X, columns_Y, test_size=st.TEST_SIZE, random_state=st.RANDOM_STATE)

        elif self._type == 'multiple' or self._type == 'global':
            self._X_train, self._X_test, self._y_train, self._y_test = ut.get_splited_data(
                self._trained_data_names, self._type)
        else:
            raise Exception('El tipo de modelo no es válido.')

        # Save the data use by single models to be able to use them in multiple
        # and global models
        ut.save_splitted_data(
            self._X_train,
            self._X_test,
            self._y_train,
            self._y_test,
            self._name,
            self._type)

        # Get the columns of the input data
        columns = self._X_train.columns.copy()

        # Preprocces the data using the StandardScaler to normalize the data
        scaler = StandardScaler()
        scaler.fit(self._X_train)
        self._X_train = scaler.transform(self._X_train)
        self._X_test = scaler.transform(self._X_test)

        # Assign the correct data type to the data to allow the use of the data
        # in the models
        self._X_train = pd.DataFrame(
            self._X_train, columns=columns)
        self._X_test = pd.DataFrame(
            self._X_test, columns=columns)

    def train(self) -> None:
        """Train the models.

        Returns:
            None
        """
        print('Training the models...')
        # Create a pipeline with the model and the parameters to be used in the
        # grid search
        pipe = [Pipeline([('model', self._model[i])])
                for i in range(len(self._model))]

        # Create a grid search with the pipeline and the parameters
        # Grid search is used to find the best parameters for the model
        grid_search = [
            GridSearchCV(
                pipe[i],
                param_grid=self._params[i],
                cv=5,
                refit=True,
                scoring='r2') for i in range(
                len(pipe))]

        # Create a multi output regressor to be able to train the model with
        # multiple outputs
        self._model = [MultiOutputRegressor(grid_search[i])
                       for i in range(len(grid_search))]

        # Train the model
        for i in range(len(self._model)):
            self._model[i].fit(self._X_train, self._y_train)

    def predict(self) -> None:
        """Predict the output data.

        Returns:
            None
        """
        print('Predicting the data...')
        # The predict method returns a list of lists,
        # so the data will be converted in save_predictions_results
        self._y_pred = [self._model[i].predict(self._X_test)
                        for i in range(len(self._model))]

    def _save_predictions_results(self) -> None:
        """Save the predictions results in a .xlsx file.

        Raises:
            ValueError: If the type of model is not valid.

        """
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
        """Save the model in a .pkl file.

        Returns
            None
        """
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
        """Save the model and the predictions results.

        Returns
            None
        """
        print('Saving the results...')
        self._save_model()
        self._save_predictions_results()
