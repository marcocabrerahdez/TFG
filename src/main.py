"""Main program.

This program is part of a final degree project
at the University of La Laguna.

The objective of this program is to generate prediction models
for the waiting time of type 1 diabetes patients in the Canary Islands.
And compare the results of the simulation models with the prediction models.

The data has been previously processed
and is located in the data directory.
Graphs have been generated with the results
and are located in the figures directory.
In addition, the models have been generated and are located in the model directory.
Finally, the prediction results have been generated
and are located in the predictions directory.

Author:
    Marco Antonio Cabrera Hernández
"""

import json
import os
import sys

import pandas as pd

import settings as st
from scripts import automl as ml
from scripts import compare as cp
from scripts import plot as pl
import utils as ut


def main() -> None:
    """Main function.

    Parameters:
        -h, --help: Displays the program's help.
        -v, --version: Displays the program's version.
        -f, --file: Data file. Must be an Excel file.
          Additionally, you must specify:
            - The sheet of the data file.
            - The model configuration file.
            - The comparison model configuration file.

    Example:
        python3 main.py -v
    """
    if len(sys.argv) > 1:
        # Program help
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            print(main.__doc__)
            sys.exit()
        # Versión del programa
        elif sys.argv[1] == '-v' or sys.argv[1] == '--version':
            print(st.VERSION)
            sys.exit()

        # Data files
        elif sys.argv[1] == '-f' or sys.argv[1] == '--file':
            data_file_path = os.path.join(st.ROOT_DIR, sys.argv[2])
            data_file_sheet = sys.argv[3]
            if os.path.isfile(data_file_path):
                data_frame = pd.read_excel(data_file_path, data_file_sheet)

            # Open the data configuration file
            configuration_model_file_path = os.path.join(
                st.ROOT_DIR, sys.argv[4])
            if os.path.isfile(configuration_model_file_path):
                with open(configuration_model_file_path, 'r', encoding='utf8') as file_name:
                    config_list = json.load(file_name)

            # Open the comparison configuration file
            compare_file_path = os.path.join(st.ROOT_DIR, sys.argv[5])
            if os.path.isfile(compare_file_path):
                with open(compare_file_path, 'r', encoding='utf8') as file_name:
                    compare_list = json.load(file_name)

            # Open the plot configuration file
            plot_file_path = os.path.join(st.ROOT_DIR, sys.argv[6])
            if os.path.isfile(plot_file_path):
                with open(plot_file_path, 'r', encoding='utf8') as file_name:
                    plot_list = json.load(file_name)
        else:
            print('Argumento no válido.')
            sys.exit()

    # Data preprocessing
    df_cols = data_frame.columns[data_frame.columns.str.contains('UPTO')]
    data_frame[df_cols] = data_frame[df_cols].div(500) * 100
    df_cols = data_frame.columns[data_frame.columns.str.contains('INC')]
    data_frame[df_cols] = data_frame[df_cols].div(500) * 100

    # Change SEX: MAN -> 0, WOMAN -> 1
    data_frame['SEX'] = data_frame['SEX'].replace({'MAN': 0, 'WOMAN': 1})

    # Fill the nan values ​​with 0
    data_frame = data_frame.fillna(0)

    # For each model configuration
    for config in config_list['config_list']:
        # Create the model
        if config['type'] == 'single':
            automl = ml.AutoML(config['name'],
                               config['class_name'],
                               config['model'],
                               config['type'],
                               config['params'],
                               columns_X=data_frame[config['columns_X']],
                               columns_Y=data_frame[config['columns_Y']])
        else:
            automl = ml.AutoML(config['name'], config['class_name'],
                               config['model'], config['type'],
                               config['params'], config['trained_data_names'])
        # Train the model
        automl.train()

        # Predict the model
        automl.predict()

        # Save the model and the results
        automl.save(folder='')

    # Generate the plots
    for plot in plot_list['plot_config']:
        for model in plot['model_list']:
            for name in plot['name_list']:
                y_test_file = ut.get_test_file(name, 'y')
                x_test_file = ut.get_test_file(name, 'x')

                # Get the prediction file
                if (plot['type_train'] == 'single'):
                    prediction_file = ut.get_prediction_file(
                        model, plot['prediction_folder'], plot['type_train'], name)
                elif (plot['type_train'] == 'multiple'):
                    if (name == 'Fallo Cardiaco' or name == 'Angina' or name == 'Infarto de Miocardio' or name == 'Ictus'):
                        prediction_file = ut.get_prediction_file(
                            model, plot['prediction_folder'], plot['type_train'], 'Enfermedades Cardíacas')
                    if (name == 'Ceguera' or name == 'Edema macular diabético' or name == 'Retinopatía de fondo' or name == 'Retinopatía proliferativa'):
                        prediction_file = ut.get_prediction_file(
                            model, plot['prediction_folder'], plot['type_train'], 'Retinopatías')
                    if (name == 'Neuropatía' or name == 'Amputación extremidades inferiores'):
                        prediction_file = ut.get_prediction_file(
                            model, plot['prediction_folder'], plot['type_train'], 'Neuropatías')
                    if (name == 'Microalbuminuria' or name == 'Macroalbuminuria' or name == 'Enfermedad renal terminal'):
                        prediction_file = ut.get_prediction_file(
                            model, plot['prediction_folder'], plot['type_train'], 'Nefropatías')

                if (plot['type_train'] == 'global'):
                    prediction_file = ut.get_prediction_file(
                        model, plot['prediction_folder'], plot['type_train'], 'Comorbilidades')

                # Delete the columns that are not in the test file
                if (plot['type_train'] == 'multiple' or plot['type_train'] == 'global'):
                    prediction_file = prediction_file[y_test_file.columns]

                # Set the index
                prediction_file = prediction_file.set_index(y_test_file.index)

                # Delete rows with nan values
                y_test_file = ut.delete_nan_values(
                    y_test_file, x_test_file, name)
                prediction_file = ut.delete_nan_values(
                    prediction_file, x_test_file, name)

                # pl.plot_upto_time(y_test_file, prediction_file, model, plot['type_train'], name)
                pl.plot_avg_time(y_test_file, prediction_file, model, plot['type_train'], name)

    # Compare the results of the models
    for compare in compare_list['compare_config']:
        for name in compare['name_list']:
            y_test_file = ut.get_test_file(name, 'y')
            x_test_file = ut.get_test_file(name, 'x')
            single_list = []
            multiple_list = []
            global_list = []
            # Get the prediction file
            for model in compare['model_list']:
                single_prediction_file = ut.get_prediction_file(
                    model, compare['prediction_folder'], 'single', name)
                if (name == 'Fallo Cardiaco' or name == 'Angina' or name == 'Infarto de Miocardio' or name == 'Ictus'):
                    multiple_prediction_file = ut.get_prediction_file(
                        model, compare['prediction_folder'], 'multiple', 'Enfermedades Cardíacas')
                if (name == 'Ceguera' or name == 'Edema macular diabético' or name == 'Retinopatía de fondo' or name == 'Retinopatía proliferativa'):
                    multiple_prediction_file = ut.get_prediction_file(
                        model, compare['prediction_folder'], 'multiple', 'Retinopatías')
                if (name == 'Neuropatía' or name == 'Amputación extremidades inferiores'):
                    multiple_prediction_file = ut.get_prediction_file(
                        model, compare['prediction_folder'], 'multiple', 'Neuropatías')
                if (name == 'Microalbuminuria' or name == 'Macroalbuminuria' or name == 'Enfermedad renal terminal'):
                    multiple_prediction_file = ut.get_prediction_file(
                        model, compare['prediction_folder'], 'multiple', 'Nefropatías')

                global_prediction_file = ut.get_prediction_file(
                    model, compare['prediction_folder'], 'global', 'Comorbilidades')

                # Delete the columns that are not in the test file
                multiple_prediction_file = multiple_prediction_file[single_prediction_file.columns]
                global_prediction_file = global_prediction_file[single_prediction_file.columns]

                # Set index to be the same as the test data
                single_prediction_file = single_prediction_file.set_index(
                    y_test_file.index)
                multiple_prediction_file = multiple_prediction_file.set_index(
                    y_test_file.index)
                global_prediction_file = global_prediction_file.set_index(
                    y_test_file.index)

                # Delete rows with nan values
                single_prediction_file = ut.delete_nan_values(
                    single_prediction_file, x_test_file, name)
                multiple_prediction_file = ut.delete_nan_values(
                    multiple_prediction_file, x_test_file, name)
                global_prediction_file = ut.delete_nan_values(
                    global_prediction_file, x_test_file, name)

                # Crea a list with the prediction files of each model
                single_list.append(single_prediction_file)
                multiple_list.append(multiple_prediction_file)
                global_list.append(global_prediction_file)

            y_test_file = ut.delete_nan_values(y_test_file, x_test_file, name)
            cp.create_tables(y_test_file, single_list, multiple_list, global_list,
                             compare['model_list'], name, compare['prediction_folder'])


if __name__ == '__main__':
    main()
