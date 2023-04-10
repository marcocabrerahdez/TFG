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
from api import api
from scripts import automl as ml
from scripts import compare as cp


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

    # Save the nan values ​​in a boolean matrix
    nan_pos = data_frame.isna()

    # Fill the nan values ​​with 0
    data_frame = data_frame.fillna(0)

    # For each model configuration
    for config in config_list['config_list']:
        # Create the model
        if config['type'] == 'single':
            automl = ml.AutoML(config['name'], config['class_name'],
                               config['model'], config['type'], config['params'],
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

        # R2 and MAPE score
        automl.metrics(nan_pos, increment=3)

        # Save the model and the results
        automl.save()

        # Generate the graphs (UPTO and AVG_TIME)
        # automl.plot_upto_time() # Uncomment to generate the graphs Upto time
        automl.plot_avg_time(nan_pos)

    # Comparar las métricas de los resultados de los modelos
    cp.create_score_table(compare_list['r2']['list'], compare_list['r2']['name_list'], st.R2_TABLE_DIR, st.R2_AVERAGE_TIME_DIR)
    cp.create_score_table(compare_list['mape']['list'], compare_list['mape']['name_list'], st.MAPE_TABLE_DIR, st.MAPE_AVERAGE_TIME_DIR)
    cp.compare_r2_tables(
        compare_list['r2']['name_list'], st.R2_INCIDENCE_PLOT_DIR, st.R2_INCIDENCE_DIR)


if __name__ == '__main__':
    main()
