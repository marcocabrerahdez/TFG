"""Compare functions for the models."""

import os
import pandas as pd
from typing import List
from sklearn.metrics import (mean_absolute_percentage_error, r2_score)

import settings as st


def create_tables(y_test: pd.DataFrame, single_list: List[pd.DataFrame],
                  multiple_list: List[pd.DataFrame], global_list: List[pd.DataFrame],
                  model: List[str], name: str, folder: str) -> None:
    """Create Excel files for the R-squared scores for each model.

    Args:
        name_list (List[str]): A list of strings, each representing the name of a model.
        model_list (List[str]): A list of strings, each representing the name of a model.
        folder_prediction (str): A string representing the directory where the predictions are stored.

    Returns:
        None
    """
    print("Creating tables...")

    # Create a dataframe to store the R-squared and MAPE scores
    r2 = pd.DataFrame(columns=['Modelo', 'single', 'multiple', 'global'])
    mape = pd.DataFrame(columns=['Modelo', 'single', 'multiple', 'global'])

    # Calculate the R2 score for the current model
    for i in range(len(model)):
        single_predicction = single_list[i]
        multiple_predicction = multiple_list[i]
        global_predicction = global_list[i]

        # Calculate the R2 score for the current model
        single_r2 = r2_score(y_test, single_predicction)
        multiple_r2 = r2_score(y_test, multiple_predicction)
        global_r2 = r2_score(y_test, global_predicction)

        # Calculate the MAPE for the current model
        single_mape = mean_absolute_percentage_error(
            y_test, single_predicction)
        multiple_mape = mean_absolute_percentage_error(
            y_test, multiple_predicction)
        global_mape = mean_absolute_percentage_error(
            y_test, global_predicction)

        # Append the results to the DataFrame using concat
        r2 = r2.append({'Modelo': model[i], 'single': single_r2,
                       'multiple': multiple_r2, 'global': global_r2}, ignore_index=True)
        mape = mape.append({'Modelo': model[i], 'single': single_mape,
                           'multiple': multiple_mape, 'global': global_mape}, ignore_index=True)

    # Create the Excel file with the results
    os.makedirs(os.path.join(st.R2_TABLE_DIR, folder), exist_ok=True)
    r2.to_excel(
        os.path.join(st.R2_TABLE_DIR, folder, f"{name}.xlsx"),
        index=False
    )

    os.makedirs(os.path.join(st.MAPE_TABLE_DIR, folder), exist_ok=True)
    mape.to_excel(
        os.path.join(st.MAPE_TABLE_DIR, folder, f"{name}.xlsx"),
        index=False
    )
