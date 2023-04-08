import os
import glob
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import settings as st
import utils as ut


def create_score_table(metrics_list: List[str], name_list: List[str], path=st.R2_TABLE_DIR, outdir=st.R2_AVERAGE_TIME_DIR) -> None:
  """Create Excel files for the R-squared scores for each model.

  Args:
    metrics_list (List[str]): A list of strings, each representing the name of a metric.
    name_list (List[str]): A list of strings, each representing the name of a model.
    path (str, optional): A string representing the directory where the R-squared tables are stored.
        Defaults to st.R2_TABLE_DIR.
    outdir (str, optional): A string representing the directory where the output Excel files will be saved.
        Defaults to st.R2_AVERAGE_TIME_DIR.

  Returns:
    None
  """
  # Create an empty DataFrame to hold the results
  df_results = pd.DataFrame()

  # Iterate over the list of metrics and model names
  for metric, name in zip(metrics_list, name_list):
    # Retrieve the DataFrame with the score results for the current metric
    result = ut.get_score_file(metric, path)

    # Create a new directory to store the Excel files if it does not exist
    if not os.path.exists(os.path.join(outdir)):
      os.mkdir(outdir)

    # Save the score results for the current model to an Excel file
    result.to_excel(os.path.join(outdir, f'{name}.xlsx'), index=False)

    # Reset the DataFrame for the next iteration
    result = pd.DataFrame()



def compare_r2_tables(name_list: List[str], figpath=st.R2_AVERAGE_TIME_PLOT_DIR, path=st.R2_TABLE_DIR) -> None:
    """ Create and save R2 comparison plots for each model in the given list of names.
        The function reads R2 scores from .xlsx files in the given directory path and generates
        three types of comparison plots: single, multiple, and global R2 scores for each model.

    Args:
      name_list (List[str]): List of names of the models to create R2 comparison plots for.
      figpath (str, optional): Directory path to save the generated R2 comparison plots. Default is st.R2_AVERAGE_TIME_PLOT_DIR.
      path : (str, optional): Directory path to read the R2 scores from. Default is st.R2_TABLE_DIR.

    Returns:
      None
    """
    for name in name_list:
        # Initialize empty dataframe to store results
        results = pd.DataFrame()

        # Search for file in directory and concatenate dataframes
        for root, dirs, files in os.walk(path):
            if f"{name}.xlsx" in files:
                filepath = os.path.join(root, f"{name}.xlsx")
                results = pd.concat([results, pd.read_excel(filepath)], axis=1)

        # Create plot using scatter plot with error bars
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.errorbar(x=range(len(results)), y=results["single"], yerr=results["single"].std(), fmt="o", capsize=5, label="Entrenamiento de tipo single")
        ax.errorbar(x=[i + 0.2 for i in range(len(results))], y=results["multiple"], yerr=results["multiple"].std(), fmt="o", capsize=5, label="Entrenamiento  de tipo multiple")
        ax.errorbar(x=[i + 0.4 for i in range(len(results))], y=results["global"], yerr=results["global"].std(), fmt="o", capsize=5, label="Entrenamiento de tipo global")

        # Set xticks and labels to be the model names
        ax.set_xticks([i + 0.2 for i in range(len(results))])
        ax.set_xticklabels(results["Modelo"])

        # Set x and y labels and title
        ax.set_xlabel("Modelo", fontsize=12, fontweight="bold")
        ax.set_ylabel("$\\mathbf{R}^\\mathbf{2}$", fontsize=12, fontweight="bold")

        # Set title in spanish
        ax.set_title(f"Comparación de $\\mathbf{{R}}^\\mathbf{{2}}$ para {name}", fontsize=14, fontweight="bold")

        # Add legend
        ax.legend()

        # Save figure with appropriate file name and directory
        filename = f"{name}.png"
        os.makedirs(figpath, exist_ok=True)
        fig.savefig(os.path.join(figpath, filename), dpi=300, bbox_inches="tight")

        # Close the figure
        plt.close(fig)
