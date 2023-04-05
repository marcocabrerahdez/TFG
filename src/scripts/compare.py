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

    # Plot and save bar plots for single, multiple, and global R^2 scores
    for i, col_name in enumerate(["single", "multiple", "global"]):
      fig, ax = plt.subplots(figsize=(10, 10))

      # Plot bar chart
      results.plot.bar(ax=ax, width=0.4, x="Modelo", y=col_name, color="deepskyblue", rot=0)

      # Annotate bars with their values
      for p in ax.patches:
          ax.annotate(str(round(p.get_height(), 7)), (p.get_x() + p.get_width() / 2., p.get_height()), ha="center", va="center", xytext=(0, 10), textcoords="offset points")

      # Set x and y labels and title
      ax.set_xlabel("Modelo", fontsize=12, fontweight="bold")
      ax.set_ylabel("$\\mathbf{R}^\\mathbf{2}$", fontsize=12, fontweight="bold")

      if i == 0:
        ax.set_title(f"Comparación entrenamiento single del $\\mathbf{R}^\\mathbf{2}$ para {name}", fontsize=15, fontweight="bold")
      elif i == 1:
        ax.set_title(f"Comparación entrenamiento multiple del $\\mathbf{R}^\\mathbf{2}$ para {name}", fontsize=15, fontweight="bold")
      else:
        ax.set_title(f"Comparación entrenamiento global del $\\mathbf{R}^\\mathbf{2}$ para {name}", fontsize=15, fontweight="bold")

      # Remove legend
      ax.legend().set_visible(False)

      # Save figure with appropriate file name and directory
      filename = f"{name} ({col_name.capitalize()}).png"
      filepath = os.path.join(figpath, name, filename)
      os.makedirs(os.path.dirname(filepath), exist_ok=True)
      fig.savefig(filepath, dpi=300, bbox_inches="tight")

      # Close the figure
      plt.close(fig)
