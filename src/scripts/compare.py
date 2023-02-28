import os
import glob
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import settings as st
import utils as ut


def create_score_table(metrics_list: List[str], name_list: List[str], path=st.R2_TABLE_DIR) -> None:
  # Creamos un diccionario para guardar cada dataframe
  df_results = pd.DataFrame()

  # Recorremos la lista de modelos
  for metric, name in zip(metrics_list, name_list):
    # Obtenemos el dataframe de los resultados
    result = ut.get_score_file(metric, path)

    # Guardamos cada dataframe en un excel
    if not os.path.exists(os.path.join(path)):
      os.mkdir(path)
    result.to_excel(os.path.join(path, f'{name}.xlsx'), index=False)

    # Reseteamos el dataframe
    result = pd.DataFrame()