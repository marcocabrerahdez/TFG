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


def compare_r2_tables(name_list: List[str], figpath=st.R2_AVERAGE_TIME_PLOT_DIR, path=st.R2_TABLE_DIR) -> None:
  # Creamos un diccionario para guardar cada dataframe
  results = pd.DataFrame()

  for name in name_list:
    # Obtener el archivo
    filename = name + '.xlsx'
    for root, dirs, files in os.walk(path):
      if filename in files:
        results = pd.concat([results, pd.read_excel(os.path.join(root, filename))], axis=1)

    for i in range(3):
      # Gráficar los resultados
      fig, ax = plt.subplots(figsize=(10, 10))

      # Grafica de barras
      results.plot.bar(
        ax=ax,
        width=0.4,
        x='Modelo',
        y=('single' if i == 0 else 'multiple' if i == 1 else 'global'),
        color='deepskyblue',
        rot=0
      )

      # Añadir el valor de cada barra
      for p in ax.patches:
        ax.annotate(
          str(round(p.get_height(), 7)),
          (p.get_x() + p.get_width() / 2., p.get_height()),
          ha='center',
          va='center',
          xytext=(0, 10),
          textcoords='offset points'
        )

      # Etiquetas
      ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
      ax.set_ylabel('$\\mathbf{R}^\\mathbf{2}$', fontsize=12, fontweight='bold')

      # Título
      if i == 0:
        ax.set_title('Comparación entrenamiento single del $\\mathbf{R}^\\mathbf{2}$ para cada modelo', fontsize=15, fontweight='bold')

      elif i == 1:
        ax.set_title('Comparación entrenamiento multiple del $\\mathbf{R}^\\mathbf{2}$ para cada modelo', fontsize=15, fontweight='bold')

      else:
        ax.set_title('Comparación entrenamiento global del $\\mathbf{R}^\\mathbf{2}$ para cada modelo', fontsize=15, fontweight='bold')

      # Eliminar leyenda
      ax.legend().set_visible(False)

      # Guardar la gráfica
      if i == 0:
        # Crear directorio con el nombre del modelo
        if not os.path.exists(os.path.join(figpath, name)):
          os.mkdir(os.path.join(figpath, name))
        fig.savefig(os.path.join(figpath, name, name + ' (Single).png'), dpi=300, bbox_inches='tight')

      elif i == 1:
        if not os.path.exists(os.path.join(figpath, name)):
          os.mkdir(os.path.join(figpath, name))
        fig.savefig(os.path.join(figpath, name, name + ' (Multiple).png'), dpi=300, bbox_inches='tight')

      else:
        if not os.path.exists(os.path.join(figpath, name)):
          os.mkdir(os.path.join(figpath, name))
        fig.savefig(os.path.join(figpath, name, name + ' (Global).png'), dpi=300, bbox_inches='tight')

      # Cerrar la gráfica
      plt.close(fig)

    # Borrar el dataframe
    results = pd.DataFrame()