''' Carga un dataset. '''

import pandas as pd
import settings as st

def load_dataset(data) -> pd.DataFrame:
  '''
  Carga los datos de un archivo xlsx.

  Parámetros:
      data (str): Ruta del archivo xlsx.
  '''
  df = pd.read_excel(data, 'Processed')
  return df