''' Guarda los resultados. '''

import os
import joblib
import pandas as pd
import settings as st

def save_results(df: pd.DataFrame, path: str) -> None:
  '''
  Guarda los resultados en un archivo xlsx.

  Parameters:
      df (str): DataFrame con los resultados.
  '''
  result_location = os.path.join(st.DATA_PREDICTIONS, path) + '.xlsx'
  with open(result_location, "wb") as f:
    df.to_excel(f)



def save_model(model, path) -> None:
  '''
  Guarda el modelo en un archivo pickle.

  Parameters:
      model (str): Modelo.
  '''
  model_location = os.path.join(st.MODEL_DIR, path) + '.pkl'
  with open(model_location, "wb") as f:
    joblib.dump(model, f)