import pandas as pd

# Carga los datos de un archivo xlsx
def load_data(data):
    # Carga los datos
    df = pd.read_excel(data, 'Processed')
    return df