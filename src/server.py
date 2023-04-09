import json
import os
import sys

import pandas as pd
from api import api
import pdb

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/diabetes', methods=['POST'])
def diabetes():
  print('Se realizó una petición GET a la ruta /diabetes')
  # Convertir JSON a DataFrame
  patient = pd.DataFrame.from_dict(request.get_json(), orient='index').transpose()

  # Transformaciones del JSON
  # Eliminamos las columnas que no nos interesan ('annualCost')
  patient = patient.drop(columns=['annualCost'])

  # Cambiamos los nombres de las columnas
  patient_data_base = patient.rename(columns={'baseHbA1cLevel': 'HBA1C', 'age': 'AGE', 'durationOfDiabetes': 'DURATION', 'hypoRate': 'HYPO_RATE', 'man': 'SEX'})
  patient_data_int = patient.rename(columns={'objHbA1cLevel': 'HBA1C', 'age': 'AGE', 'durationOfDiabetes': 'DURATION', 'hypoRateRR': 'HYPO_RATE', 'man': 'SEX'})

  # Cambiamos los tipos de datos
  patient_data_base['HBA1C'] = patient_data_base['HBA1C'].astype(int)
  patient_data_int['HBA1C'] = patient_data_int['HBA1C'].astype(int)
  patient_data_base['AGE'] = patient_data_base['AGE'].astype(int)
  patient_data_int['AGE'] = patient_data_int['AGE'].astype(int)
  patient_data_base['DURATION'] = patient_data_base['DURATION'].astype(int)
  patient_data_int['DURATION'] = patient_data_int['DURATION'].astype(int)
  patient_data_base['HYPO_RATE'] = patient_data_base['HYPO_RATE'].astype(float)
  patient_data_int['HYPO_RATE'] = patient_data_int['HYPO_RATE'].astype(float)

  # Multiplicamos los valores de HYPO_RATE por el valor de base
  patient_data_int['HYPO_RATE'] = patient_data_int['HYPO_RATE'].values * patient_data_base['HYPO_RATE'].values

  # Cambiar SEX: man = true -> 0, woman = false -> 1
  patient_data_base['SEX'] = patient_data_base['SEX'].replace({'true': 0, 'false': 1})
  patient_data_int['SEX'] = patient_data_int['SEX'].replace({'true': 0, 'false': 1})

  patient_data_base = patient_data_base.drop(columns=['hypoRateRR'])
  patient_data_base = patient_data_base.drop(columns=['objHbA1cLevel'])
  patient_data_int = patient_data_int.drop(columns=['hypoRate'])
  patient_data_int = patient_data_int.drop(columns=['baseHbA1cLevel'])

  # Definir las posibles manifestaciones
  possible_manifestations = ["BGRET", "PRET", "ME", "BLI", "ALB1", "ALB2", "ESRD", "ANGINA", "STROKE", "MI", "HF", "NEU", "LEA"]

  # Crear una columna binaria para cada posible manifestación
  for manifestation in possible_manifestations:
    patient_data_base[manifestation] = patient_data_base["manifestations"].apply(lambda x: 1 if manifestation in x else 0)
    patient_data_int[manifestation] = patient_data_int["manifestations"].apply(lambda x: 1 if manifestation in x else 0)

  # Eliminar la columna original de manifestaciones
  patient_data_base = patient_data_base.drop(columns=["manifestations"])
  patient_data_int = patient_data_int.drop(columns=["manifestations"])

  # Definimos el orden de las columnas
  column_order = [
      "SEX",
      "AGE",
      "DURATION",
      "HYPO_RATE",
      "BGRET",
      "PRET",
      "ME",
      "BLI",
      "ALB1",
      "ALB2",
      "ESRD",
      "ANGINA",
      "STROKE",
      "MI",
      "HF",
      "NEU",
      "LEA",
      "HBA1C"
  ]

  # Reordenamos las columnas
  patient_data_base = patient_data_base.reindex(columns=column_order)
  patient_data_int = patient_data_int.reindex(columns=column_order)

  # Llamada a la api para obtener los datos
  data = api.run(patient_data_base, patient_data_int)

  # Devolvemos los datos
  return data

if __name__ == '__main__':
  app.run()