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

# Carga los modelos
model_time_to_event, model_incidence, model_left_years, model_quality_of_life, model_severe_hypoglucemic_event, model_cost, model_risk = api.load_models()

@app.route('/diabetes', methods=['POST'])
def diabetes():
  print('Se realizó una petición GET a la ruta /diabetes')
  # Convertir JSON a DataFrame
  patient = pd.DataFrame.from_dict(request.get_json(), orient='index').transpose()

  # Transformaciones del JSON
  data_base, data_int = api.transform_data(patient)

  # Predice el time to event
  time_to_event_base = model_time_to_event.predict(data_base)
  time_to_event_int = model_time_to_event.predict(data_int)

  # Predice la incidencia
  incidence_base = model_incidence.predict(data_base)
  incidence_int = model_incidence.predict(data_int)

  # Predice los años restantes
  left_years_base = model_left_years.predict(data_base)
  left_years_int = model_left_years.predict(data_int)

  # Predice la calidad de vida
  quality_of_life_base = model_quality_of_life.predict(data_base)
  quality_of_life_int = model_quality_of_life.predict(data_int)

  # Predice el evento severo de hipoglucemia
  severe_hypoglucemic_event_base = model_severe_hypoglucemic_event.predict(data_base)
  severe_hypoglucemic_event_int = model_severe_hypoglucemic_event.predict(data_int)

  # Predice el costo
  cost_base = model_cost.predict(data_base)
  cost_int = model_cost.predict(data_int)

  # Predice el riesgo
  risk_base = model_risk.predict(data_base)
  risk_int = model_risk.predict(data_int)

  # Devolvemos los datos
  return api.create_json_file(time_to_event_base, incidence_base, left_years_base, quality_of_life_base, severe_hypoglucemic_event_base, cost_base, risk_base, time_to_event_int, incidence_int, left_years_int, quality_of_life_int, severe_hypoglucemic_event_int, cost_int, risk_int)

if __name__ == '__main__':
  app.run()