import joblib
import os
import pandas as pd
import json

import settings as st

def load_data():
  # Carga los datos
  path_data = os.path.join(st.API_DATA, 'test_patient1.json')
  data = pd.read_json(path_data)

  # Traspone los datos
  data = data.transpose()

  return data



def load_models():
  # Carga el modelo de time to event
  path_time_to_event = os.path.join(st.API_MODEL_TIME_TO_EVENT, 'Comorbilidades.pkl')
  path_incidence = os.path.join(st.API_MODEL_INCIDENCE, 'Comorbilidades (INC).pkl')
  path_left_years = os.path.join(st.API_MODEL_LY, 'LY.pkl')
  path_quality_of_life = os.path.join(st.API_MODEL_QALY, 'QUALY.pkl')
  path_severe_hypoglucemic_event = os.path.join(st.API_MODEL_SHE, 'SHE.pkl')
  path_cost = os.path.join(st.API_MODEL_COST, 'Cost.pkl')
  path_risk = os.path.join(st.API_MODEL_RISK, 'Comorbilidades (UPTO).pkl')

  # Abre el archivo y carga el modelo
  with open(path_time_to_event, 'rb') as f:
    model_time_to_event = joblib.load(f)

  with open(path_incidence, 'rb') as f:
    model_incidence = joblib.load(f)

  with open(path_left_years, 'rb') as f:
    model_left_years = joblib.load(f)

  with open(path_quality_of_life, 'rb') as f:
    model_quality_of_life = joblib.load(f)

  with open(path_severe_hypoglucemic_event, 'rb') as f:
    model_severe_hypoglucemic_event = joblib.load(f)

  with open(path_cost, 'rb') as f:
    model_cost = joblib.load(f)

  with open(path_risk, 'rb') as f:
    model_risk = joblib.load(f)

  return model_time_to_event, model_incidence, model_left_years, model_quality_of_life, model_severe_hypoglucemic_event, model_cost, model_risk



def predict(data_base, data_int):
  # Carga los modelos
  model_time_to_event, model_incidence, model_left_years, model_quality_of_life, model_severe_hypoglucemic_event, model_cost, model_risk = load_models()

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

  return time_to_event_base, incidence_base, left_years_base, quality_of_life_base, severe_hypoglucemic_event_base, cost_base, risk_base, time_to_event_int, incidence_int, left_years_int, quality_of_life_int, severe_hypoglucemic_event_int, cost_int, risk_int



def create_json_file(time_to_event_base, incidence_base, left_years_base, quality_of_life_base, severe_hypoglucemic_event_base, cost_base, risk_base, time_to_event_int, incidence_int, left_years_int, quality_of_life_int, severe_hypoglucemic_event_int, cost_int, risk_int):
  list_risk_base = []
  list_risk_int = []
  for i in range(0, len(risk_base)):
    new_list_base = []  # Inicializar la nueva lista fuera del bucle j
    new_list_int = []  # Inicializar la nueva lista fuera del bucle j
    for j in range(0, len(risk_base[i]), 3):
      # Crear lista con los valores de 3 en 3
      value_base = risk_base[i][j:j+3][0]
      value_int = risk_int[i][j:j+3][0]
      for k in range(0, 9):
        new_list_base.append(value_base)
        new_list_int.append(value_int)
      # Agregar la lista a la lista principal
      list_risk_base.append(new_list_base)
      list_risk_int.append(new_list_int)
      # Reiniciar la lista
      new_list_base = []
      new_list_int = []

  # Agrupar sublista en listas de 9 elementos
  list_risk_base = [list_risk_base[i:i+9] for i in range(0, len(list_risk_base), 9)]
  list_risk_int = [list_risk_int[i:i+9] for i in range(0, len(list_risk_int), 9)]

  # Fusionar los valores de las sublistas
  list_risk_base = [[elemento for sublista in sublista_grande for elemento in sublista] for sublista_grande in list_risk_base]
  list_risk_int = [[elemento for sublista in sublista_grande for elemento in sublista] for sublista_grande in list_risk_int]
  # Añadir el ultimo elemento de la lista por duplicado
  for sublista in list_risk_base:
    sublista.append(sublista[-1])

  for sublista in list_risk_base:
    sublista.append(sublista[-1])

  # Crea un diccionario con los resultados
  results = {
              "interventions": [
                {
                  "cost": {
                    "avg": cost_base[0][0],
                    "uci95": cost_base[0][2],
                    "lci95": cost_base[0][1],
                    "base": cost_base[0][0]
                  },
                  "name": "DIAB+BASE",
                  "QALY": {
                    "avg": quality_of_life_base[0][0],
                    "uci95": quality_of_life_base[0][2],
                    "lci95": quality_of_life_base[0][1],
                    "base": quality_of_life_base[0][0]
                  },
                  "acute manifestations": [
                    {
                      "number of events": {
                        "avg": severe_hypoglucemic_event_base[0][0],
                        "uci95": severe_hypoglucemic_event_base[0][2],
                        "lci95": severe_hypoglucemic_event_base[0][1],
                        "base": severe_hypoglucemic_event_base[0][0]
                      },
                      "name": "Severe hypoglycemic event"
                    }
                  ],
                  "chronic manifestations": [
                    {
                      "annual risk": [
                      ],
                      "name": "Background Retinopathy",
                      "time to event": {
                        "avg": time_to_event_base[0][18],
                        "uci95": time_to_event_base[0][20],
                        "lci95": time_to_event_base[0][19],
                        "base": time_to_event_base[0][18],
                      },
                      "incidence": {
                        "avg": incidence_base[0][18],
                        "uci95": incidence_base[0][20],
                        "lci95": incidence_base[0][19],
                        "base": incidence_base[0][18]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Proliferative Retinopathy",
                      "time to event": {
                        "avg": time_to_event_base[0][21],
                        "uci95": time_to_event_base[0][23],
                        "lci95": time_to_event_base[0][22],
                        "base": time_to_event_base[0][21]
                      },
                      "incidence": {
                        "avg": incidence_base[0][21],
                        "uci95": incidence_base[0][23],
                        "lci95": incidence_base[0][22],
                        "base": incidence_base[0][21]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Macular edema",
                      "time to event": {
                        "avg": time_to_event_base[0][15],
                        "uci95": time_to_event_base[0][17],
                        "lci95": time_to_event_base[0][16],
                        "base": time_to_event_base[0][15]
                      },
                      "incidence": {
                        "avg": incidence_base[0][15],
                        "uci95": incidence_base[0][17],
                        "lci95": incidence_base[0][16],
                        "base": incidence_base[0][15],
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Blindness",
                      "time to event": {
                        "avg": time_to_event_base[0][12],
                        "uci95": time_to_event_base[0][14],
                        "lci95": time_to_event_base[0][13],
                        "base": time_to_event_base[0][12]
                      },
                      "incidence": {
                        "avg": incidence_base[0][12],
                        "uci95": incidence_base[0][14],
                        "lci95": incidence_base[0][13],
                        "base": incidence_base[0][12]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Microalbuminuria",
                      "time to event": {
                        "avg": time_to_event_base[0][30],
                        "uci95": time_to_event_base[0][32],
                        "lci95": time_to_event_base[0][31],
                        "base": time_to_event_base[0][30]
                      },
                      "incidence": {
                        "avg": incidence_base[0][30],
                        "uci95": incidence_base[0][32],
                        "lci95": incidence_base[0][31],
                        "base": incidence_base[0][30]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Macroalbuminuria",
                      "time to event": {
                        "avg": time_to_event_base[0][33],
                        "uci95": time_to_event_base[0][35],
                        "lci95": time_to_event_base[0][34],
                        "base": time_to_event_base[0][33]
                      },
                      "incidence": {
                        "avg": incidence_base[0][33],
                        "uci95": incidence_base[0][35],
                        "lci95": incidence_base[0][34],
                        "base": incidence_base[0][33]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "End-Stage Renal Disease",
                      "time to event": {
                        "avg": time_to_event_base[0][36],
                        "uci95": time_to_event_base[0][38],
                        "lci95": time_to_event_base[0][37],
                        "base": time_to_event_base[0][36]
                      },
                      "incidence": {
                        "avg": incidence_base[0][36],
                        "uci95": incidence_base[0][38],
                        "lci95": incidence_base[0][37],
                        "base": incidence_base[0][36]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Angina",
                      "time to event": {
                        "avg": time_to_event_base[0][6],
                        "uci95": time_to_event_base[0][8],
                        "lci95": time_to_event_base[0][7],
                        "base": time_to_event_base[0][6]
                      },
                      "incidence": {
                        "avg": incidence_base[0][6],
                        "uci95": incidence_base[0][8],
                        "lci95": incidence_base[0][7],
                        "base": incidence_base[0][6]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Stroke",
                      "time to event": {
                        "avg": time_to_event_base[0][9],
                        "uci95": time_to_event_base[0][11],
                        "lci95": time_to_event_base[0][10],
                        "base": time_to_event_base[0][9]
                      },
                      "incidence": {
                        "avg": incidence_base[0][9],
                        "uci95": incidence_base[0][11],
                        "lci95": incidence_base[0][10],
                        "base": incidence_base[0][9]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Myocardial Infarction",
                      "time to event": {
                        "avg": time_to_event_base[0][3],
                        "uci95": time_to_event_base[0][5],
                        "lci95": time_to_event_base[0][4],
                        "base": time_to_event_base[0][3]
                      },
                      "incidence": {
                        "avg": incidence_base[0][3],
                        "uci95": incidence_base[0][5],
                        "lci95": incidence_base[0][4],
                        "base": incidence_base[0][3]
                      }
                    },
                    {
                      "annual risk": [

                      ],
                      "name": "Heart Failure",
                      "time to event": {
                        "avg": time_to_event_base[0][0],
                        "uci95": time_to_event_base[0][2],
                        "lci95": time_to_event_base[0][1],
                        "base": time_to_event_base[0][0]
                      },
                      "incidence": {
                        "avg": incidence_base[0][0],
                        "uci95": incidence_base[0][2],
                        "lci95": incidence_base[0][1],
                        "base": incidence_base[0][0]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Neuropathy",
                      "time to event": {
                        "avg": time_to_event_base[0][24],
                        "uci95": time_to_event_base[0][26],
                        "lci95": time_to_event_base[0][25],
                        "base": time_to_event_base[0][24]
                      },
                      "incidence": {
                        "avg": incidence_base[0][24],
                        "uci95": incidence_base[0][26],
                        "lci95": incidence_base[0][25],
                        "base": incidence_base[0][24]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Low extremity amputation",
                      "time to event": {
                        "avg": time_to_event_base[0][27],
                        "uci95": time_to_event_base[0][29],
                        "lci95": time_to_event_base[0][28],
                        "base": time_to_event_base[0][27]
                      },
                      "incidence": {
                        "avg": incidence_base[0][27],
                        "uci95": incidence_base[0][29],
                        "lci95": incidence_base[0][28],
                        "base": incidence_base[0][27]
                      }
                    }
                  ],
                  "LY": {
                    "avg": left_years_base[0][0],
                    "uci95": left_years_base[0][2],
                    "lci95": left_years_base[0][1],
                    "base": left_years_base[0][0]
                  }
                },
                {
                  "cost": {
                    "avg": cost_int[0][0],
                    "uci95": cost_int[0][2],
                    "lci95": cost_int[0][1],
                    "base": cost_int[0][0]
                  },
                  "name": "DIAB+INT",
                  "QALY": {
                    "avg": quality_of_life_int[0][0],
                    "uci95": quality_of_life_int[0][2],
                    "lci95": quality_of_life_int[0][1],
                    "base": quality_of_life_int[0][0]
                  },
                  "acute manifestations": [
                    {
                      "number of events": {
                        "avg": severe_hypoglucemic_event_int[0][0],
                        "uci95": severe_hypoglucemic_event_int[0][2],
                        "lci95": severe_hypoglucemic_event_int[0][1],
                        "base": severe_hypoglucemic_event_int[0][0]
                      },
                      "name": "Severe hypoglycemic event"
                    }
                  ],
                  "chronic manifestations": [
                    {
                      "annual risk": [
                      ],
                      "name": "Background Retinopathy",
                      "time to event": {
                        "avg": time_to_event_int[0][18],
                        "uci95": time_to_event_int[0][20],
                        "lci95": time_to_event_int[0][19],
                        "base": time_to_event_int[0][18],
                      },
                      "incidence": {
                        "avg": incidence_int[0][18],
                        "uci95": incidence_int[0][20],
                        "lci95": incidence_int[0][19],
                        "base": incidence_int[0][18]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Proliferative Retinopathy",
                      "time to event": {
                        "avg": time_to_event_int[0][21],
                        "uci95": time_to_event_int[0][23],
                        "lci95": time_to_event_int[0][22],
                        "base": time_to_event_int[0][21]
                      },
                      "incidence": {
                        "avg": incidence_int[0][21],
                        "uci95": incidence_int[0][23],
                        "lci95": incidence_int[0][22],
                        "base": incidence_int[0][21]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Macular edema",
                      "time to event": {
                        "avg": time_to_event_int[0][15],
                        "uci95": time_to_event_int[0][17],
                        "lci95": time_to_event_int[0][16],
                        "base": time_to_event_int[0][15]
                      },
                      "incidence": {
                        "avg": incidence_int[0][15],
                        "uci95": incidence_int[0][17],
                        "lci95": incidence_int[0][16],
                        "base": incidence_int[0][15],
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Blindness",
                      "time to event": {
                        "avg": time_to_event_int[0][12],
                        "uci95": time_to_event_int[0][14],
                        "lci95": time_to_event_int[0][13],
                        "base": time_to_event_int[0][12]
                      },
                      "incidence": {
                        "avg": incidence_int[0][12],
                        "uci95": incidence_int[0][14],
                        "lci95": incidence_int[0][13],
                        "base": incidence_int[0][12]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Microalbuminuria",
                      "time to event": {
                        "avg": time_to_event_int[0][30],
                        "uci95": time_to_event_int[0][32],
                        "lci95": time_to_event_int[0][31],
                        "base": time_to_event_int[0][30]
                      },
                      "incidence": {
                        "avg": incidence_int[0][30],
                        "uci95": incidence_int[0][32],
                        "lci95": incidence_int[0][31],
                        "base": incidence_int[0][30]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Macroalbuminuria",
                      "time to event": {
                        "avg": time_to_event_int[0][33],
                        "uci95": time_to_event_int[0][35],
                        "lci95": time_to_event_int[0][34],
                        "base": time_to_event_int[0][33]
                      },
                      "incidence": {
                        "avg": incidence_int[0][33],
                        "uci95": incidence_int[0][35],
                        "lci95": incidence_int[0][34],
                        "base": incidence_int[0][33]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "End-Stage Renal Disease",
                      "time to event": {
                        "avg": time_to_event_int[0][36],
                        "uci95": time_to_event_int[0][38],
                        "lci95": time_to_event_int[0][37],
                        "base": time_to_event_int[0][36]
                      },
                      "incidence": {
                        "avg": incidence_int[0][36],
                        "uci95": incidence_int[0][38],
                        "lci95": incidence_int[0][37],
                        "base": incidence_int[0][36]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Angina",
                      "time to event": {
                        "avg": time_to_event_int[0][6],
                        "uci95": time_to_event_int[0][8],
                        "lci95": time_to_event_int[0][7],
                        "base": time_to_event_int[0][6]
                      },
                      "incidence": {
                        "avg": incidence_int[0][6],
                        "uci95": incidence_int[0][8],
                        "lci95": incidence_int[0][7],
                        "base": incidence_int[0][6]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Stroke",
                      "time to event": {
                        "avg": time_to_event_int[0][9],
                        "uci95": time_to_event_int[0][11],
                        "lci95": time_to_event_int[0][10],
                        "base": time_to_event_int[0][9]
                      },
                      "incidence": {
                        "avg": incidence_int[0][9],
                        "uci95": incidence_int[0][11],
                        "lci95": incidence_int[0][10],
                        "base": incidence_int[0][9]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Myocardial Infarction",
                      "time to event": {
                        "avg": time_to_event_int[0][3],
                        "uci95": time_to_event_int[0][5],
                        "lci95": time_to_event_int[0][4],
                        "base": time_to_event_int[0][3]
                      },
                      "incidence": {
                        "avg": incidence_int[0][3],
                        "uci95": incidence_int[0][5],
                        "lci95": incidence_int[0][4],
                        "base": incidence_int[0][3]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Heart Failure",
                      "time to event": {
                        "avg": time_to_event_int[0][0],
                        "uci95": time_to_event_int[0][2],
                        "lci95": time_to_event_int[0][1],
                        "base": time_to_event_int[0][0]
                      },
                      "incidence": {
                        "avg": incidence_int[0][0],
                        "uci95": incidence_int[0][2],
                        "lci95": incidence_int[0][1],
                        "base": incidence_int[0][0]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Neuropathy",
                      "time to event": {
                        "avg": time_to_event_int[0][24],
                        "uci95": time_to_event_int[0][26],
                        "lci95": time_to_event_int[0][25],
                        "base": time_to_event_int[0][24]
                      },
                      "incidence": {
                        "avg": incidence_int[0][24],
                        "uci95": incidence_int[0][26],
                        "lci95": incidence_int[0][25],
                        "base": incidence_int[0][24]
                      }
                    },
                    {
                      "annual risk": [
                      ],
                      "name": "Low extremity amputation",
                      "time to event": {
                        "avg": time_to_event_int[0][27],
                        "uci95": time_to_event_int[0][29],
                        "lci95": time_to_event_int[0][28],
                        "base": time_to_event_int[0][27]
                      },
                      "incidence": {
                        "avg": incidence_int[0][27],
                        "uci95": incidence_int[0][29],
                        "lci95": incidence_int[0][28],
                        "base": incidence_int[0][27]
                      }
                    }
                  ],
                  "LY": {
                    "avg": left_years_int[0][0],
                    "uci95": left_years_int[0][2],
                    "lci95": left_years_int[0][1],
                    "base": left_years_int[0][0]
                  }
                }
              ]
            }
  # Add the risk of the base case to the results
  results["interventions"][0]["chronic manifestations"][0]["annual risk"] = list_risk_base[6]
  results["interventions"][0]["chronic manifestations"][1]["annual risk"] = list_risk_base[7]
  results["interventions"][0]["chronic manifestations"][2]["annual risk"] = list_risk_base[5]
  results["interventions"][0]["chronic manifestations"][3]["annual risk"] = list_risk_base[4]
  results["interventions"][0]["chronic manifestations"][4]["annual risk"] = list_risk_base[10]
  results["interventions"][0]["chronic manifestations"][5]["annual risk"] = list_risk_base[11]
  results["interventions"][0]["chronic manifestations"][6]["annual risk"] = list_risk_base[12]
  results["interventions"][0]["chronic manifestations"][7]["annual risk"] = list_risk_base[2]
  results["interventions"][0]["chronic manifestations"][8]["annual risk"] = list_risk_base[3]
  results["interventions"][0]["chronic manifestations"][9]["annual risk"] = list_risk_base[1]
  results["interventions"][0]["chronic manifestations"][10]["annual risk"] = list_risk_base[0]
  results["interventions"][0]["chronic manifestations"][11]["annual risk"] = list_risk_base[8]
  results["interventions"][0]["chronic manifestations"][12]["annual risk"] = list_risk_base[9]

  results["interventions"][1]["chronic manifestations"][0]["annual risk"] = list_risk_int[6]
  results["interventions"][1]["chronic manifestations"][1]["annual risk"] = list_risk_int[7]
  results["interventions"][1]["chronic manifestations"][2]["annual risk"] = list_risk_int[5]
  results["interventions"][1]["chronic manifestations"][3]["annual risk"] = list_risk_int[4]
  results["interventions"][1]["chronic manifestations"][4]["annual risk"] = list_risk_int[10]
  results["interventions"][1]["chronic manifestations"][5]["annual risk"] = list_risk_int[11]
  results["interventions"][1]["chronic manifestations"][6]["annual risk"] = list_risk_int[12]
  results["interventions"][1]["chronic manifestations"][7]["annual risk"] = list_risk_int[2]
  results["interventions"][1]["chronic manifestations"][8]["annual risk"] = list_risk_int[3]
  results["interventions"][1]["chronic manifestations"][9]["annual risk"] = list_risk_int[1]
  results["interventions"][1]["chronic manifestations"][10]["annual risk"] = list_risk_int[0]
  results["interventions"][1]["chronic manifestations"][11]["annual risk"] = list_risk_int[8]
  results["interventions"][1]["chronic manifestations"][12]["annual risk"] = list_risk_int[9]

  # Crea un JSON con los resultados
  path_results = os.path.join(st.API_DATA, 'results.json')
  with open(path_results, 'w') as f:
    json.dump(results, f, indent=2)

  # Devuelve el JSON
  return results



def run(data_base, data_int):
  time_to_event_base, incidence_base, left_years_base, quality_of_life_base, severe_hypoglucemic_event_base, cost_base, risk_base, time_to_event_int, incidence_int, left_years_int, quality_of_life_int, severe_hypoglucemic_event_int, cost_int, risk_int = predict(data_base, data_int)
  return create_json(time_to_event_base, incidence_base, left_years_base, quality_of_life_base, severe_hypoglucemic_event_base, cost_base, risk_base, time_to_event_int, incidence_int, left_years_int, quality_of_life_int, severe_hypoglucemic_event_int, cost_int, risk_int)
