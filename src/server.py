import pandas as pd
import time
from api import api

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models
(model_time_to_event,
 model_incidence,
 model_left_years,
 model_quality_of_life,
 model_severe_hypoglucemic_event,
 model_cost,
 model_risk) = api.load_models()


@app.route('/diabetes', methods=['POST'])
def diabetes():
    """Handle POST requests to the '/diabetes' endpoint."""
    print('A POST request was made to the /diabetes endpoint.')
    # Convert JSON to DataFrame
    patient = pd.DataFrame.from_dict(
        request.get_json(), orient='index').transpose()

    # Transform JSON
    data_base, data_int = api.transform_data(patient)

    # Predict time to event
    # Calculate execution time
    start_time = time.time()
    time_to_event_base = model_time_to_event.predict(data_base)
    end_time = time.time()
    time_to_event_int = model_time_to_event.predict(data_int)

    # Predict incidence
    incidence_base = model_incidence.predict(data_base)
    incidence_int = model_incidence.predict(data_int)

    # Predict remaining years
    left_years_base = model_left_years.predict(data_base)
    left_years_int = model_left_years.predict(data_int)

    # Predict quality of life
    quality_of_life_base = model_quality_of_life.predict(data_base)
    quality_of_life_int = model_quality_of_life.predict(data_int)

    # Predict severe hypoglucemic event
    severe_hypoglucemic_event_base = model_severe_hypoglucemic_event.predict(
        data_base)
    severe_hypoglucemic_event_int = model_severe_hypoglucemic_event.predict(
        data_int)

    # Predict cost
    cost_base = model_cost.predict(data_base)
    cost_int = model_cost.predict(data_int)

    # Predict risk
    risk_base = model_risk.predict(data_base)
    risk_int = model_risk.predict(data_int)

    # Add time to log dataframe using append
    api.add_to_log(end_time - start_time)

    # Return data
    return api.create_json_file(time_to_event_base, incidence_base,
                                left_years_base, quality_of_life_base,
                                severe_hypoglucemic_event_base, cost_base,
                                risk_base, time_to_event_int, incidence_int,
                                left_years_int, quality_of_life_int,
                                severe_hypoglucemic_event_int, cost_int,
                                risk_int)


if __name__ == '__main__':
    app.run()
