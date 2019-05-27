# libs
import pandas as pd

# modules
from packages.raize_ml_model.config import config
from packages.raize_ml_model.processing.data_management import load_pipeline


# load the pipeline
_raize_default_pipe = load_pipeline(file_name=config.pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline"""

    data = pd.read_json(input_data)
    prediction = _raize_default_pipe.predict(data[config.FEATURES])
    prediction_probability = _raize_default_pipe.predict_proba(data[config.FEATURES])
    response = {'predictions': prediction, 'probabilities': prediction_probability}

    return response


