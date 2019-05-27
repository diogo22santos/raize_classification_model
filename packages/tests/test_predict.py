# libs

# modules
from packages.raize_ml_model.predict import make_prediction
from packages.raize_ml_model.processing.data_management import load_dataset

def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name='train.csv')
    single_test_json = test_data[1:2].to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None # ensures an output from prediction
    assert subject.get('predictions')[0] == 1 # ensures that predict the 2nd row, return array of 1 (default)


def test_make_multiple_predictions():
    # Given
    test_data = load_dataset(file_name='train.csv')
    single_test_json = test_data.to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None