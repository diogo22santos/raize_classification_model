# libs
import pandas as pd
from sklearn.model_selection import train_test_split

# modules
from packages.raize_ml_model import pipeline
from packages.raize_ml_model.config import config
from packages.raize_ml_model.processing.data_management import save_pipeline
from packages.raize_ml_model.processing.features import balanced_data_imputer


def run_training() -> None:
    """Train the model"""

    # import training data
    data = pd.read_csv(filepath_or_buffer=config.TRAINING_DATA_FILE)

    # create target variable
    data['default'] = [1 if row in ['Incobrável', 'Em recuperação'] else 0 for row in data['Estado']]

    # balanced the data
    data = balanced_data_imputer(df=data)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        stratify=data[config.TARGET],
        test_size=0.2,
        random_state=0) # setting the seed for reproducibility

    print('Train Set: ')
    print(y_train.value_counts(), '\n')

    # apply machine learning pipeline
    pipeline.default_pipe.fit(X_train[config.FEATURES],
                              y_train)

    # save pipeline
    save_pipeline(pipeline_to_persist=pipeline.default_pipe)


if __name__ == '__main__':
    run_training()