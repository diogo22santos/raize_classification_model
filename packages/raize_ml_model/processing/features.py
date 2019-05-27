# libs
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

# modules
from packages.raize_ml_model.config import config


def balanced_data_imputer(*, df: pd.DataFrame):
    df = df.copy()

    if df[df.columns].isnull().any().any():
        null_counts = df[df.columns].isnull().any()
        vars_ = {key: value for (key, value) in null_counts.items()
                 if value is True}

    # convert to array
    vars_ = [x for x in vars_.keys()]

    # extract columns with data missing
    df_columns_missing = df[vars_]

    # dependent and independent variables
    X = df.drop(labels=vars_ + [config.TARGET], axis=1)
    y = df[config.TARGET]

    # for handle imbalanced dataset by Oversampling
    ros = RandomOverSampler(random_state=0)

    # fit to data
    X_resampled, y_resampled = ros.fit_sample(X, y)

    # combined data
    df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                    df_columns_missing,
                    pd.DataFrame(y_resampled, columns=['default'])], axis=1)

    return df
