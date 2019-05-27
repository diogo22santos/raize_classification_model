import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing imputer"""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X

class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical data missing imputer"""

    def __init__(self, variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        self.imputer_dict_ = {}

        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X

class StringtoFloatConverter(BaseEstimator, TransformerMixin):
    """Converts strings to floats """

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None
            ) -> 'StringtoFloatConverter':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()

        for feature in self.variables:
            X[feature] = [str(row).rstrip("%") for row in X[feature]]
            X[feature] = [row.replace(',', '.') for row in X[feature]]
            X[feature] = [float(row) / 100 if float(row) >= 1 else row for row in X[feature]]
            X[feature] = X[feature].astype('float64')

        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Rare labels data encoder"""

    def __init__(self, tol=0.1, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        self.encoder_dict_ = {}

        for feature in self.variables:
            t = pd.Series(X[feature].value_counts() / np.float(len(X)))
            self.encoder_dict_[feature] = list(t[t >= self.tol].index)
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]),
                                  X[feature], 'Rare')

        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Strigs to numbers categorical encoder"""

    def __init__(self, variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']

        self.encoder_dict_ = {}

        for feature in self.variables:
            t = temp.groupby([feature])['target'].mean().sort_values(
                ascending=True).index
            self.encoder_dict_[feature] = {k:i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        # check if the transformer introduces NaN values
        if X[self.variables].isnull().any().any():
            null_counts = X[self.variables].isnull().any()
            vars_ = {key: value for (key, value) in null_counts.items()
                     if value is True}
            raise ValueError(
                f'Categorical encoder has introduced NaN when '
                f'transforming categorical variables: {vars_.keys()}')

        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X










