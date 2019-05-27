# libs
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# modules
from packages.raize_ml_model.processing import preprocessors as pp
from packages.raize_ml_model.config import config


default_pipe = Pipeline(
    [
        ('categorical_imputer',
         pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA)),
        ('numerical_imputer',
         pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA)),
        ('rare_label_encoder',
         pp.RareLabelCategoricalEncoder(tol=0.1,
                                        variables=config.CATEGORICAL_VARS)),
        ('string_converter',
         pp.StringtoFloatConverter(variables=config.STR_VARS_TO_CONVERT_INT)),
        ('catorical_encoder',
         pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        ('model', RandomForestClassifier(n_estimators = 5,
                                         max_leaf_nodes=10,
                                         random_state=0))
    ]
)