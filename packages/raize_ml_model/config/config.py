# libs
import pathlib

# modules
from packages import raize_ml_model

PACKAGE_ROOT = pathlib.Path(raize_ml_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'
TARGET = 'default'

# features to consider for model training
FEATURES = ['Anos atividade', 'Região', 'Receitas','Ativos',
            'Montante', 'BR', 'Prazo','# pmts pagas', 'Taxa ind.']


# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = ['# pmts pagas']

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = []

# categorical variables to convert no numerical
STR_VARS_TO_CONVERT_INT = ['Taxa ind.']

# features to convert to numbers
CATEGORICAL_VARS = ['Região', 'Receitas', 'Ativos', 'BR']

# name of machine learning pipeline
pipeline_file_name = 'raize_random_forest_model.pkl'