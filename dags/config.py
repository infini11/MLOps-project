import os
import sys
from datetime import datetime as dt
from sklearn.ensemble import RandomForestClassifier
import datetime
import xgboost as xgb
import numpy as np

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')

ML_DATASET_OUTPUT_FOLDER = "/opt/airflow/output"
AIRFLOW_PREFIX_TO_DATA = '/opt/airflow/data/'
NAME_FILES = ['Features data set.csv', 'stores data-set.csv', 'sales data-set.csv']
MLRUNS_DIR = '/mlruns'

COL_TO_IMPUTE = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
COLS_IMPUTE_BY_MEAN = ["CPI", "Unemployment"]

TRAIN_DATA = os.path.join(AIRFLOW_PREFIX_TO_DATA, "train/train.csv")
TEST_DATA = os.path.join(AIRFLOW_PREFIX_TO_DATA , "test/test.csv")

START_DATE = dt(2023, 10, 30)
CONCURRENCY = 4
SCHEDULE_INTERVAL = datetime.timedelta(hours=6)
DEFAULT_ARGS = {'owner': 'airflow'}

TRACKING_URI = 'http://mlflow:5000'

MODELS_PARAM = {
    'xgboost': {
        'model': xgb.XGBClassifier(),
        'grid_parameters': {
            'nthread':[4],
            'learning_rate': [0.1, 0.01, 0.05],
            'max_depth': np.arange(3, 5, 2),
            'scale_pos_weight':[1],
            'n_estimators': np.arange(5, 10, 2),
            'missing':[-999]
        }
    },
    'random_forest':  {
        'model': RandomForestClassifier(),
        'grid_parameters': {
            'min_samples_leaf': np.arange(1, 5, 1),
            'max_depth': np.arange(1, 7, 1),
            'max_features': ['auto'],
            'n_estimators': np.arange(10, 20, 2)}
    }
}