import os
import sys
from datetime import datetime as dt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet, Lasso, RidgeCV
from sklearn.svm import SVR, LinearSVR, NuSVR
import datetime
import xgboost as xgb
import lightgbm as lgb
import numpy as np

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')

ML_DATASET_OUTPUT_FOLDER = "/opt/airflow/output"
AIRFLOW_PREFIX_TO_DATA = '/opt/airflow/data/'
NAME_FILES = ['Features data set.csv', 'stores data-set.csv', 'sales data-set.csv']
MLRUNS_DIR = '/mlruns'

COL_TO_IMPUTE = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
COLS_IMPUTE_BY_MEAN = ["CPI", "Unemployment"]

TRAIN_DATA = os.path.join(ML_DATASET_OUTPUT_FOLDER, "train/train.csv")
TEST_DATA = os.path.join(ML_DATASET_OUTPUT_FOLDER , "test/test.csv")

START_DATE = dt(2023, 10, 30)
CONCURRENCY = 4
SCHEDULE_INTERVAL = datetime.timedelta(hours=6)
DEFAULT_ARGS = {'owner': 'airflow'}

TRACKING_URI = 'http://mlflow:5000'

MODELS_PARAM = {
    'xgboost': {
        'model': xgb.XGBRegressor(),
        'grid_parameters': {
            'nthread':[4],
            'learning_rate': [0.1, 0.01, 0.05],
            'max_depth': np.arange(3, 5, 2),
            'scale_pos_weight':[1],
            'n_estimators': np.arange(2, 6, 2),
            'missing':[-999]
        }
    },
    'random_forest': {
        'model': RandomForestRegressor(),
        'grid_parameters': {
            'min_samples_leaf': np.arange(1, 5, 1),
            'max_depth': np.arange(1, 7, 1),
            'max_features': ['auto'],
            'n_estimators': np.arange(3, 7, 2)}
    },
    'lightgbm' : {
        'model' : lgb.LGBMRegressor(),
        'grid_parameters': {
            'learning_rate': [0.1, 0.01, 0.05],
            'min_data_in_leaf': np.arange(1, 5, 1),
            "subsample": [0.4, 0.8],
            'n_estimators': np.arange(3, 7, 2)}
    },
    'gradient_boost': {
        'model' : GradientBoostingRegressor(),
        'grid_parameters': {
            "max_depth": [3, 10, 30],
            "max_leaf_nodes": [2, 5, 10, 20],
            "learning_rate": [0.1, 0.01, 0.05]
        }
    },
    'ada_boost': {
        'model' : AdaBoostRegressor(),
        'grid_parameters': {
            'learning_rate': [0.1, 0.01, 0.05],
            'n_estimators': np.arange(3, 7, 2),
            'loss' : ['linear', 'square', 'exponential']
        }
    },
    'elastic_net': {
        'model' : ElasticNet(),
        'grid_parameters': {
            "max_iter": [1, 5, 10],
            "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
            "l1_ratio": np.arange(0.0, 1.0, 0.1)
        }
    },
    'lasso' : {
        'model' : Lasso(),
        'grid_parameters': {
            'alpha': np.logspace(-8, 8, 100)
        }
    },
    # 'ridge_cv': {
    #     'model' : RidgeCV(),
    #     'grid_parameters': {
    #         'alphas': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    #     }
    # },
    # 'svr': {
    #     'model' : SVR(),
    #     'grid_parameters': {
    #         'C': [0.1, 1, 10, 100, 1000],  
    #         'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    #         'kernel': ['rbf']
    #     }
    # },
    # 'linear_svr': {
    #     'model' : LinearSVR(),
    #     'grid_parameters': {
    #         'C': [0.1, 1, 10, 100, 1000],  
    #         'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    #         'kernel': ['rbf']
    #     }
    # },
    # 'nu_svr': {
    #     'model' : NuSVR(),
    #     'grid_parameters': {
    #         'C': [0.1, 1, 10, 100, 1000],  
    #         'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    #         'kernel': ['rbf']
    #     }
    # }
}