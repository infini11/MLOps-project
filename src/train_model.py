import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

sys.path.append('.')
from src.preprocessing import *

def load_train_and_test_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """load the train and test data used for train

    Args:
        train_path (str): train dir
        test_path (str): test dir

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train and test dataframe
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return (df_train, df_test)


def split_data_X_y_train_test(df_train: pd.DataFrame, testsize: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """split data into X, y and train and test

    Args:
        df_train (pd.DataFrame): train data
        testsize (float) : size of test data for validating the model

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: train and test data for model validation
    """

    X = df_train.drop('Weekly_Sales', axis=1)
    y = df_train['Weekly_Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize)

    return (X_train, X_test, y_train, y_test)


def train(models_params: dict, data : list):
    pass

def train_model_io():
    pass


def train_model():
    pass