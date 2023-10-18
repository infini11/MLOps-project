import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from fancyimpute import IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.linear_model import ElasticNet, Lasso, RidgeCV,LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb




def load_all_data(names_files: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """aim to load all data

    Args:
        names_files (list): name of data sources

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 3 dataframe that contains different parts of the dataset
    """
    BASE_PATH = '' #constant to be created
    df_feature = pd.read_csv(os.path.join(BASE_PATH, names_files[0]), parse_dates=["Date"])
    df_store = pd.read_csv(os.path.join(BASE_PATH, names_files[1]))
    df_sales = pd.read_csv(os.path.join(BASE_PATH, names_files[2]), parse_dates=["Date"])
    
    return (df_feature,  df_store, df_sales)


def group_by_feature_by_date(df_feature: pd.DataFrame) -> pd.DataFrame:
    """aim to group by feature by date and compute agg using mean.

    Args:
        df_feature (pd.DataFrame): feature dataframe

    Returns:
        pd.DataFrame: data aggregated
    """
    data_date = df_feature.groupby("Date").agg({"Temperature":"mean"
                                                ,"Fuel_Price":"mean"
                                                ,"IsHoliday":"sum"
                                                ,"CPI":"mean"
                                                ,"Unemployment":"mean"})
    data_date = data_date.sort_index()
    temp_date_data = data_date[:'2012-12-10']

    return temp_date_data

def data_inputation(df: pd.DataFrame) -> None:
    pass

def transform_data(df: pd.DataFrame) -> None:
    pass

def split_data(df: pd.DataFrame) -> None:
    pass

def data_processing_with_io(df: pd.DataFrame) -> None:
    pass

def data_processing(df: pd.DataFrame) -> None:
    pass