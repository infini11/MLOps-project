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
    
    return (df_feature, df_store, df_sales)


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


def group_by_sales_by_date(df_sales: pd.DataFrame) -> pd.DataFrame:
    """aims to group by date and compute agg using sum

    Args:
        df_sales (pd.DataFrame): sales dataframe

    Returns:
        pd.DataFrame: return aggregated data
    """
    data_sales_date = df_sales.groupby("Date").agg({"Weekly_Sales":"sum"})
    data_sales_date.sort_index(inplace=True)

    return data_sales_date


def merge_feature_and_sales(df_feature: pd.DataFrame, df_sales: pd.DataFrame) -> pd.DataFrame:
    """Will merge feature and sales on indexes

    Args:
        df_feature (pd.DataFrame): features aggregated data
        df_sales (pd.DataFrame): sales aggregated data

    Returns:
        pd.DataFrame: merged dataframe
    """
    df_sales.Weekly_Sales = df_sales.Weekly_Sales/1000000 #convert weekly sales in million
    df_sales.Weekly_Sales = df_sales.Weekly_Sales.apply(int)
    df_sales_features = pd.merge(df_sales, df_feature, left_index=True, right_index=True, how='left')
    df_sales_features["IsHoliday"] = df_sales_features["IsHoliday"].apply(lambda x: True if x == 45.0 else False )

    return df_sales_features


def agg_store_on_temp_fuel_price_holiday(df_store: pd.DataFrame, df_feature: pd.DataFrame, df_sales: pd.DataFrame) -> pd.DataFrame:
    """scall columns (temperature, fuel price) in df_store by mean, (weekly_sales and isholliday by sum)

    Args:
        df_sales (pd.DataFrame) : sales dataframe
        df_store (pd.DataFrame): store dataframe
        df_features (pd.DataFrame): features dataframe

    Returns:
        pd.DataFrame: scalled dataframe
    """
    data_Store = df_feature.groupby("Store").agg(
        {
            "Temperature": "mean", 
            "Fuel_Price": "mean", 
            "IsHoliday": "sum"
        }
    )

    temp_store = df_sales.groupby("Store").agg({"Weekly_Sales":"sum"})
    temp_store.Weekly_Sales = temp_store.Weekly_Sales/1000000
    temp_store.Weekly_Sales = temp_store.Weekly_Sales.apply(int)
    data_Store.set_index(np.arange(0,45),inplace=True)
    df_store["Temperature"] = data_Store.Temperature
    df_store["Fuel_Price"] = data_Store.Fuel_Price
    df_store["Holiday"] = data_Store.IsHoliday
    df_store["Weekly_Sales"] = temp_store.Weekly_Sales

    return df_store

def transform_data(df: pd.DataFrame) -> None:
    pass

def split_data(df: pd.DataFrame) -> None:
    pass

def data_processing_with_io(df: pd.DataFrame) -> None:
    pass

def data_processing(df: pd.DataFrame) -> None:
    pass