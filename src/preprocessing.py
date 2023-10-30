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


def load_all_data(basepath: str, names_files: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """aim to load all data

    Args:
        names_files (list): name of data sources
        basepath (str) : base path

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 3 dataframe that contains different parts of the dataset
    """
     #constant to be created
    df_feature = pd.read_csv(os.path.join(basepath, names_files[0]), parse_dates=["Date"])
    df_store = pd.read_csv(os.path.join(basepath, names_files[1]))
    df_sales = pd.read_csv(os.path.join(basepath, names_files[2]), parse_dates=["Date"])
    
    return (df_feature, df_store, df_sales)


def group_by_feature_by_date(df_feature: pd.DataFrame) -> pd.DataFrame:
    """aim to group by feature by date and compute agg using mean.

    Args:
        df_feature (pd.DataFrame): feature dataframe

    Returns:
        pd.DataFrame: data aggregated
    """
    data_date = df_feature.groupby("Date").agg(
        {
            "Temperature":"mean", 
            "Fuel_Price":"mean",
            "IsHoliday":"sum",
            "CPI":"mean",
            "Unemployment":"mean"
        }
    )
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


def agg_store_on_temp_fuel_price_holiday(df_store: pd.DataFrame,
                                         df_feature: pd.DataFrame,
                                         df_sales: pd.DataFrame
                                        ) -> pd.DataFrame:
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

def dataset_construction(df_sales: pd.DataFrame,
                          df_feature: pd.DataFrame,
                          df_store: pd.DataFrame
                        ) -> pd.DataFrame:
    """create dataset and divide the dataset into train and test data

    Args:
        df_sales (pd.DataFrame): sales data
        df_features (pd.DataFrame): features data
        df_store (pd.DataFrame): stores data

    Returns:
        pd.DataFrame : dataset as dataframe
    """

    sales_date_store = df_sales.groupby(["Date","Store"]).agg({"Weekly_Sales":"sum"})
    sales_date_store.sort_index(inplace=True)
    sales_date_store.Weekly_Sales = sales_date_store.Weekly_Sales/10000
    sales_date_store.Weekly_Sales = sales_date_store.Weekly_Sales.apply(int)
    data_table = pd.merge(df_feature, sales_date_store,  how='left', on=["Date", "Store"])
    data_table = pd.merge(data_table, df_store[["Store", "Type"]],  how='left', on=["Store"])

    return data_table


def markdown_data_imputation(data_table: pd.DataFrame, col_to_impute: list) -> pd.DataFrame:
    """impute missing values

    Args:
        data_table (pd.DataFrame): dataset
        col_to_impute (list): list of column to impute

    Returns:
        pd.DataFrame: dataset imputed
    """
    itt = IterativeImputer()
    df = itt.fit_transform(data_table[col_to_impute])
    compte = 0
    for col in col_to_impute:
        data_table[col] = df[:,compte]
        compte = compte + 1
    
    return data_table


def data_imputation_by_mean(data_table: pd.DataFrame, cols: list) -> pd.DataFrame:
    """impute data by mean

    Args:
        data_table (pd.DataFrame): dataset
        cols (list): col to impute by mean

    Returns:
        pd.DataFrame: data imputed by mean
    """
    CPI = cols[0]
    Unemployment = cols[1]
    data_table[CPI].fillna((data_table[CPI].mean()), inplace=True)
    data_table[Unemployment].fillna((data_table[Unemployment].mean()), inplace=True)

    return data_table

def createdummies(data, cols):
    for col in cols:
        one_hot = pd.get_dummies(data[col], prefix=col)
        data = data.join(one_hot)
        data.drop(col, axis = 1, inplace=True)

    return data


def create_columns_and_convert_categorical_data(data_table: pd.DataFrame) -> pd.DataFrame:
    """create columns and convert categorical data

    Args:
        data_table (pd.DataFrame): dataset

    Returns:
        pd.DataFrame: transformed data
    """
    data_table['IsHoliday'] = data_table['IsHoliday'].map({True:0, False:1})
    data_table["Month"] = data_table.Date.dt.month
    data_table["Year"] = data_table.Date.dt.year
    data_table["WeekofYear"] = data_table.Date.dt.weekofyear
    data_table.drop(['Date'], axis=1, inplace=True)

    #create dummies out of categorical column
    data_table = createdummies(data_table, ["Type", "Month", "Year", "WeekofYear"])
    
    return data_table

def data_processing(base_path: str,
                    names_files: list,
                    col_to_impute: list,
                    cols_: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df_feature, df_store, df_sales = load_all_data(base_path, names_files)

    #TBC
    # df_agg_feature_by_date = group_by_feature_by_date(df_feature=df_feature)
    # df_agg_sales_by_date = group_by_sales_by_date(df_sales=df_sales)
    # df_feature_agg_sales_agg = merge_feature_and_sales(
    #     df_feature=df_agg_feature_by_date,
    #     df_sales=df_agg_sales_by_date
    # )

    df_scalled_store = agg_store_on_temp_fuel_price_holiday(
        df_store=df_store,
        df_feature=df_feature,
        df_sales=df_sales
    )

    data_table = dataset_construction(
        df_sales=df_sales,
        df_feature=df_feature,
        df_store=df_scalled_store
    )

    data_table_imputed_markdown = markdown_data_imputation(
        data_table=data_table,
        col_to_impute=col_to_impute
    )

    data_table_compltete_imputed = data_imputation_by_mean(
        data_table=data_table_imputed_markdown,
        cols=cols_
    )
    
    data_table_with_new_features = create_columns_and_convert_categorical_data(
        data_table=data_table_compltete_imputed
    )

    #convert from Fahrenheit to Celcus
    data_table_with_new_features['Temperature'] = (data_table_with_new_features['Temperature']- 32) * 5./9.

    # creating train and test data
    data_train = data_table_with_new_features[data_table_with_new_features.Weekly_Sales.notnull()]
    data_test = data_table_with_new_features[data_table_with_new_features.Weekly_Sales.isnull()]

    return data_table_with_new_features, data_train, data_test


def data_processing_with_io(base_path: str,
                    names_files: list,
                    col_to_impute: list,
                    cols_: list,
                    output_path: str
                    ) -> None:
    
    data_table, data_train, data_test = data_processing(base_path=base_path,
        names_files=names_files,
        col_to_impute=col_to_impute,
        cols_=cols_
    )

    data_table.to_csv(os.path.join(output_path, 'preprocess_dataset'), index=False)
    data_train.to_csv(os.path.join(output_path, 'train/train.csv'), index=False)
    data_test.to_csv(os.path.join(output_path, 'test/test.csv'), index=False)