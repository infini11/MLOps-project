import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import mlflow

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


def train(tracking_uri: str, models_params: dict, data : list) -> pd.DataFrame:
    """train selectioned model and store result in some dataframe

    Args:
        models_params (dict): dict of model params
        data (list): train and test data

    Returns:
        pd.DataFrame: a dataframe with model and score sorted
    """

    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run():

        X_train, X_test, y_train, y_test = data

        name = []
        score = []
        models = []
        rmse = []

        for model_param in models_params:

            grid_search = GridSearchCV(estimator=model_param['model'],
                                    param_grid=model_param['grid_parameters'],
                                    scoring='rmse',
                                    cv=5,
                                    verbose=5,
                                    n_jobs=-1
                                )
            
            grid_search.fit(X_train, y_train)
            name.append(type(grid_search).__name__)
            score.append(grid_search.score(X_test, y_test))
            
            models.append(grid_search)
            rmse.append(np.sqrt(mean_squared_error(grid_search.predict(X_test), y_test)))

            mlflow.log_param('best_param', grid_search.best_params_)
            mlflow.log_param('rmse', np.sqrt(mean_squared_error(grid_search.predict(X_test), y_test)))
            mlflow.log_param('score', grid_search.score(X_test, y_test))

        df_score = pd.DataFrame(
            list(zip(name, rmse, score, models)),
            columns=['name', 'rmse', 'score', "model"]
        )
        df_score.set_index('name', inplace=True)
        df_score.sort_values(by=['score'], inplace=True)


    return df_score

def train_model_io(tracking_uri: str, models_params: dict, train_path : str, test_path, output_path: str):
    
    df_score, best_model_row = train_model(tracking_uri=tracking_uri,
                           models_params=models_params,
                           train_path=train_path,
                           test_path=test_path
                        )

    df_score.to_csv(os.path.join(output_path, 'df_model_score.csv'), index=False)
    best_model_row.to_csv(os.path.join(output_path, 'best_row_model.csv'), index=False)


def train_model(tracking_uri: str, models_params: dict, train_path : str, test_path):

    df_train, df_test = load_train_and_test_data(
        train_path=train_path,
        test_path=test_path
    )

    X_train, X_test, y_train, y_test = split_data_X_y_train_test(df_train=df_train, testsize=0.2)

    df_score = train(tracking_uri=tracking_uri, models_params=models_params, data=[X_train, X_test, y_train, y_test])

    best_model_row = df_score.iloc[-1]
    # y_true = df_test['Weekly_Sales']
    # df_test.drop(['Weekly_Sales'], axis=1, inplace=True)
    # pred = best_model_row['model'].predict(df_test)

    return df_score, best_model_row

