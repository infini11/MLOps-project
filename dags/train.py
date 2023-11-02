import sys
from airflow.decorators import dag, task

sys.path.append('.')
print(sys.path)
from src.preprocessing import data_processing_with_io
from src.train_model import train_model_io

from config.config import (DATA_FOLDER, ML_DATASET_OUTPUT_FOLDER, NAME_FILES, TRAIN_DATA, TEST_DATA, START_DATE,
                         CONCURRENCY, SCHEDULE_INTERVAL, DEFAULT_ARGS, TRACKING_URI, COL_TO_IMPUTE, COLS_IMPUTE_BY_MEAN,
                         MODELS_PARAM)

@dag(default_args=DEFAULT_ARGS,
     start_date=START_DATE,
     schedule_interval=SCHEDULE_INTERVAL,
     catchup=False,
     concurrency=CONCURRENCY)
def training_dag():

    @task
    def preprocessing_data_task(base_path: str,
                    names_files: list,
                    col_to_impute: list,
                    cols_: list,
                    output_path: str) -> None:
        
        data_processing_with_io(base_path=base_path,
                                names_files=names_files,
                                col_to_impute=col_to_impute,
                                cols_=cols_,
                                output_path=output_path)


    @task
    def training_model_task(tracking_uri: str,
                            models_params: dict,
                            train_path : str,
                            test_path: str,
                            output_path: str) -> None:

        train_model_io(tracking_uri=tracking_uri,
                       models_params=models_params,
                       train_path=train_path,
                       test_path=test_path,
                       output_path=output_path)
        


    preprocessing_data_task(base_path=DATA_FOLDER,
                       names_files=NAME_FILES,
                       col_to_impute=COL_TO_IMPUTE,
                       cols_=COLS_IMPUTE_BY_MEAN,
                       output_path=ML_DATASET_OUTPUT_FOLDER) >> \
    training_model_task(tracking_uri=TRACKING_URI,
                        models_params=MODELS_PARAM,
                        train_path=TRAIN_DATA,
                        test_path=TEST_DATA,
        output_path=ML_DATASET_OUTPUT_FOLDER)



train_ml_dag = training_dag()