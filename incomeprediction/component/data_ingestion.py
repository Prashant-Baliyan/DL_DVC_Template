from incomeprediction.logger import logging
from incomeprediction.exception import incomepredictionexception
from incomeprediction.entity.config_entity import DataIngestionConfig
from incomeprediction.entity.artifact_entity import DataIngestionArtifact
import os,sys
import numpy as np
from six.moves import urllib
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:  
            raise incomepredictionexception (e,sys) from e

    def download_housing_data(self,) -> str:
        try:
            #extraction remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url
            logging.info(f"Downloading file from :[{download_url}]")
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
            os.makedirs(raw_data_dir,exist_ok=True)
            shutil.copy(download_url, raw_data_dir)
            return raw_data_dir
        except Exception as e:
            raise incomepredictionexception (e,sys) from e

    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            file_name = os.listdir(raw_data_dir)[0]
            incompred_file_path = os.path.join(raw_data_dir,file_name)
            logging.info(f"Reading csv file: [{incompred_file_path}]")
            incompred_data_frame = pd.read_csv(incompred_file_path)

            logging.info(f"Splitting data into train and test")
            X = incompred_data_frame.iloc[:,:-1]
            Y = incompred_data_frame.iloc[:,-1]

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

            y_train_new = pd.DataFrame(y_train)
            x_frame = [X_train,y_train_new]

            strat_train_set = pd.concat(x_frame,axis= 1)

            
            y_test_new = pd.DataFrame(y_test)
            y_frame = [X_test,y_test_new]

            strat_test_set = pd.concat(y_frame,axis= 1)

            #     strat_train_set = incompred_data_frame.loc[train_index].drop(["salary"],axis=1)
            #     strat_test_set = incompred_data_frame.loc[test_index].drop(["salary"],axis=1)
            
            
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                            file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                        file_name)
            if strat_train_set is not None:
                if os.path.exists(self.data_ingestion_config.ingested_train_dir):
                   os.remove(self.data_ingestion_config.ingested_train_dir)
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index=False)

            if strat_test_set is not None:
                if os.path.exists(self.data_ingestion_config.ingested_test_dir):
                   os.remove(self.data_ingestion_config.ingested_test_dir)               
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,index=False)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact   

        except Exception as e:
            raise incomepredictionexception (e,sys) from e
    
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            self.download_housing_data()
            return self.split_data_as_train_test()
        except Exception as e:
            raise incomepredictionexception(e,sys) from e 


    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")