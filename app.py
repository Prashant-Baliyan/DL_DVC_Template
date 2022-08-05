from incomeprediction.pipeline.pipeline import Pipeline
from incomeprediction.exception import incomepredictionexception
from incomeprediction.logger import logging
from incomeprediction.config.configuration import configuration
from incomeprediction.component.data_validation import DataValidation
from incomeprediction.component.data_ingestion import DataIngestion
from incomeprediction.entity.artifact_entity import DataIngestionArtifact
from incomeprediction.entity.config_entity import DataIngestionConfig
import os
def main():
    try:

        #pipeline = Pipeline()
        #pipeline.run_pipeline()
        data_ingestion_config = configuration().get_data_ingestion_config()
        print(data_ingestion_config)
        # schema_file_path=r"D:\Project\machine_learning_project\config\schema.yaml"
        # file_path=r"D:\Project\machine_learning_project\housing\artifact\data_ingestion\2022-06-27-19-13-17\ingested_data\train\housing.csv"
        dataing = DataIngestion(data_ingestion_config=data_ingestion_config)
        dataing.initiate_data_ingestion()
    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__=="__main__":
    main()