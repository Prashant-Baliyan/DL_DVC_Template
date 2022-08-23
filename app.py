from incomeprediction.component.data_transformation import datatransformation
from incomeprediction.pipeline.pipeline import Pipeline
from incomeprediction.exception import incomepredictionexception
from incomeprediction.logger import logging
from incomeprediction.config.configuration import configuration
from incomeprediction.component.data_validation import DataValidation
from incomeprediction.component.data_ingestion import DataIngestion
from incomeprediction.component.data_transformation import datatransformation
from incomeprediction.component.model_trainer import ModelTrainer
from incomeprediction.entity.artifact_entity import DataIngestionArtifact
from incomeprediction.entity.config_entity import DataIngestionConfig
import os
def main():
    try:

        #pipeline = Pipeline()
        #pipeline.run_pipeline()
        data_ingestion_config = configuration().get_data_ingestion_config()
        data_ingestion_aritifact_1 = DataIngestion(data_ingestion_config=data_ingestion_config)
        dataingstnartfct = data_ingestion_aritifact_1.initiate_data_ingestion()
        
        data_validation_config = configuration().get_data_validation_config()
        data_validation_aritifact = DataValidation(data_ingestion_artifact=dataingstnartfct,data_validation_config=data_validation_config)
        datavaldartfct = data_validation_aritifact.initiate_data_validation()

        data_transformation_config = configuration().get_data_transformation_config()
        data_transformation_artifact  = datatransformation(data_validation_artifact= datavaldartfct, data_ingestion_artifact=dataingstnartfct,data_transformation_config= data_transformation_config)
        datatransartifact  = data_transformation_artifact.initiate_data_transformation()

        model_trainer_config = configuration().get_model_trainer_config()
        
        # schema_file_path=r"D:\Project\machine_learning_project\config\schema.yaml"
        # file_path=r"D:\Project\machine_learning_project\housing\artifact\data_ingestion\2022-06-27-19-13-17\ingested_data\train\housing.csv"
        # dataing = DataIngestion(data_ingestion_config=data_ingestion_config)
        # dataing.initiate_data_ingestion()
        #dataval = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=dataingstnartfct)
        #dataval.initiate_data_validation()

        modeltraner= ModelTrainer(model_trainer_config = model_trainer_config, data_transformation_artifact= datatransartifact)
        modeltraner.initiate_model_trainer()
    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__=="__main__":
    main()