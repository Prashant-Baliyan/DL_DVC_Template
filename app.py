from incomeprediction.pipeline.pipeline import Pipeline
from incomeprediction.exception import incomepredictionexception
from incomeprediction.logger import logging
from incomeprediction.config.configuration import configuration
from incomeprediction.component.data_validation import DataValidation
from incomeprediction.component.data_ingestion import DataIngestion
from incomeprediction.entity.artifact_entity import DataIngestionArtifact
import os
def main():
    try:

        #pipeline = Pipeline()
        #pipeline.run_pipeline()
        data_ingestion_config = configuration().get_data_ingestion_config()
        print(data_ingestion_config)
           
    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__=="__main__":
    main()