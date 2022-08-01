from collections import namedtuple
from datetime import datetime
import uuid
import os, sys
import pandas as pd
from incomeprediction.config.configuration import configuration
from incomeprediction.logger import logging
from incomeprediction.exception import incomepredictionexception
from incomeprediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from incomeprediction.component.data_ingestion import DataIngestion
from incomeprediction.component.data_validation import DataValidation
from incomeprediction.constant import EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME

Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp", "artifact_time_stamp",
                                       "running_status", "start_time", "stop_time", "execution_time", "message",
                                       "experiment_file_path", "accuracy", "is_model_accepted"])




class Pipeline():
    experiment: Experiment = Experiment(*([None] * 11))
    experiment_file_path = None

    def __init__(self, config: configuration ) -> None:
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            Pipeline.experiment_file_path=os.path.join(config.training_pipeline_config.artifact_dir)
            super().__init__(daemon=False, name="pipeline")
            self.config = config
        except Exception as e:
            raise incomepredictionexception(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise incomepredictionexception(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact,)->DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact
                                             )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise incomepredictionexception (e,sys) from e