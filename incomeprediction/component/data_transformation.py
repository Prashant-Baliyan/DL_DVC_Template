from tkinter import E
from incomeprediction.exception import incomepredictionexception
from incomeprediction.logger import logging
from incomeprediction.entity.config_entity import *
from incomeprediction.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
from incomeprediction.constant import *
from incomeprediction.util.util import *
import os, sys 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler 

# class datatransformation():
#     def __init__(self,data_transformation_config: DataTransformationConfig,
#                  data_ingestion_artifact: DataIngestionArtifact,
#                  data_validation_artifact: DataValidationArtifact
#                  ):
#         try:
#             logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
#             self.data_transformation_config= data_transformation_config
#             self.data_ingestion_artifact = data_ingestion_artifact
#             self.data_validation_artifact = data_validation_artifact
#         except Exception as e:
#             raise incomepredictionexception (e,sys) from e

class datatransformation():
    def __init__(self,data_validation_artifact: DataValidationArtifact,
                data_ingestion_artifact: DataIngestionArtifact,
                data_transformation_config: DataTransformationConfig):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_validation_artifact = data_validation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config= data_transformation_config
            self.schema_file_path = self.data_validation_artifact.schema_file_path
        except Exception as e:
            raise incomepredictionexception (e,sys) from e 

    # def remove_null_values(self):
    #     try:
    #         train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
    #         train_df = train_df.replace('?', np.nan)

    #         null_columns = train_df.loc[:, train_df.isnull().any()].columns.to_list()
    #         if null_columns is not None:
    #             train_df[null_columns] = train_df[null_columns].fillna(train_df.mode().iloc[0])
    #         return train_df
    #     except Exception as e:
    #         raise incomepredictionexception (e,sys) from e

    # def is_correlation_exist(self):
    #     try: 
    #         #train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
    #         schema_file_path = self.data_validation_artifact.schema_file_path
    #         dataset_schema = read_yaml_file(file_path=schema_file_path)
    #         new_df = self.remove_null_values()

    #         for col in new_df.columns:
    #             if new_df[col].dtype == 'object':
    #                 le = LabelEncoder()
    #                 new_df[col] = le.fit_transform(new_df[col].astype(str))
    #             else:
    #                 pass
    #         threshold = dataset_schema[THRESHOLD_KEY]
    #         col_corr = set()
    #         corr_matrix = new_df.corr()
    #         for i in range(len(corr_matrix.columns)):
    #             for j in range(i):
    #                 if abs(corr_matrix.iloc[i,j])> threshold:
    #                     colname = corr_matrix.columns[i]
    #                     col_corr.add(colname)
    #                     new_df.drop(col_corr,axis=1)
    #         return new_df          
    #     except Exception as e:
    #         raise incomepredictionexception (e,sys) from e

    # def feature_scaling(self):
    #     try:
    #         #train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
    #         new_df = self.is_correlation_exist()
    #         scaler = StandardScaler()
    #         scaled_data = scaler.fit_transform(new_df)
    #         scaled_df = pd.DataFrame(scaled_data)
    #         return scaled_df
    #     except Exception as e:
    #         raise incomepredictionexception (e,sys) from e
    
    def handling_imbalance_data(self,train_arr):
        try:
            rs = RandomOverSampler(random_state=30)
            train_file_path = self.data_ingestion_artifact.train_file_path
            schema_file_path = self.data_validation_artifact.schema_file_path
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)           
            schema = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema[TARGET_COLUMN_KEY]
            train_arr = train_arr
            trasnformed_train_df = pd.DataFrame(train_arr, columns = train_df.columns)

            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            x = trasnformed_train_df.iloc[:,:-1]
            y = trasnformed_train_df[target_column_name]
            rs.fit(x,y)
            # rs.fit(input_feature_X_df,input_feature_Y_df)
            X_new,y_new = rs.fit_resample(x,y)
            y_new_df = pd.DataFrame(y_new)
            x_frame = [X_new,y_new_df]
            fianl_train_dataframe = pd.concat(x_frame,axis= 1)
            final_train_arr = np.array(fianl_train_dataframe)
            return final_train_arr
        except Exception as e:
            raise incomepredictionexception(e,sys) from e

    def get_data_transformer_object(self)->ColumnTransformer:
        try:

            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]


            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ]
            )

            cat_pipeline = Pipeline(steps=[
                 ('impute', SimpleImputer(strategy="most_frequent")),
                 #('one_hot_encoder',  OneHotEncoder()),
                 #('scaler', StandardScaler(with_mean=False))
            ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
            ])
            return preprocessing
        except Exception as e:
            raise incomepredictionexception(e,sys) from e

    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            #self.feature_scaling()
            
            schema_file_path = self.data_validation_artifact.schema_file_path
            dataset_schema = read_yaml_file(file_path=schema_file_path)
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]

            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info(f"Obtaining training and test file path.")

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            new_train_df = train_df.replace(' ?', np.nan)

            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)
            new_test_df = test_df.replace(' ?', np.nan)

            schema = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]

            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = new_train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = new_train_df[target_column_name]

            for col in input_feature_train_df[categorical_columns]:
                df_frequency_map = input_feature_train_df[col].value_counts().to_dict()
                input_feature_train_df[col] = input_feature_train_df[col].map(df_frequency_map)

            input_feature_test_df = new_test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = new_test_df[target_column_name]

            for col in input_feature_test_df[categorical_columns]:
                df_frequency_map = input_feature_test_df[col].value_counts().to_dict()
                input_feature_test_df[col] = input_feature_test_df[col].map(df_frequency_map)

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            balance_train_arr = self.handling_imbalance_data(train_arr=train_arr)

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=balance_train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise incomepredictionexception (e,sys) from e

    

