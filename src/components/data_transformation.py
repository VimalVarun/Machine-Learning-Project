import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "Preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function constructs and returns a preprocessing object
        to transform the input data (both numerical and categorical features).
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This method initiates the data transformation process by reading the
        train and test datasets, applying preprocessing, and returning the
        transformed data as numpy arrays.
        """
        try:
            # Read the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Split into features and target variable
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Log shapes of dataframes to ensure correctness
            logging.info(f"Shape of input training features: {input_feature_train_df.shape}")
            logging.info(f"Shape of input test features: {input_feature_test_df.shape}")
            logging.info(f"Shape of training target: {target_feature_train_df.shape}")
            logging.info(f"Shape of test target: {target_feature_test_df.shape}")

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Apply preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Log transformed feature arrays
            logging.info(f"Transformed training features (sample): {input_feature_train_arr[:5]}")
            logging.info(f"Transformed test features (sample): {input_feature_test_arr[:5]}")
            logging.info(f"Training target (sample): {target_feature_train_df[:5]}")

            # Combine features and target variable into a single array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Log final shapes of train/test arrays
            logging.info(f"Final shape of train array: {train_arr.shape}")
            logging.info(f"Final shape of test array: {test_arr.shape}")

            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)