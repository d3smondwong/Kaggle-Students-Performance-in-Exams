# data_transformation.py objective is to transform the data. Eg. Handling missing values, transform categorical and numerical features
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# A data class is a special type of class that is designed to store data. 
# creates a variable preprocessor_obj_file_path set to the path where a preprocessor object will be saved (artifacts/proprocessor.pkl).
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl") # to set the path where we will load a pickle file

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig() # __init__ method: Initializes the class and creates a data_transformation_config object to hold configuration.

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:          
            
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
           # It creates two pipelines:
           # num_pipeline: Handles numerical data using SimpleImputer for median imputation and StandardScaler for standardization.

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )
           # cat_pipeline: Handles categorical data using SimpleImputer for most frequent value imputation, 
           # OneHotEncoder for one-hot encoding, and StandardScaler for scaling (with_mean=False to avoid centering categorical features).

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Now, we will need to combine the categorical and numerical pipeline together
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    # This method performs the data transformation process on training and testing data.
    # Return single NumPy arrays (train_arr and test_arr) for training and testing, respectively. This is to prepare data for model fitting.     
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object() # It calls get_data_transformer_object to get the preprocessor object. This will be used to transform the data. defined above

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separates features and target from both training and testing data
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # transforms the training and testing features using the preprocessor
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # np.c is to concatenate the arrays along columns. 
            # np.array is to convert the target variables from potentially pandas DataFrames (target_feature_train_df and target_feature_test_df) into NumPy arrays.
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # to save the data transformer object(preprocessing_obj) into the pickle (.pkl) file via file_path
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
            raise CustomException(e,sys)
