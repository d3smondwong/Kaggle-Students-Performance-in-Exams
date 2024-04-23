import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

"""
# to initiate and test the data_ingestion.py: python src/components/data_ingestion.py   
# create 'artifact' folder and the logs     
if __name__=="__main__":
    obj=DataIngestion() #Creates an instance of DataIngestion.
    obj.initiate_data_ingestion() 
"""     

"""
# to initiate and test the data_ingestion.py with data_transformation.py: python src/components/data_ingestion.py   
# create 'artifact' folder and the logs
# create pickle file     
if __name__=="__main__":
    obj=DataIngestion() # Creates an instance of DataIngestion.
    train_data,test_data=obj.initiate_data_ingestion() # Calls initiate_data_ingestion to get training and testing data paths.

    data_transformation=DataTransformation() # Creates an instance of DataTransformation 
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data) # Calls initiate_data_transformation on the data transformation object, passing the training and testing data paths
 """

if __name__=="__main__":
    obj=DataIngestion() # Creates an instance of DataIngestion.
    train_data,test_data=obj.initiate_data_ingestion() # Calls initiate_data_ingestion to get training and testing data paths.

    data_transformation=DataTransformation() # Creates an instance of DataTransformation 
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data) # Calls initiate_data_transformation on the data transformation object, passing the training and testing data paths

    modeltrainer=ModelTrainer() # Creates an instance of ModelTrainer (likely for training a machine learning model).
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr)) # Prints the output of initiate_model_trainer, which might be model evaluation metrics or some other training result.
    
    
 
