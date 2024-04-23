# data_ingestion.py objective is to load the dataset and split into tran - test split
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# @dataclass is a decorator. with @dataclass, you can directly define class variable without using the constructor def __init__(self). This is to define where to save 
# store the files in the 'artifacts' folder
# Defines a configuration class (DataIngestionConfig) to store file paths for training, testing, and raw data.
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

# The DataIngestion class handles loading data and splitting into train, test split.
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() # to read the 3 .csv files

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('data\StudentsPerformance.csv') # Change this for mongoDB / SQL
            """
            # Path to database file
            database_path = "data/noshow.db"

            # Connect to the database
            connection = sqlite3.connect(database_path)

            # Define the SQL query to read data from a table (replace 'your_table_name' with the actual table name)
            query = "SELECT * FROM noshow"
            df = pd.read_sql_query(query, connection)
            """
                       
            logging.info('Read the dataset as dataframe')
            
            # rename the columns
            df = df.rename(columns={'race/ethnicity': 'race_ethnicity',
                        'parental level of education': 'parental_level_of_education',
                        'test preparation course': 'test_preparation_course',
                        'math score': 'math_score',
                        'reading score': 'reading_score',
                        'writing score': 'writing_score'})
            
            logging.info('Rename the columns to be more suitable for analysis')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # Creates the directory structure for the output files if it doesn't exist (using os.makedirs). = artifacts

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) # Saves the loaded dataframe as a CSV file to the raw_data_path.

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # Saves the training and testing sets as separate CSV files to their respective paths.

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return( # return the data points back to the data path. This allows retrieval in the next step for Data transformation
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
        """
        finally:
            # Close the database connection (ensure it's closed even if errors occur)
            connection.close()
        """
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
 
 
