import os 
import sys
from src.exception import CustomException
import pandas as pd
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation,DataTransformationConfig

# Using dataclass decorator , we can use this to directly define variables in class which generally require a init constructor for defination
# and then we can save the train and test data in a file to access in future
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path :  str=os.path.join('artifacts','test.csv')
    raw_data_path :  str=os.path.join('artifacts','data.csv')

class DataIngestion:
    # Initiate config variables from above class so that to store the data in files 
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
            
    # Data Ingestion function where we read the data and prepare test and train data
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        
        try:
            # Reading the dataset
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the Dataset as Dataframe')
            
            # Creating directories which will contain raw train and test data         
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            # Saving Raw Data in created directories 
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Train Test Split Started after saving raw data in artifacts/')
            
            # Train Test Split
            train_set , test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            # Saving Train and test data in artifacts folder 
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Ingestion of the data is completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Using Exception to raise custom exception 
            raise CustomException(e,sys)
        
if __name__ =='__main__':
    obj = DataIngestion()
    
    train_data,test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)