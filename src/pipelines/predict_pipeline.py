import sys
from src.logger import logging 
from src.exception import CustomException

import pandas as pd 
import numpy as np 

from src.utils import loadObject
class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            logging.info('Entered Pipeline for prediction')
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model=loadObject(file_path=model_path)
            preprocessor = loadObject(file_path=preprocessor_path)
            logging.info('Loaded Model and preprocessor')

            # Standardizing Data
            data_scaled = preprocessor.transform(features)
            # Predicting New Data O/P
            preds = model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e,sys)
        
        logging.info('Returned Predictions by model')
        return preds
        
        
# Used for creating our custom data for prediction through model
class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int):
        
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
        
    def get_input_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
            }
            logging.info('Returned Dataframe of new data')
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)