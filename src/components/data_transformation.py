# Feature Engineering + Data Cleaning
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from src.logger import logging
from src.exception import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.utils import save_object
import os 


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path :str=os.path.join('artifacts','preprocessor.pkl')
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
        
    # This function Creates a column transformer    
    def get_data_transformer_object(self):
        try:

            num_attribs = ['reading_score', 'writing_score']
            complex_cat_attribs = ['race_ethnicity', 'parental_level_of_education']
            simple_cat_attribs = ['gender', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline([
                ('Imputer',SimpleImputer(strategy='median')),
                ('Standardization',StandardScaler())
                ])
            
            logging.info('Numerical Standardization and Imputation Completed')
            complex_cat_pipeline = Pipeline([
                ('Imputer',SimpleImputer(strategy='most_frequent')),
                ('OnehotEncoding',OneHotEncoder())
                ])
            simple_cat_pipeline = Pipeline([
                ('Imputer',SimpleImputer(strategy='most_frequent')),
                ('OrdinalEncoder',OrdinalEncoder())
                ])
    
            logging.info('Categorical Encoding Completed')
            final_pipeline = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_attribs),
                ('cat_pipeline1',complex_cat_pipeline,complex_cat_attribs),
                ('cat_pipeline2',simple_cat_pipeline,simple_cat_attribs)
                ])
            
            return final_pipeline
        except Exception as e:
            raise CustomException(e,sys)
    # Function that transforms data using data transformer
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Read train and test data completed')
            logging.info('Obtaining Preprocessor Object')
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = 'math_score'

            input_features_train=train_df.drop(target_column_name,axis=1)
            output_feature_train = train_df[target_column_name]
            
            input_features_test=test_df.drop(target_column_name,axis=1)
            output_feature_test = test_df[target_column_name]
            
            logging.info('Applying preprocessor object')
            input_features_train=preprocessing_obj.fit_transform(input_features_train)
            input_features_test=preprocessing_obj.transform(input_features_test)
            
            train_arr = np.c_[input_features_train,np.array(output_feature_train)]
            test_arr = np.c_[input_features_test,np.array(output_feature_test)]
            
            logging.info('Data transformed and Preprocessor saved')
            save_object(self.data_transformation_config.preprocessor_obj_file_path,preprocessing_obj)
            
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e,sys)