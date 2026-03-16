import os 
import sys
import numpy as np 
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

# Reusable save_object function used to save preprocessor or model file
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj) 
            
    except Exception as e:
        raise CustomException(e,sys)
# Model Evaluation function used to evalute model reusable
def evaluateModel(X_train,Y_train,X_test,Y_test,models):
    try:
        # Empty report defination
        report = {}
        logging.info('Model Evaluation Started')
        # Taking Each Model from model Dictionary and checking performance
        for i in range(len(list(models))):
            
            model = list(models.values())[i]
            
            # Fitting Models
            model.fit(X_train,Y_train)
            # Training Data Prediction
            Y_train_pred = model.predict(X_train)            
            
            # Test Data Prediction
            Y_test_pred = model.predict(X_test)
            
            # Finding r2 Score for every model and adding in report
            train_model_score = r2_score(Y_train,Y_train_pred)
            test_model_score = r2_score(Y_test,Y_test_pred)
            
            report[list(models.keys())[i]]=test_model_score
        
        logging.info('Model Evaluation Completed')
        return report
        
    except Exception as e:
        raise CustomException(e,sys)