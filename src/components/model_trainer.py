import sys
from src.logger import logging
from src.exception import CustomException
import os 

from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,root_mean_squared_error,mean_absolute_error

from src.utils import save_object,evaluateModel
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path :str=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Initiated model training process...')
            x_train,y_train , x_test , y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            
            # List Of models used 
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "ElasticNet":ElasticNet(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            logging.info('Finding Model report for all models tested')
            # Getting Model report from evaluate model reusable function
            model_report:dict=evaluateModel(X_train=x_train,Y_train=y_train,X_test=x_test,Y_test=y_test,models=models)
            
            # Getting Best Model Score
            best_model_score = max(model_report.values())
            
            # Getting Best Model Name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Getting best model from list 
            best_model = models[best_model_name]
            
            if best_model_score<0.60:
                raise CustomException('No Best Model Found')
            logging.info(f'Best Model for training and test data found {best_model_name} with score {best_model_score}')
            
            predicted_data = best_model.predict(x_test)
            r2 = r2_score(y_test,predicted_data)
            
            # Saving Model in pkl file
            save_object(self.model_trainer_config.trained_model_file_path,best_model)
            logging.info('Successfully saved best model in pkl file')
        
            
            
            return r2
        
        except Exception as e:
            raise CustomException(e,sys)
        
