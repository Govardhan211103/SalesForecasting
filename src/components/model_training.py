import os
import sys

from dataclasses import dataclass

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from data_transformation import DataTransformation
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
#from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training into train and test data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            cbr=CatBoostRegressor(verbose=True)
            logging.info("initiated model training.")
            cbr.fit(X_train,y_train)

            #hyper parameter tuning 
        
            best_estimator=cbr

            data_transform=DataTransformation()
            best_model="RandomForestRegressor"
            data_transform.save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_estimator
            )
            
            logging.info("Predicting test set.")

            test_predicted=best_estimator.predict(X_test)

            train_predicted=best_estimator.predict(X_train)

            test_score=r2_score(y_test,test_predicted)
            train_score=r2_score(y_train,train_predicted)

            mean_error_test=mean_squared_error(y_test,test_predicted)
            mean_error_train=mean_squared_error(y_train,train_predicted)
            logging.info(f"test error: {mean_error_test}, | train error:{mean_error_train}")

            return (train_score, test_score)

        except Exception as e:
            raise CustomException(e,sys)
