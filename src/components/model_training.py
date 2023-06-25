import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from data_transformation import DataTransformation
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

            rfc=RandomForestRegressor()

            rfc.fit(X_train,y_train)
            logging.info("best found model on traing and test dataests is RandomForestRegressor.")
        
            data_transform=DataTransformation()
            best_model="RandomForestRegressor"
            data_transform.save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted=rfc.predict(X_test)
            train_predicted=rfc.predict(X_train)
            r2_square=r2_score(y_test,predicted)
            train_score=r2_score(y_train,train_predicted)
            return (r2_square, train_score)

        except Exception as e:
            raise CustomException(e,sys)
