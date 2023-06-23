import sys
import os
import pandas as pd
import numpy as np


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer


@dataclass

class DataTransformation:
    def __init__(self):
        self.preprocessor_path_obj=os.path.join('artifacts','preprocessor.pkl')

    def get_transformer_object(self):
        """DataTransformation Function for transforming the data"""

        try:
            #features
            numerical_columns=[
                'Store',
                'Dept',
                'Temperature',
                'Fuel_Price',
                'CPI',
                'Unemployment',
                'Size',
                'year', 
                'week']
            categorical_columns=[
                'Type',
                'IsHoliday']

            #Pipeline for features

            numerical_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler())
            ])
            categorical_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info(f'numerical_columns: {numerical_columns}')
            logging.info(f'categorical_columns: {categorical_columns}')
            logging.info("Pipeline implemented")
            #preprocessor object

            preprocessor=ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_columns),
                ('categorical_pipeline', categorical_pipeline, categorical_columns),
            ])
            
            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:

            logging.info("Reading test and train data form given path.")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Reading data completed, Obtaining preprocessor object.")
            preprocessor_obj = self.get_transformer_object()

            target_column = 'Weekly_Sales'

            # dividing data into input and target features for preprocessing
            input_feature_train_df = train_data.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_data[target_column]

            input_feature_test_df = test_data.drop(target_column,axis=1)
            target_feature_test_df = test_data[target_column]

            logging.info("Applying preprocessing object to train and test data.")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.fit_transform(input_feature_test_df)
            
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            pre_obj=self.preprocessor_path_obj
            self.save_object(
                file_path=pre_obj,
                obj=preprocessor_obj
            )

            logging.info("Saved the preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.preprocessor_path_obj
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj=DataTransformation()
    obj.initiate_data_transformation('artifacts/train.csv','artifacts/test.csv')