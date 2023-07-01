import sys
import os
import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import datetime as dt

## Feature transformation

#Date transformation into week, month, year_probability
class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X):
        return self
    def transform(self,X):
        try:
            #colums=['Date']
            logging.info(f"{type(X)}")
            df=X
            df['year']=df['Date'].astype('datetime64[ns]').dt.year
            df['month']=df['Date'].astype('datetime64[ns]').dt.month
            df['week']=df['Date'].astype('datetime64[ns]').dt.isocalendar().week

            # defining probablity instead of year itself
            d=dict(df['year'].value_counts(normalize=True))
            df['Year_prob']=df['year'].apply(lambda x:round(d[x], 5))
            
            logging.info("DateTransform has been done")

            # dropping the year and Date columns as we have probability of year and month,week
            df.drop(['year','Date'],axis=1,inplace=True)

            return df.to_numpy()
        except Exception as e:
            raise CustomException(e,sys)


# Categorical variables like Store,Size,Dept are turned into probabilities
# CPI into a categorical feature acc to analysation in Model_Training.ipynb
class FeatureTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X):
        return self
    def transform(self,X):
        columns=['CPI','Size','Store','Dept']
        df=X[columns].copy()
        for column in columns:
            if column=='CPI':
                df['CPI_cat']=df['CPI'].apply(lambda x:0 if x<160 else 1)
            if column=='Size':
                d=dict(df['Size'].value_counts(normalize=True))
                df['Size_prob']=df['Size'].apply(lambda x:round(d[x],5))
            if column=='Store':
                d=dict(df['Store'].value_counts(normalize=True))
                df['Store_prob']=df['Store'].apply(lambda x:round(d[x],5))
            if column=='Dept':
                d=dict(df['Dept'].value_counts(normalize=True))
                df['Dept_prob']=df['Dept'].apply(lambda x:round(d[x],5))
        logging.info("FeatureTransformation has been done")

        #dropping the cat columns as probabilities are replaced
        df.drop(columns,axis=1,inplace=True)

        return df.to_numpy()






@dataclass
class DataTransformationConfig:
    preprocessor_path_obj=os.path.join("artifacts",'preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_transformer_object(self):
        """DataTransformation Function for transforming the data"""

        try:
            # feature lists
            date_column = ['Date']
            num_columns = ['Temperature', 'Fuel_Price', 'Unemployment']
            cat_columns = ['Type', 'IsHoliday']
            trans_columns = ['CPI', 'Size', 'Dept', 'Store']

            #Pipeline for features
            date_pipeline=Pipeline([
                ('datetransformer',DateTransformer()),
                ('dateimputer',SimpleImputer(strategy='most_frequent'))
            ])
            num_pipeline=Pipeline([
                ('numimputer',SimpleImputer(strategy='median'))
            ])
            cat_pipeline=Pipeline([
                ('catimputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder(dtype='int'))
            ])
            transform_pipeline=Pipeline([
                ('featuretransformer',FeatureTransformer()),
                ('transformimputer',SimpleImputer(strategy='most_frequent'))
            ])

            logging.info('Pipe lines for numerical,categorical,date and transforming features are created.')
            

            #preprocessor object
            preprocessor=ColumnTransformer([
                ('date_pipeline',date_pipeline,date_column),
                ('numerical_pipeline', num_pipeline, num_columns),
                ('categorical_pipeline', cat_pipeline, cat_columns),
                ('transformer_pipeline',transform_pipeline,trans_columns)
            ])

            logging.info("Pipeline implemented")

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:

            logging.info("Reading test and train data form given path.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading data completed, Obtaining preprocessor object.")
            preprocessing_obj = self.get_transformer_object()

            target_column_name = 'Weekly_Sales'

            # dividing data into input and target features for preprocessing
            train_X_data=train_df.drop(columns=[target_column_name],axis=1)
            train_y_data=train_df[target_column_name]

            test_X_data=test_df.drop(columns=[target_column_name],axis=1)
            test_y_data=test_df[target_column_name]

            logging.info("Applying preprocessor object on train and test dataframes ")

            train_X_data_scaled=preprocessing_obj.fit_transform(train_X_data)
            test_X_data_scaled=preprocessing_obj.fit_transform(test_X_data)

            train_arr=np.c_[train_X_data_scaled, np.array(train_y_data)]
            test_arr=np.c_[test_X_data_scaled, np.array(test_y_data)]

            logging.info("Combined the scaled data and target feature")

            return (
                train_arr,
                test_arr,
                preprocessing_obj,
                self.data_transformation_config.preprocessor_path_obj
            )
           
        except Exception as e:
            raise CustomException(e,sys)

    def save_object(self,file_path,obj):
        try:
            dir_path=os.path.dirname(file_path)

            os.makedirs(dir_path,exist_ok=True)

            with open(file_path, "wb") as file_obj:
                dill.dump(obj,file_obj)
        except Exception as e:
            raise CustomException(e,sys)

