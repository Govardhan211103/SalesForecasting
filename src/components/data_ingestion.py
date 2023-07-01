import sys
import os
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomException
from src.logger import logging

from data_transformation import DataTransformation

from model_training import ModelTrainer
#from src.components.model_training import ModelTrainer


import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass 

@dataclass# to generate common methods automatically
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","data.csv")
class DataIngestion:
    def __init__(self):
        self.ingestionconfig=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion.")
        try:
            df=pd.read_csv("notebook/data/Merged_Sales.csv")
            logging.info("Read the dataset as a DataFrame.")

            os.makedirs(os.path.dirname(self.ingestionconfig.train_data_path),exist_ok=True)
            df.to_csv(self.ingestionconfig.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated.")
            train_dataset,test_dataset=train_test_split(df,test_size=0.3,random_state=42)

            train_dataset.to_csv(self.ingestionconfig.train_data_path,index=False,header=True)
            test_dataset.to_csv(self.ingestionconfig.test_data_path,index=False,header=True)

            logging.info("DataIngestion of the data completed.")

            return(
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    transform_object=DataTransformation()
    train_arr, test_arr, preprocessor_obj, preproccessor_obj_path = transform_object.initiate_data_transformation(train_path=train_data_path,test_path=test_data_path)
    #saving the preprcoessor pickle file
    transform_object.save_object(preproccessor_obj_path,preprocessor_obj)
    logging.info("Saved the preprocesor.pkl file")
"""
    modeltrainer=ModelTrainer()
    test_sc,train_sc=modeltrainer.initiate_model_trainer(train_arr,test_arr)
    logging.info(f"Test score: {test_sc}, Train score: {train_sc}")
    print(f"Test score: {test_sc}, Train score: {train_sc}")

"""