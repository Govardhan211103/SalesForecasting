import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.utils import save_object


import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass 

@dataclass# to generate common methods automatically
class DataIngestion:
    def __init__(self):
        self.train_data_path: str=os.path.join("artifacts","train.csv")
        self.test_data_path: str=os.path.join("artifacts","test.csv")
        self.raw_data_path: str=os.path.join("artifacts","data.csv")

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion.")
        try:
            df=pd.read_csv("notebook/data/Sales_data.csv")
            logging.info("Read the dataset as a DataFrame.")

            os.makedirs(os.path.dirname(self.train_data_path),exist_ok=True)
            df.to_csv(self.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated.")
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.train_data_path,index=False,header=True)
            test_set.to_csv(self.test_data_path,index=False,header=True)

            logging.info("DataIngestion of the data completed.")

            return(
                self.train_data_path,
                self.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    transform_object=DataTransformation()
    train_arr,test_arr,preproccessor_obj_path = transform_object.initiate_data_transformation(train_path=train_data_path,test_path=test_data_path)
