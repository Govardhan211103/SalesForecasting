import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=self.load_object(file_path=model_path)
            preprocessor=self.load_object(file_path=preprocessor_path)
            logging.info(f"{features}")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    def load_object(self,file_path):
        try:
            with open(file_path,"rb") as file_obj:
                return dill.load(file_obj)
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
            Store: str,
            Dept:str,
            IsHoliday: str,
            Temperature :float,
            Fuel_Price :float,
            CPI :float,
            Unemployment :float,
            Type: str,
            Size : float,
            year: str,
            week:str
            ):
        self.Store=Store,
        self.Dept=Dept,
        self.IsHoliday=IsHoliday,
        self.Temperature =Temperature,
        self.Fuel_Price =Fuel_Price,
        self.CPI =CPI,
        self.Unemployment =Unemployment,
        self.Type=Type,
        self.Size =Size,
        self.year=year,
        self.week=week

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Store':[self.Store],
                'Dept':[self.Dept],
                'IsHoliday':[self.IsHoliday], 
                'Temperature':[self.Temperature], 
                'Fuel_Price':[self.Fuel_Price], 
                'CPI':[self.CPI],
                'Unemployment':[self.Unemployment], 
                'Type':[self.Type], 
                'Size':[self.Size], 
                'year':[self.year], 
                'week':[self.week]
            }

            return pd.DataFrame(custom_data_input_dict)


        except Exception as e:
            raise CustomException(e,sys)
    
    