import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging
from src.components import data_transformation
from src.utils import load_object
class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            logging.info(f"{features}")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
            data_list:list
            ):
        #print(data_list)
        self.Store=(data_list[0]),
        self.Dept=(data_list[1]),
        self.IsHoliday=(data_list[2]),
        self.Temperature =(data_list[3]),
        self.Fuel_Price =(data_list[4]),
        self.CPI =(data_list[5]),
        self.Unemployment =(data_list[6]),
        self.Type=(data_list[7]),
        self.Size =(data_list[8]),
        self.Date=(data_list[9])

    def get_data_as_data_frame(self):
        try:
            
            custom_data_input_dict = {
                'Store':int(self.Store[0]),
                'Dept':int(self.Dept[0]),
                'IsHoliday':bool(self.IsHoliday[0]), 
                'Temperature':float(self.Temperature[0]), 
                'Fuel_Price':float(self.Fuel_Price[0]), 
                'CPI':float(self.CPI[0]),
                'Unemployment':float(self.Unemployment[0]), 
                'Type':str(self.Type[0]), 
                'Size':float(self.Size[0]), 
                'Date':str(self.Date)
            }
            logging.info(f"df:, {pd.DataFrame(custom_data_input_dict,index=[0])}")
            return pd.DataFrame(custom_data_input_dict,index=[0])

        except Exception as e:
            raise CustomException(e,sys)
    
    