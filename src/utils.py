import os
import sys
import numpy as np
import pandas as pd 
import dill 
from src.components import data_transformation
from src.exception import CustomException
from sklearn.metrics import r2_score

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
        return obj
    except Exception as e:
        raise CustomException(e,sys)