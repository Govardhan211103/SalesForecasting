import os
import sys
import numpy as np
import pandas as pd 
import dill 
from src.exception import CustomException

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)