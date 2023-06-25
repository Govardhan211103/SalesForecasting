import os
import sys
import numpy as np
import pandas as pd 
import dill 
from src.exception import CustomException

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

