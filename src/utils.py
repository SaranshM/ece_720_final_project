import os
import sys
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def random_nan(df, frac=0.1):
    total_cells = np.prod(df.shape)
    num_cells_to_replace = int(total_cells * frac)
    
    rows = np.random.randint(0, df.shape[0], size=num_cells_to_replace)
    cols = np.random.randint(0, df.shape[1], size=num_cells_to_replace)
    
    for row, col in zip(rows, cols):
        df.iat[row, col] = np.nan
    
    return df

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)