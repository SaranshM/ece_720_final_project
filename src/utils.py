import os
import sys
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score

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

def evaluate_model(X_train, Y_train, X_test, Y_test, model):
    try:
        report = {}
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        
        report['Accuracy'] = accuracy_score(Y_test, y_test_pred)
        report['Precision'] = precision_score(Y_test, y_test_pred)
        report['Recall'] = recall_score(Y_test, y_test_pred)
        report['F1 Score'] = f1_score(Y_test, y_test_pred)
        report['ROC AUC Score'] = roc_auc_score(Y_test, y_test_pred_proba)
        report['Confusion Matrix'] = confusion_matrix(Y_test, y_test_pred).tolist()  
        
        return report

    except Exception as e:
        raise CustomException(e, sys)