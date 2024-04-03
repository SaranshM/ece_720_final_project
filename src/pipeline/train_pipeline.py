import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.utils import evaluate_model, load_object
import json

if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    train_data, test_data = data_ingestion_obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_file_path = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    X_train, X_test, Y_train, Y_test = model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_file_path)

    model_path=os.path.join("artifacts","model.pkl")
    model=load_object(file_path=model_path)

    results = evaluate_model(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, model=model)
    logging.info("Model results: %s", json.dumps(results, indent=4))