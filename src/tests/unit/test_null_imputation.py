import unittest
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.components.null_imputer import NullImputer
from src.components.data_transformation import DataTransformation
from src.components.feature_engineering import FeatureEngineering
from src.components.data_ingestion import DataIngestion
from sklearn.model_selection import train_test_split
import sys
import os
import numpy as np
from src.utils import random_nan


class TestNullImputation(unittest.TestCase):

    def setUp(self):
        """Setup method to initialize before each test method."""
        self.null_imputer = NullImputer()
        self.fe = FeatureEngineering()
        self.data_transformation = DataTransformation()
        self.data_ingestion = DataIngestion()
        self.train_path, self._ = self.data_ingestion.initiate_data_ingestion()

        self.dataset = pd.read_csv(self.train_path).head().drop(["smoker_status"], axis = 1)
        self.data_not_imputed = random_nan(self.dataset)
        self.data_imputed = self.null_imputer.fit_transform(self.dataset)
    
        self.null_imputer.categorical_columns = ['gender', 'oral', 'tartar']
        self.null_imputer.numerical_columns = ['age', 'height', 'weight', 'waist', 'eyesight_left',
       'eyesight_right', 'hearing_left', 'hearing_right', 'systolic',
       'relaxation', 'fasting_blood_sugar', 'cholesterol', 'triglyceride',
       'hdl', 'ldl', 'hemoglobin', 'urine_protein', 'serum_creatinine', 'ast',
       'alt', 'gtp', 'dental_caries']
    
    @classmethod
    def setUpClass(cls):
        """Class setup to run once for all tests."""
        # Define a directory for test artifacts
        cls.data_ingestion = DataIngestion()
        cls.train_path, cls._ = cls.data_ingestion.initiate_data_ingestion()

        cls.artifacts_dir = "test_artifacts"
        if not os.path.exists(cls.artifacts_dir):
            os.makedirs(cls.artifacts_dir)

        # Sample dataset
        cls.sample_data = pd.read_csv(cls.train_path).head()

        # Paths for train and test data files
        cls.train_path_v2 = os.path.join(cls.artifacts_dir, "train.csv")
        cls.test_path_v2 = os.path.join(cls.artifacts_dir, "test.csv")
        cls.sample_data.to_csv(cls.train_path_v2, index=False)
        cls.sample_data.to_csv(cls.test_path_v2, index=False)

        # Set the preprocessor object file path
        cls.preprocessor_path = os.path.join(cls.artifacts_dir, "preprocessor.pkl")
    
    def test_null_imputation(self):
        """Test imputation of numerical and categorical columns and verify correct imputation."""
        expected_medians = self.data_not_imputed[self.null_imputer.numerical_columns].median()
        expected_modes = self.data_not_imputed[self.null_imputer.categorical_columns].mode(dropna=True).iloc[0]

        transformed = self.null_imputer.fit_transform(self.data_not_imputed)

        self.assertFalse(transformed.isnull().values.any(), "DataFrame contains NaN values after imputation.")
        
        for col in self.null_imputer.numerical_columns:
            if self.data_not_imputed[col].isnull().any():
                imputed_values = transformed[col][self.data_not_imputed[col].isnull()]
                for imputed_value in imputed_values:
                    self.assertEqual(imputed_value, expected_medians[col], f"Imputed value for {col} does not match the expected median.")
            
        for col in self.null_imputer.categorical_columns:
            if self.data_not_imputed[col].isnull().any():
                mode_value = expected_modes[col]
                imputed_values = transformed[col][self.data_not_imputed[col].isnull()]
                for imputed_value in imputed_values:
                    self.assertEqual(imputed_value, mode_value, f"Imputed value for {col} does not match the most frequent value.")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove created test files and directory
        os.remove(cls.train_path_v2)
        os.remove(cls.test_path_v2)
        if os.path.exists(cls.preprocessor_path):
            os.remove(cls.preprocessor_path)
        os.rmdir(cls.artifacts_dir)
    
if __name__ == '__main__':
    unittest.main()
