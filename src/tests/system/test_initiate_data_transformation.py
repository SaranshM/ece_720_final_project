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


class TestInitiateDataTransformation(unittest.TestCase):

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
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove created test files and directory
        os.remove(cls.train_path_v2)
        os.remove(cls.test_path_v2)
        if os.path.exists(cls.preprocessor_path):
            os.remove(cls.preprocessor_path)
        os.rmdir(cls.artifacts_dir)

    def test_initiate_data_transformation(self):
        """Test the initiate_data_transformation function for correct execution."""
        train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(self.train_path_v2, self.test_path_v2)

        self.assertTrue(os.path.exists(preprocessor_path), "Preprocessing object file was not saved.")

        np.testing.assert_array_equal(train_arr[:, -1], self.sample_data['smoker_status'].values, "Training target labels are incorrect.")
        np.testing.assert_array_equal(test_arr[:, -1], self.sample_data['smoker_status'].values, "Testing target labels are incorrect.")

        original_num_features = self.sample_data.drop(columns=['smoker_status']).shape[1]
        self.assertTrue(train_arr.shape[1] > original_num_features + 1, "Train array does not reflect additional features created by feature engineering.")
        self.assertTrue(test_arr.shape[1] > original_num_features + 1, "Test array does not reflect additional features created by feature engineering.")


    
    
if __name__ == '__main__':
    unittest.main()
