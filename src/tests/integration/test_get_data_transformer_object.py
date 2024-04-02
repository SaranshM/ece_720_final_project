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


class TestGetDataTransformerObject(unittest.TestCase):

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
    
    def test_get_data_transformer_object(self):
        """Test if the get_data_transformer_object method returns correctly configured ColumnTransformer."""
        columns = self.dataset.columns
        transformer = self.data_transformation.get_data_transformer_object(columns)

        self.assertIsInstance(transformer, ColumnTransformer, "The returned object is not an instance of ColumnTransformer.")

        found_df_pipeline = False
        for name, transformer, cols in transformer.transformers:
            if name == 'df_pipeline' and isinstance(transformer, Pipeline):
                found_df_pipeline = True
                self.assertListEqual(list(cols), list(columns), "ColumnTransformer does not target the expected columns.")
                steps = transformer.steps
                self.assertEqual(len(steps), 2, "Pipeline does not contain expected number of steps.")
                self.assertIsInstance(steps[0][1], NullImputer, "First step in pipeline is not NullImputer.")
                self.assertIsInstance(steps[1][1], FeatureEngineering, "Second step in pipeline is not FeatureEngineering.")
                break

        self.assertTrue(found_df_pipeline, "Data pipeline (df_pipeline) not found inside ColumnTransformer.")
    
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
