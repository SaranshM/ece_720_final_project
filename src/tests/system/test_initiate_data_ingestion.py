import unittest
import pandas as pd
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from sklearn.model_selection import train_test_split
import sys
import os


class TestInitiateDataIngestion(unittest.TestCase):

    def setUp(self):
        """Setup method to initialize before each test method."""
        self.data_ingestion = DataIngestion()
    
    def test_initiate_data_ingestion(self):
        """Test whether initiate_data_ingestion correctly processes and returns paths."""
        # Execute the method under test
        train_path, test_path = self.data_ingestion.initiate_data_ingestion()
        
        # Check if the paths are strings
        self.assertIsInstance(train_path, str, "The train path should be a string.")
        self.assertIsInstance(test_path, str, "The test path should be a string.")
        
        # Check if the paths exist
        self.assertTrue(os.path.exists(train_path), "The train dataset path does not exist.")
        self.assertTrue(os.path.exists(test_path), "The test dataset path does not exist.")
        
if __name__ == '__main__':
    unittest.main()
