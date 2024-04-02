import unittest
import pandas as pd
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from sklearn.model_selection import train_test_split
import sys
import os


class TestFileReading(unittest.TestCase):

    def setUp(self):
        """Setup method to initialize before each test method."""
        self.data_ingestion = DataIngestion()
        
    def test_file_reading(self):
        """Test that the data ingestion component can correctly read the CSV file."""
        df = self.data_ingestion.read_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "Dataframe shouldn't be empty after reading a CSV file.")
        
if __name__ == '__main__':
    unittest.main()
