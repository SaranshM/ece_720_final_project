import unittest
import pandas as pd
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from sklearn.model_selection import train_test_split
import sys
import os


class TestOutlierRemoval(unittest.TestCase):

    def setUp(self):
        """Setup method to initialize before each test method."""
        self.data_ingestion = DataIngestion()
    
    def test_outlier_removal(self):
        """Ensure that the outlier removal function correctly identifies and removes outliers."""
        df = self.data_ingestion.read_data()
        df = self.data_ingestion.rename_columns(df)

        initial_row_count = df.shape[0]
        df_no_outliers = self.data_ingestion.remove_outliers(df)

        self.assertIsInstance(df_no_outliers, pd.DataFrame)
        self.assertTrue(df_no_outliers.shape[0] < initial_row_count, "Outliers were not removed.")
    
    
    
   
if __name__ == '__main__':
    unittest.main()
