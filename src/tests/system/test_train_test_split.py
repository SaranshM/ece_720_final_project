import unittest
import pandas as pd
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from sklearn.model_selection import train_test_split
import sys
import os


class TestTrainTestSplit(unittest.TestCase):

    def setUp(self):
        """Setup method to initialize before each test method."""
        self.data_ingestion = DataIngestion()
    
    def test_train_test_split(self):
        """Validate that the train-test split correctly partitions the data into expected proportions."""
        df = self.data_ingestion.read_data()
        df = self.data_ingestion.rename_columns(df)
        df = self.data_ingestion.remove_outliers(df)

        train_set, test_set = self.data_ingestion.do_train_test_split(df)

        self.assertEqual(train_set.shape[0] + test_set.shape[0], df.shape[0], "Train-test split did not partition data correctly.")
        self.assertTrue(train_set.shape[0] > test_set.shape[0], "Train set should be larger than test set.")
    
    
if __name__ == '__main__':
    unittest.main()
