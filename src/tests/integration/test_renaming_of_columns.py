import unittest
import pandas as pd
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from sklearn.model_selection import train_test_split
import sys
import os


class TestRenamingOfColumns(unittest.TestCase):

    def setUp(self):
        """Setup method to initialize before each test method."""
        self.data_ingestion = DataIngestion()
        
    def test_renaming_of_columns(self):
        """Check if columns are correctly renamed according to the provided mapping."""
        df = self.data_ingestion.read_data()
        expected_columns = [
            "gender","age", "height", "weight", "waist", "eyesight_left", "eyesight_right",
            "hearing_left", "hearing_right", "systolic", "relaxation",
            "fasting_blood_sugar", "cholesterol", "triglyceride", "hdl", "ldl",
            "hemoglobin", "urine_protein", "serum_creatinine", "ast", "alt", "gtp", "oral",
            "dental_caries", "tartar", "smoker_status"
        ]
        df_renamed = self.data_ingestion.rename_columns(df)
        self.assertListEqual(list(df_renamed.columns), expected_columns, "Column names do not match the expected renaming.")
        
if __name__ == '__main__':
    unittest.main()
