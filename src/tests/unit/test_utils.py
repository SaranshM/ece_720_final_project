import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import numpy as np
from src.utils import save_object, random_nan, load_object

class TestUtils(unittest.TestCase):

    def test_save_object(self):
        with patch("builtins.open", mock_open()) as mocked_file:
            with patch("os.makedirs") as mocked_makedirs:
                obj = {'key': 'value'}
                save_object("path/to/file.pkl", obj)
                mocked_makedirs.assert_called_with("path/to", exist_ok=True)
                mocked_file.assert_called_with("path/to/file.pkl", "wb")


    def test_load_object(self):
        with patch("builtins.open", mock_open(read_data=b"mock")) as mocked_file:
            with patch("pickle.load") as mocked_pickle:
                load_object("path/to/file.pkl")
                mocked_file.assert_called_with("path/to/file.pkl", "rb")
                mocked_pickle.assert_called_once()

if __name__ == '__main__':
    unittest.main()
