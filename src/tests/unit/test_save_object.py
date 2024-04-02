# import unittest
# from unittest.mock import patch, MagicMock
# import os
# import numpy as np
# import pandas as pd
# from src.utils import save_object, random_nan, load_object  # Adjust the import according to your project structure
# import shutil

# class TestSaveObject(unittest.TestCase):
#     def setUp(self):
#         # Create a temporary directory to store files created during tests
#         self.temp_dir = tempfile.mkdtemp()

#     def tearDown(self):
#         # Remove the temporary directory after the test
#         shutil.rmtree(self.temp_dir)

#     @patch("pickle.dump")
#     def test_save_object_success(self, mock_dump):
#         test_obj = {"test": "object"}
#         test_path = os.path.join(self.temp_dir, "test.pkl")
#         save_object(test_path, test_obj)
#         mock_dump.assert_called_once()
#         self.assertTrue(os.path.exists(test_path))

#     def test_save_object_failure(self):
#         # Test with an invalid directory path to trigger an exception
#         test_obj = {"test": "object"}
#         test_path = os.path.join("/invalid/path", "test.pkl")
#         with self.assertRaises(Exception):
#             save_object(test_path, test_obj)

