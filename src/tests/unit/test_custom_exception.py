import unittest
from unittest.mock import patch, MagicMock
from src.exception import CustomException, error_message_detail  # Adjust the import path as needed
import sys

class MockTraceback:
    def __init__(self):
        self.tb_frame = MagicMock()
        self.tb_lineno = 123
        self.tb_frame.f_code.co_filename = "mock_file.py"

class TestCustomException(unittest.TestCase):

    def test_error_message_detail(self):
        # Creating a mock error to pass to the function
        try:
            raise ValueError("Test error")
        except ValueError as e:
            # Simulating capture of error detail
            _, _, exc_tb = sys.exc_info()

            # Testing the error_message_detail function directly
            result = error_message_detail(e, sys)
            self.assertIn("Test error", result)
            self.assertIn("line number", result)

    @patch('sys.exc_info')
    def test_CustomException_str(self, mock_exc_info):
        # Setup mock to return a tuple containing mock exception type, value, and traceback
        mock_exc_info.return_value = (ValueError, ValueError("Custom error occurred"), MockTraceback())

        try:
            raise CustomException("Custom error occurred", sys)
        except CustomException as ce:
            self.assertIn("Custom error occurred", str(ce))
            self.assertIn("line number", str(ce))
            self.assertIn("mock_file.py", str(ce))

if __name__ == '__main__':
    unittest.main()
