# import unittest
# import numpy as np
# import pandas as pd
# from sklearn.calibration import calibration_curve
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.metrics import accuracy_score
# from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer
# import os

# class TestModelOverfitting(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.data_ingestion = DataIngestion()
#         cls.train_path, cls.test_path = cls.data_ingestion.initiate_data_ingestion()

#         cls.data_transformation = DataTransformation()
#         cls.train_arr, cls.test_arr, cls.preprocessor_file_path = cls.data_transformation.initiate_data_transformation(cls.train_path, cls.test_path)

#         cls.X_train, cls.Y_train = cls.train_arr[:, :-1], cls.train_arr[:, -1]
#         cls.X_test, cls.Y_test = cls.test_arr[:, :-1], cls.test_arr[:, -1]

#         cls.Y_train = cls.Y_train.astype('int')
#         cls.Y_test = cls.Y_test.astype('int')
        
#         cls.model_trainer = ModelTrainer()
#         cls.model = cls.model_trainer.load_model()
#         cls.model.fit(cls.X_train, cls.Y_train)

#         cls.model_trainer.model_trainer_config.trained_model_file_path = "temp/model.pkl"
#         cls.model_file_path = "temp/model.pkl"
    
#     def test_predictive_confidence(self):
#         y_prob = self.model.predict_proba(self.X_test)[:, 1]

#         fraction_of_positives, mean_predicted_value = calibration_curve(self.Y_test, y_prob, n_bins=10)

#         calibration_difference = fraction_of_positives - mean_predicted_value
#         max_calibration_difference = np.max(np.abs(calibration_difference))

#         self.assertLessEqual(max_calibration_difference, 0.2, "Model's predicted probabilities are not well-calibrated.")

#     def tearDown(self):
#         # Cleanup: Remove the model file after the test to avoid clutter
#         if os.path.exists(self.model_file_path):
#             os.remove(self.model_file_path)
#             os.rmdir("temp")


# if __name__ == '__main__':
#     unittest.main()
