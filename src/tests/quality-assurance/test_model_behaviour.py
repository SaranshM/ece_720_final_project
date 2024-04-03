import unittest
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import os

from src.utils import evaluate_model

class TestModelOverfitting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_ingestion = DataIngestion()
        cls.train_path, cls.test_path = cls.data_ingestion.initiate_data_ingestion()

        cls.data_transformation = DataTransformation()
        cls.train_arr, cls.test_arr, cls.preprocessor_file_path = cls.data_transformation.initiate_data_transformation(cls.train_path, cls.test_path)

        cls.X_train, cls.Y_train = cls.train_arr[:, :-1], cls.train_arr[:, -1]
        cls.X_test, cls.Y_test = cls.test_arr[:, :-1], cls.test_arr[:, -1]

        cls.Y_train = cls.Y_train.astype('int')
        cls.Y_test = cls.Y_test.astype('int')
        
        cls.model_trainer = ModelTrainer()
        cls.model = cls.model_trainer.load_model()
        cls.model.fit(cls.X_train, cls.Y_train)

        cls.model_trainer.model_trainer_config.trained_model_file_path = "temp/model.pkl"
        cls.model_file_path = "temp/model.pkl"
    
    def test_predictive_confidence(self):
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        fraction_of_positives, mean_predicted_value = calibration_curve(self.Y_test, y_prob, n_bins=10)

        calibration_difference = fraction_of_positives - mean_predicted_value
        max_calibration_difference = np.max(np.abs(calibration_difference))

        self.assertLessEqual(max_calibration_difference, 0.2, "Model's predicted probabilities are not well-calibrated.")
    
    def test_adversarial_robustness(self):
        X_test_adv = self.X_test + np.random.normal(0, 0.01, self.X_test.shape)
        y_pred = self.model.predict(X_test_adv)
        accuracy = accuracy_score(self.Y_test, y_pred)
        self.assertTrue(accuracy > 0.8, "Model's performance on adversarial examples is too low.")

    def test_cross_validation_stability(self):
        scores = cross_val_score(self.model, self.X_train, self.Y_train, cv=5)

        self.assertLess(np.std(scores) / np.mean(scores), 0.1, "Model performance varies too much across folds.")
    
    def test_feature_importance(self):
        feature_importances = self.model.feature_importances_
        self.assertTrue(all(feature_importances >= 0), "Negative feature importance found.")
    
    def test_overfitting(self):
        y_pred_train = self.model.predict(self.X_train)
        training_accuracy = accuracy_score(self.Y_train, y_pred_train)
        
        y_pred_test = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.Y_test, y_pred_test)
        
        self.assertLessEqual(training_accuracy - test_accuracy, 0.2, "Model may be overfitting. Difference in train and test accuracy is too high.")
    
    def test_model_performance(self):
        report = evaluate_model(X_train=self.X_train, X_test=self.X_test, Y_train=self.Y_train, Y_test=self.Y_test, model=self.model)
        # @todo: Update metrics
        self.assertGreaterEqual(report['Accuracy'], 0.75, "Accuracy is below 80%")
        self.assertGreaterEqual(report['F1 Score'], 0.75, "F1 Score is below 80%")
        self.assertGreaterEqual(report['Precision'], 0.75, "Precision is below 80%")
        self.assertGreaterEqual(report['Recall'], 0.75, "Recall is below 80%")
        self.assertGreaterEqual(report['ROC AUC Score'], 0.75, "ROC AUC Score is below 80%")


    def tearDown(self):
        # Cleanup: Remove the model file after the test to avoid clutter
        if os.path.exists(self.model_file_path):
            os.remove(self.model_file_path)
            os.rmdir("temp")


if __name__ == '__main__':
    unittest.main()
