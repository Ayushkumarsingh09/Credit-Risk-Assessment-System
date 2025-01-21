import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model_training import train_model, evaluate_model

class TestModelTraining(unittest.TestCase):
    def test_train_model(self):
        X = pd.DataFrame({
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "feature2": [1, 0, 1, 0]
        })
        y = [0, 1, 0, 1]
        model = train_model(X, y)
        self.assertIsNotNone(model)

    def test_evaluate_model(self):
        X = pd.DataFrame({
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "feature2": [1, 0, 1, 0]
        })
        y = [0, 1, 0, 1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        model = train_model(X_train, y_train)
        auc = evaluate_model(model, X_test, y_test)
        self.assertGreater(auc, 0.5)

if __name__ == "__main__":
    unittest.main()
