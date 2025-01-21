import unittest
import pandas as pd
from src.risk_scoring import calculate_risk_scores
from sklearn.ensemble import RandomForestClassifier

class TestRiskScoring(unittest.TestCase):
    def test_calculate_risk_scores(self):
        X = pd.DataFrame({
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "feature2": [1, 0, 1, 0]
        })
        y = [0, 1, 0, 1]
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        scores = calculate_risk_scores(model, X)
        self.assertEqual(len(scores), len(X))

if __name__ == "__main__":
    unittest.main()
