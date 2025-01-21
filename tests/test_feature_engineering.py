import unittest
import pandas as pd
from src.feature_engineering import transform_features

class TestFeatureEngineering(unittest.TestCase):
    def test_transform_features(self):
        data = pd.DataFrame({
            "income": [50000, 60000, 70000],
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"]
        })
        numerical_features = ["income", "age"]
        categorical_features = ["gender"]
        
        transformed_data, preprocessor = transform_features(data, numerical_features, categorical_features)
        self.assertGreater(transformed_data.shape[1], 0)

if __name__ == "__main__":
    unittest.main()
