import unittest
import pandas as pd
from src.data_ingestion import load_data, clean_data

class TestDataIngestion(unittest.TestCase):
    def test_load_data(self):
        data = load_data("data/inputs/sample_data.csv")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
    
    def test_clean_data(self):
        data = pd.DataFrame({
            "income": [50000, 60000, None, 1000000],
            "age": [25, 30, 35, None]
        })
        cleaned_data = clean_data(data)
        self.assertFalse(cleaned_data.isnull().values.any())
        self.assertTrue((cleaned_data["income"] < 1e6).all())

if __name__ == "__main__":
    unittest.main()
