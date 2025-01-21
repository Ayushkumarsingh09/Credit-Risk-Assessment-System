import pandas as pd

def load_data(file_path):
    """
    Loads data from a given file path.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def clean_data(data):
    """
    Cleans the data by handling missing values and outliers.
    """
    data = data.dropna()  # Drop rows with missing values
    # Example: Remove outliers in the 'income' column
    if 'income' in data.columns:
        data = data[data['income'] < data['income'].quantile(0.99)]
    return data
