import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculates precision, recall, and F1-score.
    """
    precision = np.sum(y_true & y_pred) / np.sum(y_pred)
    recall = np.sum(y_true & y_pred) / np.sum(y_true)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score
