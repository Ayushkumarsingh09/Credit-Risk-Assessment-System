def calculate_risk_scores(model, X):
    """
    Calculates risk scores using the trained model.
    """
    scores = model.predict_proba(X)[:, 1]  # Probability of positive class
    return scores
