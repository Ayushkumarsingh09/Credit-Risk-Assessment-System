from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_model(X_train, y_train):
    """
    Trains a Random Forest model on the training data.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints key metrics.
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    auc = roc_auc_score(y_test, probabilities)
    print(f"ROC AUC Score: {auc:.2f}")
    return auc
