import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance of the trained model.
    """
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.title("Feature Importance")
    plt.show()

def plot_roc_curve(fpr, tpr):
    """
    Plots the ROC curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
