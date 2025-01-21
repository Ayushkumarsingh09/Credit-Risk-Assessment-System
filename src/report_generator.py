def generate_report(auc, feature_importances, output_file="model_summary.txt"):
    """
    Generates a report with key metrics and writes it to a file.
    """
    with open(output_file, "w") as f:
        f.write("Model Performance Summary\n")
        f.write("==========================\n")
        f.write(f"ROC AUC Score: {auc:.2f}\n")
        f.write("\nFeature Importances:\n")
        for feature, importance in feature_importances:
            f.write(f"{feature}: {importance:.4f}\n")
    print(f"Report generated at {output_file}")
