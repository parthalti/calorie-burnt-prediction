"""
Confusion Matrix Analysis for Regression via Calorie Band Discretization.
Syllabus trick: Discretizes continuous actual and predicted calories into
operational intensity tiers (Low, Medium, High) to evaluate confusion matrices,
accuracy, precision, recall, and classification reports.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

BAND_LABELS = ["Low (<50 kcal)", "Medium (50-150 kcal)", "High (>150 kcal)"]


def calories_to_band(series):
    """
    Discretizes continuous calories into 3 operational bands:
    - Low: < 50 kcal
    - Medium: 50 to 150 kcal
    - High: > 150 kcal
    """
    conditions = [
        series < 50.0,
        (series >= 50.0) & (series <= 150.0),
        series > 150.0
    ]
    return pd.Series(np.select(conditions, BAND_LABELS, default="Medium (50-150 kcal)"), index=series.index)


def analyze_discretized_confusion_matrices():
    """
    Loads model test predictions, discretizes them into calorie bands,
    and computes confusion matrices and classification reports for each model.
    """
    preds_path = os.path.join(MODELS_DIR, "test_predictions.csv")
    if not os.path.exists(preds_path):
        from src.models_training import train_and_benchmark_models
        train_and_benchmark_models()

    df_preds = pd.read_csv(preds_path)
    actual_bands = calories_to_band(df_preds["Actual"])

    model_cols = [c for c in df_preds.columns if c != "Actual"]
    band_analysis = {
        "band_definitions": {
            "Low": "< 50 kcal (Light warmup/recovery)",
            "Medium": "50 - 150 kcal (Moderate cardio)",
            "High": "> 150 kcal (High intensity athletic effort)"
        },
        "actual_band_distribution": actual_bands.value_counts().to_dict(),
        "models": {}
    }

    for model_name in model_cols:
        pred_bands = calories_to_band(df_preds[model_name])
        cm = confusion_matrix(actual_bands, pred_bands, labels=BAND_LABELS)
        acc = accuracy_score(actual_bands, pred_bands)
        report = classification_report(actual_bands, pred_bands, labels=BAND_LABELS, output_dict=True, zero_division=0)

        band_analysis["models"][model_name] = {
            "confusion_matrix": cm.tolist(),
            "band_labels": BAND_LABELS,
            "band_accuracy": float(round(acc * 100, 2)),
            "macro_precision": float(round(report["macro avg"]["precision"] * 100, 2)),
            "macro_recall": float(round(report["macro avg"]["recall"] * 100, 2)),
            "macro_f1": float(round(report["macro avg"]["f1-score"] * 100, 2)),
            "per_class_metrics": {
                label: {
                    "precision": float(round(report[label]["precision"] * 100, 2)),
                    "recall": float(round(report[label]["recall"] * 100, 2)),
                    "f1_score": float(round(report[label]["f1-score"] * 100, 2)),
                    "support": int(report[label]["support"])
                }
                for label in BAND_LABELS
            }
        }

        print(f"[BAND CONFUSION] {model_name}: Band Accuracy = {acc*100:.2f}% | Macro F1 = {report['macro avg']['f1-score']*100:.2f}%")

    with open(os.path.join(MODELS_DIR, "confusion_bands_summary.json"), "w") as f:
        json.dump(band_analysis, f, indent=2)

    print(f"\n[SUMMARY SAVED] Discretized confusion matrices saved to {os.path.join(MODELS_DIR, 'confusion_bands_summary.json')}")
    return band_analysis


if __name__ == "__main__":
    analyze_discretized_confusion_matrices()
