"""
Master Pipeline Orchestrator for Calories Burnt Prediction ML Project.
Executes the full pipeline sequentially:
1. Data Ingestion, Merge Validation & Cleaning
2. Multicollinearity (VIF) & Leak-Free Train-Test Splitting
3. Model Training & Cross-Validation (Linear Regression, Decision Tree, Random Forest, KNN, XGBoost)
4. Unsupervised Profiling (PCA + K-Means)
5. Concept Learning (Find-S)
6. Confusion Matrix Trick (Calorie Bands Discretization)
"""

import sys
import os
import time

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_pipeline import prepare_train_test_data
from src.models_training import train_and_benchmark_models
from src.clustering import build_workout_profiles
from src.find_s import run_find_s_algorithm
from src.confusion_analysis import analyze_discretized_confusion_matrices


def run_full_pipeline():
    start_time = time.time()
    print("=" * 75)
    print("   CALORIES BURNT PREDICTION: MACHINE LEARNING PIPELINE EXECUTION")
    print("=" * 75)

    print("\n>>> STEP 1: Data Pipeline, Merging & Leak-Free Splitting")
    data = prepare_train_test_data()

    print("\n>>> STEP 2: Model Training, Hyperparameter Tuning & 5-Fold CV")
    models_summary = train_and_benchmark_models()

    print("\n>>> STEP 3: Unsupervised Learning (PCA + K-Means Workout Profiling)")
    pca_artifacts = build_workout_profiles()

    print("\n>>> STEP 4: Concept Learning (Find-S Algorithm on Discretized Rules)")
    find_s_output = run_find_s_algorithm()

    print("\n>>> STEP 5: Syllabus Trick: Discretized Calorie Band Confusion Matrix")
    confusion_output = analyze_discretized_confusion_matrices()

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 75)
    print(f"   PIPELINE COMPLETED SUCCESSFULLY IN {elapsed} SECONDS!")
    print("=" * 75)

    # Print Leaderboard Table
    print("\n--- MODEL LEADERBOARD (Test Set Evaluation) ---")
    print(f"{'Model Name':<22} | {'MAE (kcal)':<10} | {'RMSE (kcal)':<12} | {'R2 Score':<10} | {'5-Fold CV R2':<15}")
    print("-" * 75)
    for model_name, m in models_summary["models"].items():
        cv_str = f"{m['CV_R2_Mean']} +/- {m['CV_R2_Std']}"
        print(f"{model_name:<22} | {m['MAE']:<10.4f} | {m['RMSE']:<12.4f} | {m['R2']:<10.4f} | {cv_str:<15}")

    print("\n--- BAND CLASSIFICATION ACCURACY (The Confusion Matrix Trick) ---")
    for model_name, info in confusion_output["models"].items():
        print(f"  * {model_name:<20}: Band Accuracy = {info['band_accuracy']:.2f}% | Macro F1 = {info['macro_f1']:.2f}%")

    print("\n--- LEARNED FIND-S HYPOTHESIS ---")
    print(f"  Concept: High Calorie Burn (>= 150 kcal)")
    print(f"  Vector : < {', '.join(find_s_output['final_hypothesis'])} >")
    print(f"  Features: {', '.join(find_s_output['feature_names'])}")

    print("\n--- PCA + K-MEANS WORKOUT CLUSTERS ---")
    for name, stats in pca_artifacts["profile_stats"].items():
        print(f"  * {name}: {stats['Count']} sessions ({stats['Percentage']}%) | Avg Duration: {stats['Avg_Duration_Min']}m | Avg HR: {stats['Avg_Heart_Rate_BPM']}bpm -> {stats['Avg_Calories_Kcal']} kcal")

    print("\nAll model binaries, scaler artifacts, and reports are saved in:")
    print(f"  --> {os.path.join(BASE_DIR, 'models')}")
    print("\nReady to launch Streamlit UI via: streamlit run app.py\n")


if __name__ == "__main__":
    run_full_pipeline()
