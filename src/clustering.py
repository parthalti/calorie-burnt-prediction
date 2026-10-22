"""
Unsupervised Workout Profiling via PCA + K-Means.
Syllabus topic: PCA Dimensionality Reduction + K-Means Clustering.
Groups workout sessions into distinct physiological effort profiles:
1. 'Light Warmup / Low Intensity'
2. 'Moderate Cardio / Aerobic Endurance'
3. 'High Intensity / Peak Effort'
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def build_workout_profiles():
    cleaned_path = os.path.join(DATA_DIR, "cleaned_calories.csv")
    if not os.path.exists(cleaned_path):
        from src.data_pipeline import prepare_train_test_data
        prepare_train_test_data()

    df = pd.read_csv(cleaned_path)
    cluster_features = ["Duration", "Heart_Rate", "Body_Temp", "Calories"]
    X_cluster = df[cluster_features]

    # Standardize features for distance-based clustering & PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # 1. PCA: Reduce to 2 Principal Components
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_

    # 2. K-Means: Cluster into 3 intensity profiles
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df["Cluster"] = cluster_labels
    df["PC1"] = pca_coords[:, 0]
    df["PC2"] = pca_coords[:, 1]

    # Sort clusters by average calories to give meaningful, consistent labels
    cluster_calorie_means = df.groupby("Cluster")["Calories"].mean().sort_values()
    relabel_map = {old_label: new_rank for new_rank, old_label in enumerate(cluster_calorie_means.index)}
    df["Cluster"] = df["Cluster"].map(relabel_map)

    # Define persona profiles
    persona_names = {
        0: "Light Warmup / Low Intensity",
        1: "Moderate Cardio / Aerobic Endurance",
        2: "High Intensity / Peak Effort"
    }
    df["Cluster_Name"] = df["Cluster"].map(persona_names)

    # Compute persona statistics
    profile_stats = {}
    for c_id in sorted(persona_names.keys()):
        subset = df[df["Cluster"] == c_id]
        profile_stats[persona_names[c_id]] = {
            "Count": int(len(subset)),
            "Percentage": float(round((len(subset) / len(df)) * 100, 2)),
            "Avg_Duration_Min": float(round(subset["Duration"].mean(), 1)),
            "Avg_Heart_Rate_BPM": float(round(subset["Heart_Rate"].mean(), 1)),
            "Avg_Body_Temp_C": float(round(subset["Body_Temp"].mean(), 1)),
            "Avg_Calories_Kcal": float(round(subset["Calories"].mean(), 1))
        }

    # Save artifact
    pca_artifacts = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "cluster_features": cluster_features,
        "persona_names": persona_names,
        "var_explained": [float(round(v, 4)) for v in var_explained],
        "profile_stats": profile_stats
    }

    joblib.dump(pca_artifacts, os.path.join(MODELS_DIR, "pca_kmeans.joblib"))

    # Save a lightweight sample for UI rendering (500 points for snappy scatter plots)
    sample_df = df.sample(n=min(800, len(df)), random_state=42)[
        ["User_ID", "Duration", "Heart_Rate", "Body_Temp", "Calories", "PC1", "PC2", "Cluster", "Cluster_Name"]
    ]
    sample_df.to_csv(os.path.join(MODELS_DIR, "pca_clusters_sample.csv"), index=False)

    print(f"[PCA + K-MEANS SUCCESS] Explained Variance: PC1={var_explained[0]*100:.1f}%, PC2={var_explained[1]*100:.1f}%")
    for name, stats in profile_stats.items():
        print(f"  * {name}: Avg {stats['Avg_Duration_Min']} min, {stats['Avg_Heart_Rate_BPM']} bpm -> {stats['Avg_Calories_Kcal']} kcal ({stats['Percentage']}%)")

    return pca_artifacts


def project_user_profile(user_dict):
    """
    Given a user's workout details (Duration, Heart_Rate, Body_Temp, Calories),
    projects the point onto the fitted 2D PCA space and assigns the nearest workout cluster.
    """
    artifact_path = os.path.join(MODELS_DIR, "pca_kmeans.joblib")
    if not os.path.exists(artifact_path):
        build_workout_profiles()

    artifacts = joblib.load(artifact_path)
    scaler = artifacts["scaler"]
    pca = artifacts["pca"]
    kmeans = artifacts["kmeans"]
    persona_names = artifacts["persona_names"]

    vals = np.array([[
        user_dict["Duration"],
        user_dict["Heart_Rate"],
        user_dict["Body_Temp"],
        user_dict["Calories"]
    ]])

    vals_scaled = scaler.transform(vals)
    pca_coords = pca.transform(vals_scaled)[0]
    cluster_id = int(kmeans.predict(vals_scaled)[0])

    return {
        "PC1": float(round(pca_coords[0], 3)),
        "PC2": float(round(pca_coords[1], 3)),
        "Cluster_ID": cluster_id,
        "Cluster_Name": persona_names.get(cluster_id, "Unknown Profile")
    }


if __name__ == "__main__":
    build_workout_profiles()
