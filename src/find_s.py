"""
Find-S Concept Learning Algorithm Implementation.
Syllabus topic: Concept Learning & Find-S Algorithm (Tom Mitchell ML Curriculum).
Demonstrates learning the concept of 'High Calorie Burn' (Calories >= 150 kcal)
from discretized workout and demographic attributes.
"""

import os
import json
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def discretize_dataset(df):
    """
    Transforms continuous exercise features into categorical bins suitable
    for symbolic Concept Learning.
    """
    df_disc = pd.DataFrame()
    df_disc["Gender"] = df["Gender"].astype(str).str.capitalize()
    df_disc["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 35, 55, 120],
        labels=["Young", "Middle_Aged", "Senior"]
    ).astype(str)
    df_disc["Duration_Level"] = pd.cut(
        df["Duration"],
        bins=[0, 14, 35],
        labels=["Short_Duration", "Long_Duration"]
    ).astype(str)
    df_disc["Heart_Rate_Level"] = pd.cut(
        df["Heart_Rate"],
        bins=[0, 100, 250],
        labels=["Moderate_HR", "High_HR"]
    ).astype(str)
    df_disc["Body_Temp_Level"] = pd.cut(
        df["Body_Temp"],
        bins=[0, 40.0, 50.0],
        labels=["Normal_Temp", "Elevated_Temp"]
    ).astype(str)

    # Concept target: High Calorie Burn (>= 150 kcal)
    df_disc["Target_Concept"] = np.where(df["Calories"] >= 150.0, "Yes", "No")

    return df_disc


def run_find_s_algorithm(sample_size=30, random_state=42):
    """
    Executes Find-S algorithm step-by-step on a representative sample of workout records.
    Returns:
    - Step-by-step trace of hypothesis updates
    - Final learned hypothesis
    - Sample dataset used
    """
    cleaned_path = os.path.join(DATA_DIR, "cleaned_calories.csv")
    if not os.path.exists(cleaned_path):
        from src.data_pipeline import prepare_train_test_data
        prepare_train_test_data()

    df = pd.read_csv(cleaned_path)
    df_disc = discretize_dataset(df)

    # Draw a balanced sample containing both positive and negative examples
    pos_samples = df_disc[df_disc["Target_Concept"] == "Yes"].sample(n=sample_size // 2, random_state=random_state)
    neg_samples = df_disc[df_disc["Target_Concept"] == "No"].sample(n=sample_size // 2, random_state=random_state)
    sample_df = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    feature_cols = ["Gender", "Age_Group", "Duration_Level", "Heart_Rate_Level", "Body_Temp_Level"]
    
    # Initialize most specific hypothesis
    hypothesis = ["phi"] * len(feature_cols)
    trace = []
    
    trace.append({
        "step": 0,
        "action": "Initialization",
        "instance": None,
        "target": None,
        "hypothesis": list(hypothesis),
        "explanation": "Start with most specific hypothesis: h_0 = <phi, phi, phi, phi, phi>"
    })

    step_num = 1
    first_positive = True

    for idx, row in sample_df.iterrows():
        instance_vals = [row[c] for c in feature_cols]
        target_val = row["Target_Concept"]

        if target_val == "Yes":
            old_h = list(hypothesis)
            if first_positive:
                hypothesis = list(instance_vals)
                first_positive = False
                explanation = f"First positive instance encountered. Set h_{step_num} directly to instance values."
            else:
                updated = False
                for i in range(len(feature_cols)):
                    if hypothesis[i] != instance_vals[i]:
                        hypothesis[i] = "?"
                        updated = True
                explanation = "Generalizing attributes that mismatch the positive instance with '?'." if updated else "Hypothesis already covers this positive instance without modification."

            trace.append({
                "step": step_num,
                "action": "Generalize (Positive Instance)",
                "instance": {col: val for col, val in zip(feature_cols, instance_vals)},
                "target": "Yes",
                "hypothesis": list(hypothesis),
                "explanation": explanation
            })
            step_num += 1
        else:
            # Negative instances are ignored by Find-S
            trace.append({
                "step": step_num,
                "action": "Ignored (Negative Instance)",
                "instance": {col: val for col, val in zip(feature_cols, instance_vals)},
                "target": "No",
                "hypothesis": list(hypothesis),
                "explanation": "Negative instance ignored by Find-S because inductive bias assumes consistent target concept."
            })
            step_num += 1

    final_h_dict = {col: val for col, val in zip(feature_cols, hypothesis)}

    output = {
        "feature_names": feature_cols,
        "target_concept": "High Calorie Burn (>= 150 kcal)",
        "most_specific_h0": ["phi"] * len(feature_cols),
        "final_hypothesis": hypothesis,
        "final_hypothesis_dict": final_h_dict,
        "trace": trace[:15], # Save first 15 steps for compact rendering
        "sample_preview": sample_df.head(10).to_dict(orient="records")
    }

    with open(os.path.join(MODELS_DIR, "find_s_trace.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"[FIND-S COMPLETE] Learned Hypothesis for High Calorie Burn:")
    print(f"  < {', '.join(hypothesis)} >")
    print(f"  Detailed explanation: Duration must be Long ({final_h_dict['Duration_Level']}), Heart Rate High ({final_h_dict['Heart_Rate_Level']}), etc.")

    return output


def evaluate_with_find_s(user_input_dict):
    """
    Evaluates whether a new workout matches the learned Find-S concept.
    """
    trace_path = os.path.join(MODELS_DIR, "find_s_trace.json")
    if not os.path.exists(trace_path):
        run_find_s_algorithm()

    with open(trace_path, "r") as f:
        data = json.load(f)

    h = data["final_hypothesis_dict"]

    # Discretize user input
    gender = str(user_input_dict.get("Gender", "Male")).capitalize()
    age = user_input_dict.get("Age", 30)
    age_group = "Young" if age <= 35 else ("Middle_Aged" if age <= 55 else "Senior")
    dur = user_input_dict.get("Duration", 15)
    dur_level = "Short_Duration" if dur <= 14 else "Long_Duration"
    hr = user_input_dict.get("Heart_Rate", 100)
    hr_level = "Moderate_HR" if hr <= 100 else "High_HR"
    temp = user_input_dict.get("Body_Temp", 37.0)
    temp_level = "Normal_Temp" if temp <= 40.0 else "Elevated_Temp"

    user_disc = {
        "Gender": gender,
        "Age_Group": age_group,
        "Duration_Level": dur_level,
        "Heart_Rate_Level": hr_level,
        "Body_Temp_Level": temp_level
    }

    matches = True
    for feat, h_val in h.items():
        if h_val != "?" and h_val != user_disc[feat]:
            matches = False
            break

    return {
        "user_discretized": user_disc,
        "hypothesis": h,
        "matches_concept": matches,
        "prediction": "High Calorie Burn (>=150 kcal)" if matches else "Standard/Low Calorie Burn (<150 kcal)"
    }


if __name__ == "__main__":
    run_find_s_algorithm()
