"""
=============================================================================
CALORIES BURNT PREDICTOR & HEALTH INTELLIGENCE SYSTEM
=============================================================================
Frontend Modules:
1. User Workout Input & Real-time Calorie Burn Prediction (Random Forest ML)
2. ML-Powered Exercise Ranking & Dynamic Personalized Prescription
3. Diet & Nutrition Plan (Target Daily Protein, Carbs & Food Recommendations)
4. Point-to-Point Health & Lifestyle Risk Flagging

Core ML Model: Scikit-Learn Random Forest Regressor
Packages: numpy, pandas, scikit-learn, matplotlib, seaborn, streamlit, joblib
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Streamlit Page Setup
st.set_page_config(
    page_title="Calories Burnt Predictor & Health Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .title-banner {
        font-size: 2.1rem;
        font-weight: 800;
        color: #FF4B4B;
        margin-bottom: 0.1rem;
    }
    .metric-container {
        background-color: #FFF5F5;
        border: 2px solid #FF4B4B;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2.6rem;
        font-weight: 900;
        color: #D9383A;
    }
    .metric-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .rec-box {
        background-color: #F0F7FF;
        border-left: 5px solid #2B6CB0;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 18px;
        font-size: 1.05rem;
        color: #1A365D;
        line-height: 1.6;
    }
    .diet-box {
        background-color: #F8F9FA;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LOAD TRAINED ARTIFACTS FROM BACKEND (Cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_backend_artifacts():
    model_path = os.path.join(MODELS_DIR, "model.joblib")
    if not os.path.exists(model_path):
        import subprocess
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "train_models.py")], check=True)

    artifacts = {
        "model": joblib.load(os.path.join(MODELS_DIR, "model.joblib")),
        "scaler": joblib.load(os.path.join(MODELS_DIR, "scaler.joblib")),
    }
    return artifacts


artifacts = load_backend_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]


# ---------------------------------------------------------------------------
# SIDEBAR: USER PHYSICAL & EXERCISE INPUT BOXES (NO SLIDERS / SCROLLBARS)
# ---------------------------------------------------------------------------
st.sidebar.markdown("## User Profile & Telemetry")
st.sidebar.caption("Enter your biometric and workout session parameters:")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0)
gender_val = 0 if gender == "Male" else 1

age = st.sidebar.number_input("Age (Years)", min_value=10, max_value=100, value=28, step=1)
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=175.0, step=0.5)
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=72.0, step=0.5)

fitness_goal = st.sidebar.selectbox(
    "Primary Fitness Objective",
    ["Fat Loss / Weight Management", "Cardiovascular Conditioning & Health", "Muscle Hypertrophy & Athletic Endurance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Workout Activity Telemetry")
duration = st.sidebar.number_input("Workout Duration (Minutes)", min_value=1.0, max_value=180.0, value=25.0, step=0.5)
heart_rate = st.sidebar.number_input("Heart Rate (BPM)", min_value=40.0, max_value=220.0, value=125.0, step=1.0)
body_temp = st.sidebar.number_input("Body Temperature (°C)", min_value=35.0, max_value=44.0, value=39.5, step=0.1)


# ---------------------------------------------------------------------------
# CORE COMPUTATIONS: BMI, HR ZONES, ML PREDICTIONS
# ---------------------------------------------------------------------------
# 1. BMI Calculation
height_m = height / 100.0
bmi = round(weight / (height_m ** 2), 1)
if bmi < 18.5:
    bmi_cat = "Underweight"
elif bmi < 25.0:
    bmi_cat = "Normal Weight"
elif bmi < 30.0:
    bmi_cat = "Overweight"
else:
    bmi_cat = "Obese"

# 2. Maximum Heart Rate
hr_max = int(220 - age)
hr_pct = round((heart_rate / hr_max) * 100.0, 1)

# 3. Model Prediction on Current Telemetry
# Training set max duration is 30 min. For sessions > 30 min, scale proportionally with session duration.
eval_dur = min(duration, 30.0)
dur_multiplier = (duration / eval_dur) if duration > 30.0 else 1.0

user_df = pd.DataFrame([{
    "Gender": gender_val,
    "Age": int(age),
    "Height": float(height),
    "Weight": float(weight),
    "Duration": float(eval_dur),
    "Heart_Rate": float(min(heart_rate, 128.0)),
    "Body_Temp": float(min(body_temp, 41.5))
}])

base_predicted = max(0.0, float(model.predict(user_df)[0]))
predicted_calories = round(base_predicted * dur_multiplier, 1)
burn_rate = round(predicted_calories / max(duration, 0.1), 1)

# Intensity Tier
if predicted_calories < 50:
    intensity_label, intensity_color = "Low Intensity", "#3498db"
elif predicted_calories <= 180:
    intensity_label, intensity_color = "Moderate Cardio", "#f39c12"
else:
    intensity_label, intensity_color = "High Intensity Effort", "#e74c3c"


# ---------------------------------------------------------------------------
# MAIN PAGE: HEADER & TOP 3 METRICS CARDS
# ---------------------------------------------------------------------------
st.markdown('<div class="title-banner">Calories Burnt Prediction & Recommendations</div>', unsafe_allow_html=True)
st.caption("Machine Learning Energy Expenditure Estimation, Exercise Ranking, Diet Planning & Risk Flags")

# 3 Focused Prediction Cards
col_cal, col_rate, col_bmi = st.columns([1.4, 1, 1])

with col_cal:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{predicted_calories} kcal</div>
        <div class="metric-label">Estimated Calories Burnt</div>
    </div>
    """, unsafe_allow_html=True)

with col_rate:
    st.markdown(f"""
    <div style="background-color:#F8F9FA; border-radius:12px; padding:18px; text-align:center; border:1px solid #E0E0E0;">
        <div style="font-size:2rem; font-weight:800; color:#2C3E50;">{burn_rate}</div>
        <div style="font-size:0.9rem; font-weight:600; color:#7F8C8D;">kcal / Minute</div>
        <div style="margin-top:6px; font-size:0.85rem; font-weight:700; color:{intensity_color};">{intensity_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col_bmi:
    st.markdown(f"""
    <div style="background-color:#F8F9FA; border-radius:12px; padding:18px; text-align:center; border:1px solid #E0E0E0;">
        <div style="font-size:2rem; font-weight:800; color:#2C3E50;">{bmi}</div>
        <div style="font-size:0.9rem; font-weight:600; color:#7F8C8D;">BMI ({bmi_cat})</div>
        <div style="margin-top:6px; font-size:0.85rem; font-weight:600; color:#555;">{weight}kg | {height}cm</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# RISK FACTORS EVALUATION (POINT-TO-POINT)
# ---------------------------------------------------------------------------
risk_items = []

# 1. Cardiovascular Risk
if hr_pct >= 85.0 or heart_rate >= 165.0:
    risk_items.append({
        "type": "error",
        "title": f"Cardiovascular Strain (Heart Rate: {heart_rate} BPM | {hr_pct}% of Max HR)",
        "risk": "High strain on heart muscle, rapid cardiovascular fatigue, and elevated blood pressure.",
        "advice": "Slow down to 120-140 BPM and take a 5-minute cooldown break."
    })
elif hr_pct >= 75.0 or heart_rate >= 145.0:
    risk_items.append({
        "type": "warning",
        "title": f"Elevated Heart Rate (Heart Rate: {heart_rate} BPM | {hr_pct}% of Max HR)",
        "risk": "Approaching anaerobic threshold; rapid lactic acid accumulation.",
        "advice": "Keep rest intervals between intense sets to prevent early exhaustion."
    })
else:
    risk_items.append({
        "type": "success",
        "title": f"Cardiovascular Load: Normal ({heart_rate} BPM | {hr_pct}% of Max HR)",
        "risk": "Heart rate is within safe aerobic limits.",
        "advice": "Maintain this steady pace for aerobic conditioning."
    })

# 2. Body Temperature / Heat Strain
if body_temp >= 40.0:
    risk_items.append({
        "type": "error",
        "title": f"Heat Strain: High ({body_temp}°C)",
        "risk": "Risk of exertional hyperthermia, severe dehydration, and heat cramps.",
        "advice": "Stop exercising immediately, drink cool electrolytes, and rest in a shaded/ventilated area."
    })
elif body_temp >= 39.2:
    risk_items.append({
        "type": "warning",
        "title": f"Heat Strain: Elevated ({body_temp}°C)",
        "risk": "Excessive sweat fluid loss and risk of mild heat exhaustion.",
        "advice": "Drink 250-500 mL water or electrolytes now; avoid exercising in humid environments."
    })
else:
    risk_items.append({
        "type": "success",
        "title": f"Heat Strain: Normal ({body_temp}°C)",
        "risk": "Body temperature reflects normal exercise heat.",
        "advice": "Maintain standard hydration during and after workout."
    })

# 3. Weight / BMI Risk
if bmi >= 30.0:
    risk_items.append({
        "type": "error",
        "title": f"Metabolic & Joint Risk (Obese: BMI {bmi})",
        "risk": "High stress on knee/hip joints, plus higher risk of Type 2 Diabetes and hypertension.",
        "advice": "Choose non-impact cardio (cycling, swimming) to protect joints; maintain a 500 kcal daily deficit."
    })
elif bmi >= 25.0:
    risk_items.append({
        "type": "warning",
        "title": f"Metabolic Risk (Overweight: BMI {bmi})",
        "risk": "Risk of joint wear, high cholesterol, and visceral fat accumulation.",
        "advice": "Target 45 minutes of moderate cardio 4 days a week with a balanced diet."
    })
elif bmi < 18.5:
    risk_items.append({
        "type": "warning",
        "title": f"Underweight Risk (BMI {bmi})",
        "risk": "Risk of muscle loss, nutrient deficiency, and reduced bone density.",
        "advice": "Focus on strength training instead of excessive cardio, and eat a 300-500 kcal surplus."
    })
else:
    risk_items.append({
        "type": "success",
        "title": f"Metabolic Profile: Healthy (BMI {bmi})",
        "risk": "Lowest risk category for chronic metabolic conditions.",
        "advice": "Maintain consistent physical activity and balanced nutrition."
    })

# 4. Duration / Overtraining Risk
if duration >= 60.0:
    risk_items.append({
        "type": "warning",
        "title": f"Workout Volume: Long Duration ({duration} min)",
        "risk": "Glycogen depletion, muscle breakdown, and delayed recovery.",
        "advice": "Cap high-intensity workouts to 45 minutes; allow 24-48 hours of recovery before retraining."
    })
else:
    risk_items.append({
        "type": "success",
        "title": f"Workout Volume: Optimal ({duration} min)",
        "risk": "Safe duration with low fatigue risk.",
        "advice": "Safe duration for regular workouts."
    })


# ---------------------------------------------------------------------------
# TABS INTERFACE: 3 CLEAN MODULES
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Personalized Recommendations",
    "Diet & Nutrition Plan",
    "Health & Risk Flags"
])


# ===========================================================================
# TAB 1: PERSONALIZED RECOMMENDATIONS & ML EXERCISE RANKING
# ===========================================================================
with tab1:
    st.subheader("Personalized Recommendations")

    # 1. Truly Dynamic Rule-Based Recommendation based on Duration, HR, and BMI
    # Analyze Heart Rate Intensity
    if heart_rate < 100:
        hr_feedback = f"Your heart rate ({heart_rate} BPM) is light. To increase calorie burn, push your pace into 125-140 BPM."
        rec_activity = "brisk running or rowing"
        rec_dur = 35
        rec_freq = 4
    elif heart_rate <= 140:
        hr_feedback = f"Your heart rate ({heart_rate} BPM) is in an optimal aerobic fat-burning zone."
        rec_activity = "running or cycling"
        rec_dur = 40
        rec_freq = 4
    else:
        hr_feedback = f"Your heart rate ({heart_rate} BPM) is very high (anaerobic zone). Limit continuous threshold training to prevent early exhaustion."
        rec_activity = "interval training (HIIT or spin cycling) with 2-minute active recovery breaks"
        rec_dur = 25
        rec_freq = 3

    # Analyze Duration
    if duration > 60:
        dur_feedback = f"Because {duration:.0f} min is a prolonged session, capping workouts at 45-50 min at higher cadence will protect joints and avoid muscle breakdown."
        rec_dur = 45
        rec_freq = 3
    elif duration < 20:
        dur_feedback = f"Your current session is short ({duration:.0f} min). Stepping up to 30-40 min builds deeper stamina."
        rec_dur = 35
        rec_freq = 5
    else:
        dur_feedback = f"Your current session duration ({duration:.0f} min) is well balanced for conditioning."

    # Analyze BMI constraints
    if bmi >= 30.0:
        rec_activity = "low-impact cycling, incline walking, or swimming"
        rec_rationale = "shield knee joints while maximizing fat oxidation"
        weekly_target = 1800
    elif bmi >= 25.0:
        rec_activity = "moderate cycling, rowing, or jog-walk intervals"
        rec_rationale = "accelerate daily fat loss and improve insulin sensitivity"
        weekly_target = 1600
    elif bmi < 18.5:
        rec_activity = "bodyweight circuits and light swimming"
        rec_dur = 25
        rec_freq = 3
        rec_rationale = "prevent muscle catabolism and preserve lean tissue"
        weekly_target = 900
    else:
        rec_rationale = "maintain peak cardiovascular endurance and metabolic conditioning"
        weekly_target = 1400

    rule_recommendation = (
        f"Your BMI is {bmi} ({bmi_cat}) and you burned {predicted_calories} kcal in {duration:.0f} min "
        f"at an average heart rate of {heart_rate} BPM ({burn_rate} kcal/min). "
        f"{hr_feedback} {dur_feedback} "
        f"Prescription: Try {rec_dur} min of {rec_activity} {rec_freq} days a week to {rec_rationale} "
        f"(Weekly Target: {weekly_target} kcal)."
    )

    st.markdown(f"""
    <div class="rec-box">
        <strong>Prescription:</strong><br>
        {rule_recommendation}
    </div>
    """, unsafe_allow_html=True)

    # 2. Machine Learning Exercise Ranking
    st.markdown(f"#### Exercise Suggestions Ranked by Calories Burnt for Your Profile ({duration:.0f} min Session)")
    st.caption(
        f"Predictions evaluated for your profile (Age: {age}, Gender: {gender}, Weight: {weight}kg, Height: {height}cm, Duration: {duration:.0f} min) "
        f"using the trained Random Forest ML model:"
    )

    # Realistic physiological profiles within training distribution range (HR 75-125 BPM)
    exercise_catalog = [
        {"Activity": "HIIT Circuit Training", "HR": 124.0, "Temp": 40.8, "Focus": "High-intensity anaerobic afterburn"},
        {"Activity": "Jump Rope", "HR": 120.0, "Temp": 40.7, "Focus": "Cardio agility & fast calorie burn"},
        {"Activity": "Outdoor Running", "HR": 118.0, "Temp": 40.6, "Focus": "Aerobic endurance & leg power"},
        {"Activity": "Rowing Machine", "HR": 112.0, "Temp": 40.4, "Focus": "Full-body muscular cardio"},
        {"Activity": "Fast Cycling / Spin", "HR": 108.0, "Temp": 40.2, "Focus": "Lower-body stamina & joint safety"},
        {"Activity": "Freestyle Swimming", "HR": 102.0, "Temp": 39.8, "Focus": "Zero-impact cardiovascular capacity"},
        {"Activity": "Incline Treadmill Walk", "HR": 92.0, "Temp": 39.4, "Focus": "Zone 2 fat oxidation without joint strain"},
        {"Activity": "Power Yoga / Calisthenics", "HR": 80.0, "Temp": 38.6, "Focus": "Core stability, flexibility & light burn"}
    ]

    rank_rows = []
    for ex in exercise_catalog:
        row_feat = {
            "Gender": gender_val,
            "Age": int(age),
            "Height": float(height),
            "Weight": float(weight),
            "Duration": float(eval_dur),
            "Heart_Rate": ex["HR"],
            "Body_Temp": ex["Temp"]
        }
        pred_base = max(0.0, float(model.predict(pd.DataFrame([row_feat]))[0]))
        pred_scaled = round(pred_base * dur_multiplier, 1)
        b_rate = round(pred_scaled / max(duration, 0.1), 2)
        rank_rows.append({
            "Exercise Modality": ex["Activity"],
            "Duration (min)": int(duration),
            "Heart Rate (BPM)": int(ex["HR"]),
            "Predicted Calories (kcal)": pred_scaled,
            "Burn Rate (kcal/min)": b_rate,
            "Training Focus": ex["Focus"]
        })

    df_ranked = pd.DataFrame(rank_rows).sort_values(by="Burn Rate (kcal/min)", ascending=False).reset_index(drop=True)
    df_ranked.index += 1
    df_ranked.index.name = "Rank"

    st.dataframe(df_ranked, use_container_width=True)


# ===========================================================================
# TAB 2: DIET & NUTRITION PLAN
# ===========================================================================
with tab2:
    st.subheader("Diet & Nutrition Plan")
    st.caption("Daily protein, carbohydrate and caloric targets for your body weight and exercise:")

    # BMR Calculation (Mifflin-St Jeor)
    if gender == "Male":
        bmr = (10.0 * weight) + (6.25 * height) - (5.0 * age) + 5.0
    else:
        bmr = (10.0 * weight) + (6.25 * height) - (5.0 * age) - 161.0
    bmr = round(bmr, 0)

    # TDEE Calculation (Baseline 1.35 factor + session expenditure)
    tdee = round((bmr * 1.35) + predicted_calories, 0)

    # Caloric Target based on Goal
    if "Fat Loss" in fitness_goal:
        target_calories = max(1400.0, round(tdee - 450.0, 0))
        target_desc = "Caloric Deficit (-450 kcal/day for fat loss)"
        carb_pct, fat_pct = 0.40, 0.30
    elif "Hypertrophy" in fitness_goal:
        target_calories = round(tdee + 350.0, 0)
        target_desc = "Caloric Surplus (+350 kcal/day for muscle growth)"
        carb_pct, fat_pct = 0.50, 0.25
    else:
        target_calories = tdee
        target_desc = "Maintenance (Balanced with daily expenditure)"
        carb_pct, fat_pct = 0.45, 0.30

    # Macronutrient Calculations
    # Protein: 2.0g per kg of bodyweight
    protein_grams = int(round(weight * 2.0, 0))
    protein_cals = protein_grams * 4

    fat_cals = int(round(target_calories * fat_pct, 0))
    fat_grams = int(round(fat_cals / 9.0, 0))

    carb_cals = max(0, int(round(target_calories - protein_cals - fat_cals, 0)))
    carb_grams = int(round(carb_cals / 4.0, 0))

    # Top Nutrition Metric Cards
    col_n1, col_n2, col_n3 = st.columns(3)
    with col_n1:
        st.metric("Daily Protein Needed", f"{protein_grams} g / day")
    with col_n2:
        st.metric("Daily Carbs Needed", f"{carb_grams} g / day")
    with col_n3:
        st.metric("Daily Caloric Target", f"{int(target_calories)} kcal / day")

    st.markdown(f"**Objective:** {target_desc} | Healthy Fats: **{fat_grams} g / day**")

    # Section 1: Protein Breakdown & Foods to Eat
    st.markdown("---")
    st.markdown(f"### 1. Protein Needed: **{protein_grams} g daily**")
    st.caption(f"Based on 2.0g per kg body weight ({weight}kg) for muscle preservation and recovery:")
    st.markdown("""
    **Foods to consume to reach your daily protein target:**
    - **Chicken Breast or Fish (150-200g):** provides **45-55g** protein
    - **Eggs (3 whole eggs or egg whites):** provides **18-20g** protein
    - **Paneer or Tofu (150g):** provides **28-30g** protein
    - **Cooked Dal / Lentils / Chickpeas (1 cup / 200g):** provides **15-18g** protein
    - **Greek Yogurt or Curd (1 bowl / 200g):** provides **15-20g** protein
    - **Whey Protein or Soya Chunks (1 scoop / 40g):** provides **24-26g** protein
    - **Almonds / Peanuts (1 handful / 30g):** provides **6-8g** protein
    """)

    # Section 2: Carbohydrate Breakdown & Foods to Eat
    st.markdown("---")
    st.markdown(f"### 2. Carbohydrates Needed: **{carb_grams} g daily**")
    st.caption("To replenish glycogen stores, sustain workout stamina, and prevent fatigue:")
    st.markdown("""
    **Foods to consume to reach your daily carbohydrate target:**
    - **Cooked Rice (White or Brown, 1.5 cups):** provides **65-70g** carbs
    - **Whole Wheat Roti / Chapati (2-3 rotis):** provides **45-55g** carbs
    - **Rolled Oats (1 bowl / 60g raw):** provides **40g** carbs
    - **Bananas or Apples (2 medium fruits):** provides **45-50g** carbs
    - **Sweet Potato or Boiled Potato (1 medium):** provides **25-30g** carbs
    - **Whole Grain Bread (2 slices):** provides **25-30g** carbs
    """)

    # Section 3: Healthy Fats & Hydration
    st.markdown("---")
    st.markdown(f"### 3. Healthy Fats & Hydration: **{fat_grams} g fats daily**")
    st.markdown("""
    - **Healthy Fat Sources:** 1-2 tablespoons olive oil/ghee (15-20g), 1 handful raw walnuts/almonds (12-15g), 1 spoon chia or flax seeds (5g).
    - **Daily Water Intake:** Minimum **3.5 Liters of water daily**, plus 500 mL following intense exercise.
    """)


# ===========================================================================
# TAB 3: HEALTH & RISK FLAGS (POINT-TO-POINT)
# ===========================================================================
with tab3:
    st.subheader("Health & Lifestyle Risk Flags")
    st.caption("Point-to-point risk screening based on your biometric and workout inputs:")

    for item in risk_items:
        if item["type"] == "error":
            st.error(f"**{item['title']}**\n- **Risk:** {item['risk']}\n- **Advice:** {item['advice']}")
        elif item["type"] == "warning":
            st.warning(f"**{item['title']}**\n- **Risk:** {item['risk']}\n- **Advice:** {item['advice']}")
        else:
            st.success(f"**{item['title']}**\n- **Risk:** {item['risk']}\n- **Advice:** {item['advice']}")


# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#888; font-size:0.85rem;'>Calories Burnt Prediction & Health Intelligence System | Powered by Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn & Streamlit</div>", unsafe_allow_html=True)
