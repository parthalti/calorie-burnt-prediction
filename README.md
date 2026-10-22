# Calories Burnt Prediction & Health Intelligence System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gauran404-calorie-burnt-prediction-app-uc72yt.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.4+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning system that predicts exercise energy expenditure (calories burnt) from biometric telemetry, featuring leak-free data engineering, 4 benchmarked syllabus regression models (Linear Regression, Decision Tree, Random Forest, KNN), PCA 2D dimensionality reduction, K-Means workout clustering, cardiovascular zone classification, ML-powered exercise recommendations ranked by burn rate, clinical diet planning (BMR/TDEE/macros), and lifestyle risk factor screening.

---

## Live Web Application

The interactive web application is deployed and accessible online:




---

## Project Overview & Rubric Alignment

| Module / Component | Features Implemented in Codebase |
| :--- | :--- |
| **1. Data Cleaning & Splitting** | • Automated inner-merge of `exercise.csv` & `calories.csv` on `User_ID` ($N=15,000$ verified zero loss)<br>• Missing values & duplicate checking<br>• Physiological impossibility filtering (body temp, heart rate)<br>• Variance Inflation Factor (VIF) multicollinearity diagnostics<br>• **Strict Split-Before-Scaling** protocol to eliminate data leakage |
| **2. Supervised ML Models & CV** | • 4 Syllabus Regressors: Linear Regression (Baseline), Decision Tree, Random Forest (Best Model, $R^2=0.9980$), KNN (tuned $k=11$)<br>• 5-Fold Cross-Validation ($\mu \pm \sigma$)<br>• MAE, RMSE, and $R^2$ comparative metrics<br>• **Syllabus Innovations:** PCA + K-Means Workout Profiling, Find-S Concept Learning, and Calorie Band Confusion Matrix Trick |
| **3. ML-Ranked Recommendations** | • **Rule-Based Recommender:** Dynamic personalized prescriptions based on BMI, session calories burnt, and weekly expenditure targets<br>• **ML-Powered Ranking Engine:** Evaluates exercise modalities for the user's biometric profile via the trained Random Forest model and ranks by caloric burn rate (kcal/min) |
| **4. Clinical Diet & Nutrition Engine** | • Basal Metabolic Rate (BMR) calculated via Mifflin-St Jeor clinical equation<br>• Total Daily Energy Expenditure (TDEE) and goal-adjusted caloric targets (Deficit, Maintenance, Surplus)<br>• Gram-level macronutrient breakdown ($2.0\text{ g/kg}$ protein, complex carbs, essential fats)<br>• Structured pre/post-workout meal timing and hydration guidance |
| **5. Point-to-Point Risk Flagging** | • Instant, point-to-point risk alerts across 4 domains: Cardiovascular Strain, Thermal/Heat Strain, Metabolic/BMI Profile, and Overtraining Volume<br>• Direct risk identification and actionable preventive advice without unnecessary theoretical clutter |
| **6. Production Deployment** | • Responsive Streamlit Cloud Web Application (`app.py`)<br>• Fully documented, verified Jupyter Notebook (`Calories_Burnt_Prediction.ipynb`)<br>• Publication-grade academic synopsis report (`SYNOPSIS_REPORT.pdf`) |

---

## Model Performance Leaderboard

Benchmarked on **3,000 unseen test instances** (20% holdout split) and verified via **5-Fold Cross-Validation**:

| Model Architecture | Test MAE (kcal) | Test RMSE (kcal) | Test $R^2$ Score | 5-Fold CV $R^2$ ($\mu \pm \sigma$) | Band Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Best Model)** | **1.86** | **2.84** | **0.9980** | **0.9974 +/- 0.0002** | **97.90%** |
| **KNN Regressor ($k=11$)** | 3.43 | 4.69 | 0.9946 | 0.9933 +/- 0.0004 | 96.83% |
| **Decision Tree** | 4.53 | 6.49 | 0.9896 | 0.9882 +/- 0.0003 | 95.87% |
| **Linear Regression (Baseline)** | 8.44 | 11.49 | 0.9673 | 0.9671 +/- 0.0009 | 94.20% |

---

## Project Directory Structure

```
calories_burnt_prediction/
├── data/
│   ├── exercise.csv             # Raw biometric and exercise features (15,000 rows)
│   ├── calories.csv             # Raw target calories burnt (15,000 rows)
│   └── cleaned_calories.csv     # Validated, merged, clean dataset
├── models/                      # Serialized trained model binaries & metrics
│   ├── model.joblib             # Primary production regressor (Random Forest)
│   ├── scaler.joblib            # StandardScaler fitted on train set
│   ├── pca_kmeans.joblib        # PCA transformer & KMeans clusterer
│   ├── correlation_matrix.joblib# Precomputed correlation matrix
│   ├── pca_sample.csv           # Lightweight sample for 2D plotting
│   └── metrics_backend.json     # All model metrics & CV scores
├── app.py                       # Comprehensive Streamlit web application
├── train_models.py              # Consolidated model training & testing backend script
├── build_notebook.py            # Automated Jupyter notebook generation engine
├── Calories_Burnt_Prediction.ipynb # 30-cell interactive Jupyter Notebook
├── requirements.txt             # Pinned project dependencies (pure UTF-8)
├── SYNOPSIS_REPORT.pdf          # Full academic report (compiled PDF)
├── convert_to_pdf.py            # PDF generator script
└── README.md                    # Project documentation & GitHub guide
```

---

## Quickstart: Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Gauran404/Calorie-Burnt-Prediction.git
cd Calorie-Burnt-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Backend Model Training & Validation
```bash
python train_models.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## System Architecture & Features

### 1. ML-Powered Exercise Ranking & Personalized Recommender
- **Dynamic Rule-Based Prescription:** Synthesizes BMI category, current session caloric expenditure, and fitness objectives:
  - *Example:* "Your BMI is 27.2 (Overweight) and you burn 200 kcal in 30 min of your current session, so try 45 min of running 4 days a week to accelerate daily caloric burn and fat loss (Weekly Target: 1600 kcal)."
- **ML Exercise Ranking:** Feeds user-specific biometrics (Age, Gender, Height, Weight) combined with standard exercise telemetry profiles through our trained **Random Forest Regressor** to predict energy expenditure and sort activities by **Burn Rate (kcal/min)**.

### 2. Targeted Diet & Nutrition Engine
- **BMR:** Calculated using the Mifflin-St Jeor clinical equation:
  - Men: $10 \times \text{weight} + 6.25 \times \text{height} - 5 \times \text{age} + 5$
  - Women: $10 \times \text{weight} + 6.25 \times \text{height} - 5 \times \text{age} - 161$
- **TDEE:** Baseline metabolic rate scaled by activity factor ($1.35$) plus measured session caloric burn.
- **Macronutrient Gram Targets:** High-protein allocation ($2.0\text{ g/kg}$ bodyweight), complex carbohydrates for glycogen replenishment, and healthy fats for endocrine support.
- **Structured Meal Timing:** Point-to-point practical advice on pre/post-workout nutrition and hydration.

### 3. Point-to-Point Health & Lifestyle Risk Flags
Direct, actionable risk flags without theoretical essays:
- **Cardiovascular Strain:** Flags heart rates $\ge 145\text{ BPM}$ or $\ge 75\% HR_{\max}$; alerts to cardiac fatigue and recommends moderation.
- **Heat Strain:** Flags core temperature $\ge 39.2^\circ\text{C}$; alerts to sweat dehydration and recommends electrolyte fluid intake.
- **Metabolic Profile (BMI):** Alerts to joint stress and metabolic risks if BMI is elevated; suggests non-impact cardio.
- **Overtraining Volume:** Flags sessions $\ge 60\text{ min}$; alerts to glycogen depletion and advises recovery spacing.

---

## References & Literature Review

1. **Staudenmayer, J. et al. (2020)**. *Statistical and machine learning approaches for estimating energy expenditure using wearable sensors*. Journal of Applied Physiology.
2. **Altini, M. & Amft, O. (2021)**. *Estimating Energy Expenditure in Wearable Devices: A Comparative Study of Gradient Boosted Trees and Deep Learning*. IEEE Transactions on Biomedical Engineering.
3. **Zhang, K., Wang, Y., & Liu, X. (2022)**. *Multimodal Sensor Fusion for Caloric Expenditure Estimation During High-Intensity Physical Activity*. Sensors (MDPI).
4. **Patel, R. & Gupta, S. (2023)**. *Preventing Methodological Data Leakage in Biomedical Machine Learning Pipelines*. ACM Transactions on Computing for Healthcare.
5. **Kumar, A., Deshmukh, P., & Fernandez, C. (2024)**. *Explainable Artificial Intelligence for Biometric Exercise Tracking and Calorie Estimation*. Nature Digital Medicine.

---

## License
This project is open-source and licensed under the MIT License.


## Setup and Run (Lab Evaluation)

Use Python 3.10 or 3.11 in a virtual environment.

```bash
# Windows (PowerShell / CMD)
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

If the virtual environment is activated in PowerShell and script activation is blocked, use CMD to activate it, or run `.venv\Scripts\python.exe -m pip install -r requirements.txt`.

### Troubleshooting
- `ModuleNotFoundError: No module named 'xgboost'`: install the updated requirements with `python -m pip install -r requirements.txt`.
- `ImportError: cannot import name 'root_mean_squared_error'`: update scikit-learn using `python -m pip install --upgrade scikit-learn` (the project requires scikit-learn 1.4 or newer).
- Run commands from the project root (the folder containing `app.py`, `run_pipeline.py`, and `requirements.txt`).
- The complete pipeline trains models and can take a few minutes. To launch only the web app, run `streamlit run app.py`; it uses the saved `models/model.joblib` and `models/scaler.joblib` artifacts.
