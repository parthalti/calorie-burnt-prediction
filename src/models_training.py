"""
Models Training and Evaluation Module for Calories Burnt Prediction.
Trains, tunes, and benchmarks 5 regression models:
1. Linear Regression (Baseline with coefficient analysis)
2. Decision Tree Regressor (Non-linear baseline with depth regularization)
3. Random Forest Regressor (Ensemble bagging with feature importances)
4. K-Nearest Neighbors Regressor (Distance-based with hyperparameter tuning on k)
5. XGBoost Regressor (State-of-the-art Gradient Boosting)

Evaluates:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- 5-Fold Cross-Validation (Mean ± Std)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib

from src.data_pipeline import prepare_train_test_data, MODELS_DIR


def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(round(mae, 4)),
        "RMSE": float(round(rmse, 4)),
        "R2": float(round(r2, 4))
    }


def perform_5fold_cv(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    neg_mae_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
    neg_rmse_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)

    return {
        "CV_R2_Mean": float(round(np.mean(r2_scores), 4)),
        "CV_R2_Std": float(round(np.std(r2_scores), 4)),
        "CV_MAE_Mean": float(round(-np.mean(neg_mae_scores), 4)),
        "CV_MAE_Std": float(round(np.std(neg_mae_scores), 4)),
        "CV_RMSE_Mean": float(round(-np.mean(neg_rmse_scores), 4)),
        "CV_RMSE_Std": float(round(np.std(neg_rmse_scores), 4))
    }


def train_and_benchmark_models():
    print("[INFO] Preparing train and test datasets...")
    data = prepare_train_test_data()
    X_train = data["X_train"]
    X_test = data["X_test"]
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["split_summary"]["features"]

    results = {}
    test_predictions = {"Actual": y_test.tolist()}

    # -------------------------------------------------------------
    # 1. Linear Regression (Baseline)
    # -------------------------------------------------------------
    print("\n[1/5] Training Linear Regression (Baseline)...")
    lr = LinearRegression()
    # Train on scaled features for standard comparison and on raw for interpretable equation
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_metrics = evaluate_predictions(y_test, lr_pred)
    lr_cv = perform_5fold_cv(lr, X_train_scaled, y_train)

    # Raw coefficient analysis for equation
    lr_raw = LinearRegression().fit(X_train, y_train)
    coeff_breakdown = {feat: float(round(coef, 4)) for feat, coef in zip(feature_names, lr_raw.coef_)}

    results["Linear Regression"] = {
        **lr_metrics,
        **lr_cv,
        "Coefficients_Raw": coeff_breakdown,
        "Intercept_Raw": float(round(lr_raw.intercept_, 4)),
        "Uses_Scaled_Features": True
    }
    test_predictions["Linear Regression"] = lr_pred.tolist()
    joblib.dump(lr, os.path.join(MODELS_DIR, "linear_regression.joblib"))
    print(f"      Linear Regression -> MAE: {lr_metrics['MAE']} | RMSE: {lr_metrics['RMSE']} | R²: {lr_metrics['R2']}")

    # -------------------------------------------------------------
    # 2. Decision Tree Regressor
    # -------------------------------------------------------------
    print("\n[2/5] Training Decision Tree Regressor...")
    dt = DecisionTreeRegressor(max_depth=9, min_samples_split=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_metrics = evaluate_predictions(y_test, dt_pred)
    dt_cv = perform_5fold_cv(dt, X_train, y_train)

    dt_importances = {feat: float(round(imp, 4)) for feat, imp in zip(feature_names, dt.feature_importances_)}
    results["Decision Tree"] = {
        **dt_metrics,
        **dt_cv,
        "Feature_Importances": dt_importances,
        "Max_Depth": 9,
        "Uses_Scaled_Features": False
    }
    test_predictions["Decision Tree"] = dt_pred.tolist()
    joblib.dump(dt, os.path.join(MODELS_DIR, "decision_tree.joblib"))
    print(f"      Decision Tree -> MAE: {dt_metrics['MAE']} | RMSE: {dt_metrics['RMSE']} | R²: {dt_metrics['R2']}")

    # -------------------------------------------------------------
    # 3. Random Forest Regressor
    # -------------------------------------------------------------
    print("\n[3/5] Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = evaluate_predictions(y_test, rf_pred)
    rf_cv = perform_5fold_cv(rf, X_train, y_train)

    rf_importances = {feat: float(round(imp, 4)) for feat, imp in zip(feature_names, rf.feature_importances_)}
    results["Random Forest"] = {
        **rf_metrics,
        **rf_cv,
        "Feature_Importances": rf_importances,
        "N_Estimators": 100,
        "Max_Depth": 12,
        "Uses_Scaled_Features": False
    }
    test_predictions["Random Forest"] = rf_pred.tolist()
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))
    print(f"      Random Forest -> MAE: {rf_metrics['MAE']} | RMSE: {rf_metrics['RMSE']} | R²: {rf_metrics['R2']}")

    # -------------------------------------------------------------
    # 4. K-Nearest Neighbors Regressor (Hyperparameter Tuning for k)
    # -------------------------------------------------------------
    print("\n[4/5] Tuning and Training KNN Regressor (Scaled Features)...")
    k_candidates = [3, 5, 7, 9, 11, 15]
    param_grid = {"n_neighbors": k_candidates, "weights": ["uniform", "distance"]}
    knn_search = GridSearchCV(
        KNeighborsRegressor(),
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    knn_search.fit(X_train_scaled, y_train)
    best_knn = knn_search.best_estimator_
    best_k = int(knn_search.best_params_["n_neighbors"])
    best_weights = str(knn_search.best_params_["weights"])

    knn_pred = best_knn.predict(X_test_scaled)
    knn_metrics = evaluate_predictions(y_test, knn_pred)
    knn_cv = perform_5fold_cv(best_knn, X_train_scaled, y_train)

    results["KNN Regressor"] = {
        **knn_metrics,
        **knn_cv,
        "Best_K": best_k,
        "Best_Weights": best_weights,
        "K_Candidates_Evaluated": k_candidates,
        "Uses_Scaled_Features": True
    }
    test_predictions["KNN Regressor"] = knn_pred.tolist()
    joblib.dump(best_knn, os.path.join(MODELS_DIR, "knn.joblib"))
    print(f"      KNN (k={best_k}, weights={best_weights}) -> MAE: {knn_metrics['MAE']} | RMSE: {knn_metrics['RMSE']} | R²: {knn_metrics['R2']}")

    # -------------------------------------------------------------
    # 5. XGBoost Regressor (State-of-the-Art)
    # -------------------------------------------------------------
    print("\n[5/5] Training XGBoost Regressor (SOTA Gradient Boosting)...")
    xgb = XGBRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_metrics = evaluate_predictions(y_test, xgb_pred)
    xgb_cv = perform_5fold_cv(xgb, X_train, y_train)

    xgb_importances = {feat: float(round(imp, 4)) for feat, imp in zip(feature_names, xgb.feature_importances_)}
    results["XGBoost"] = {
        **xgb_metrics,
        **xgb_cv,
        "Feature_Importances": xgb_importances,
        "N_Estimators": 150,
        "Learning_Rate": 0.08,
        "Max_Depth": 5,
        "Uses_Scaled_Features": False
    }
    test_predictions["XGBoost"] = xgb_pred.tolist()
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgboost.joblib"))
    print(f"      XGBoost -> MAE: {xgb_metrics['MAE']} | RMSE: {xgb_metrics['RMSE']} | R²: {xgb_metrics['R2']}")

    # Save summary dictionary and test predictions
    summary = {
        "models": results,
        "cleaning_report": data["cleaning_report"],
        "split_summary": data["split_summary"],
        "feature_names": feature_names
    }

    with open(os.path.join(MODELS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(test_predictions).to_csv(os.path.join(MODELS_DIR, "test_predictions.csv"), index=False)
    print(f"\n[METRICS SAVED] Summary saved to {os.path.join(MODELS_DIR, 'metrics_summary.json')}")
    print(f"[PREDICTIONS SAVED] Test predictions saved to {os.path.join(MODELS_DIR, 'test_predictions.csv')}")

    return summary


if __name__ == "__main__":
    train_and_benchmark_models()
