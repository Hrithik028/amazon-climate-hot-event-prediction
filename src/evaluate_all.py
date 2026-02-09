from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from scipy.stats import pearsonr

from .data import load_temperature_data, load_thresholds, add_hot_label
from .features import make_feature_matrix, transform_features
from .splitting import random_split, yearwise_split
from .plotting import save_confusion_matrix, save_true_vs_pred

import tensorflow as tf

DATA_DIR = "data"
MODELS_DIR = "models"
FIG_DIR = "figures"

def eval_classifier(df):
    # Use the same split method as training for portfolio demo
    train_df, val_df, test_df = random_split(df)

    scaler = joblib.load(os.path.join(MODELS_DIR, "classification_scaler.joblib"))
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "classification_model.keras"))

    X_test, _ = make_feature_matrix(test_df, include_month_cyclic=True)
    X_test_s = transform_features(scaler, X_test)
    y_true = test_df["Hot"].to_numpy()

    y_prob = model.predict(X_test_s, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Specificity (TNR) and Sensitivity (TPR)
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")

    save_confusion_matrix(cm, os.path.join(FIG_DIR, "clf_confusion_matrix.png"))

    print("== Classification (Random split test) ==")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Specificity (TNR): {tnr:.4f}")
    print(f"Sensitivity (TPR): {tpr:.4f}")
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(cm)

def eval_regressors(df):
    # Random split regressor
    train_df, val_df, test_df = random_split(df)

    fscaler = joblib.load(os.path.join(MODELS_DIR, "regression_feature_scaler.joblib"))
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "regression_model.keras"))

    X_test, _ = make_feature_matrix(test_df, include_month_cyclic=True)
    X_test_s = transform_features(fscaler, X_test)
    y_true = test_df["Temp"].to_numpy(dtype=float)

    y_pred = model.predict(X_test_s, verbose=0).reshape(-1)

    r, _ = pearsonr(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))

    save_true_vs_pred(y_true, y_pred, os.path.join(FIG_DIR, "reg_true_vs_pred_random.png"),
                      title="True vs Predicted (Random split)")

    print("\n== Regression (Random split test) ==")
    print(f"Pearson r: {r:.4f}")
    print(f"MAE: {mae:.4f}")

    # Year-wise regressor (inverse-transform targets)
    ytrain_df, yval_df, ytest_df = yearwise_split(df)

    y_fscaler = joblib.load(os.path.join(MODELS_DIR, "yearwise_regression_feature_scaler.joblib"))
    tscaler = joblib.load(os.path.join(MODELS_DIR, "yearwise_regression_target_scaler.joblib"))
    y_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "yearwise_regression_model.keras"))

    X_ytest, _ = make_feature_matrix(ytest_df, include_month_cyclic=True)
    X_ytest_s = transform_features(y_fscaler, X_ytest)
    y_true_y = ytest_df["Temp"].to_numpy(dtype=float)

    y_pred_s = y_model.predict(X_ytest_s, verbose=0).reshape(-1)
    y_pred = tscaler.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(-1)

    r2, _ = pearsonr(y_true_y, y_pred)
    mae2 = np.mean(np.abs(y_true_y - y_pred))

    save_true_vs_pred(y_true_y, y_pred, os.path.join(FIG_DIR, "reg_true_vs_pred_yearwise.png"),
                      title="True vs Predicted (Year-wise split)")

    print("\n== Regression (Year-wise split test) ==")
    print(f"Pearson r: {r2:.4f}")
    print(f"MAE: {mae2:.4f}")

def main():
    temp_path = os.path.join(DATA_DIR, "Amazon_temperature_student.csv")
    thr_path = os.path.join(DATA_DIR, "thresholds.csv")

    df = load_temperature_data(temp_path)
    thr = load_thresholds(thr_path)
    df = add_hot_label(df, thr)

    os.makedirs(FIG_DIR, exist_ok=True)

    eval_classifier(df)
    eval_regressors(df)

if __name__ == "__main__":
    main()
