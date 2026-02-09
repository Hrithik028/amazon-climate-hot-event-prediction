from __future__ import annotations

import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED
from .data import load_temperature_data, load_thresholds, add_hot_label
from .features import make_feature_matrix, fit_feature_scaler, transform_features
from .splitting import random_split, yearwise_split
from .models import build_classifier, build_regressor
from .plotting import save_accuracy_plot, save_loss_plot

DATA_DIR = "data"
MODELS_DIR = "models"
FIG_DIR = "figures"

def train_classifier(df):
    train_df, val_df, test_df = random_split(df)

    X_train, feat_names = make_feature_matrix(train_df, include_month_cyclic=True)
    X_val, _ = make_feature_matrix(val_df, include_month_cyclic=True)
    y_train = train_df["Hot"].to_numpy()
    y_val = val_df["Hot"].to_numpy()

    scaler = fit_feature_scaler(X_train)
    X_train_s = transform_features(scaler, X_train)
    X_val_s = transform_features(scaler, X_val)

    model = build_classifier(input_dim=X_train_s.shape[1])
    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=60,
        batch_size=32,
        verbose=0,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(os.path.join(MODELS_DIR, "classification_model.keras"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "classification_scaler.joblib"))

    save_accuracy_plot(history, os.path.join(FIG_DIR, "clf_accuracy_vs_epochs.png"), title="Classifier Accuracy vs Epochs")
    return feat_names

def train_regressors(df):
    # Random split regression
    train_df, val_df, test_df = random_split(df)

    X_train, feat_names = make_feature_matrix(train_df, include_month_cyclic=True)
    X_val, _ = make_feature_matrix(val_df, include_month_cyclic=True)
    y_train = train_df["Temp"].to_numpy(dtype=float)
    y_val = val_df["Temp"].to_numpy(dtype=float)

    fscaler = fit_feature_scaler(X_train)
    X_train_s = transform_features(fscaler, X_train)
    X_val_s = transform_features(fscaler, X_val)

    r_model = build_regressor(input_dim=X_train_s.shape[1])
    r_hist = r_model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=80,
        batch_size=32,
        verbose=0,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    r_model.save(os.path.join(MODELS_DIR, "regression_model.keras"))
    joblib.dump(fscaler, os.path.join(MODELS_DIR, "regression_feature_scaler.joblib"))
    save_loss_plot(r_hist, os.path.join(FIG_DIR, "reg_loss_vs_epochs_random.png"), title="Regressor Loss vs Epochs (Random split)")

    # Year-wise split regression (with separate target scaler)
    ytrain_df, yval_df, ytest_df = yearwise_split(df)

    X_ytrain, _ = make_feature_matrix(ytrain_df, include_month_cyclic=True)
    X_yval, _ = make_feature_matrix(yval_df, include_month_cyclic=True)
    y_ytrain = ytrain_df["Temp"].to_numpy(dtype=float)
    y_yval = yval_df["Temp"].to_numpy(dtype=float)

    # Reuse the SAME feature scaler approach: fit on year-wise train only
    y_fscaler = fit_feature_scaler(X_ytrain)
    X_ytrain_s = transform_features(y_fscaler, X_ytrain)
    X_yval_s = transform_features(y_fscaler, X_yval)

    # Separate target scaler (fit on year-wise train targets only)
    tscaler = StandardScaler()
    y_ytrain_s = tscaler.fit_transform(y_ytrain.reshape(-1, 1)).reshape(-1)
    y_yval_s = tscaler.transform(y_yval.reshape(-1, 1)).reshape(-1)

    y_model = build_regressor(input_dim=X_ytrain_s.shape[1])
    y_hist = y_model.fit(
        X_ytrain_s, y_ytrain_s,
        validation_data=(X_yval_s, y_yval_s),
        epochs=80,
        batch_size=32,
        verbose=0,
    )

    y_model.save(os.path.join(MODELS_DIR, "yearwise_regression_model.keras"))
    joblib.dump(y_fscaler, os.path.join(MODELS_DIR, "yearwise_regression_feature_scaler.joblib"))
    joblib.dump(tscaler, os.path.join(MODELS_DIR, "yearwise_regression_target_scaler.joblib"))
    save_loss_plot(y_hist, os.path.join(FIG_DIR, "reg_loss_vs_epochs_yearwise.png"), title="Regressor Loss vs Epochs (Year-wise split)")

    return feat_names

def main():
    np.random.seed(RANDOM_SEED)

    temp_path = os.path.join(DATA_DIR, "Amazon_temperature_student.csv")
    thr_path = os.path.join(DATA_DIR, "thresholds.csv")

    df = load_temperature_data(temp_path)
    thr = load_thresholds(thr_path)
    df = add_hot_label(df, thr)

    train_classifier(df)
    train_regressors(df)

    print("Done. Saved models to ./models and figures to ./figures")

if __name__ == "__main__":
    main()
