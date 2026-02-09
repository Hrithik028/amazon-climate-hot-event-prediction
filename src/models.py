from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_classifier(input_dim: int) -> keras.Model:
    """Binary classifier for Hot events."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_regressor(input_dim: int) -> keras.Model:
    """Regressor for temperature."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mae",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
