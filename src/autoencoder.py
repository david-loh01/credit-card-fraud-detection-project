from typing import Dict, Tuple

import numpy as np
from keras.layers import Input, Dense
from keras.models import Sequential
from sklearn import preprocessing


def scale_features_minmax(X: np.ndarray) -> np.ndarray:
    """MinMax scale features to [0, 1]."""
    return preprocessing.MinMaxScaler().fit_transform(X)


def split_normal_fraud(
    X_scaled: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split scaled features into normal (0) and fraud (1)."""
    X_norm = X_scaled[y == 0]
    X_fraud = X_scaled[y == 1]
    return X_norm, X_fraud


def build_autoencoder(input_dim: int):
    """Construct the autoencoder architecture."""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(100, activation="tanh"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(50, activation="tanh"))
    model.add(Dense(100, activation="tanh"))
    model.add(Dense(input_dim, activation="tanh"))
    model.compile(optimizer="adadelta", loss="mean_squared_error")
    return model


def train_autoencoder(
    X_norm: np.ndarray,
    batch_size: int = 10,
    epochs: int = 50,
    validation_split: float = 0.2,
):
    """Train autoencoder on normal data only."""
    model = build_autoencoder(X_norm.shape[1])
    history = model.fit(
        X_norm,
        X_norm,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split=validation_split,
    )
    return model, history


def build_encoder(autoencoder) -> Sequential:
    """
    Build an encoder model by taking the first two layers
    of the trained autoencoder (as in your notebook).
    """
    return Sequential(autoencoder.layers[:2])


def latent_representations(
    encoder,
    X_norm: np.ndarray,
    X_fraud: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute latent representations and reconstructed labels
    (0 for normal, 1 for fraud).
    """
    norm_repr = encoder.predict(X_norm)
    fraud_repr = encoder.predict(X_fraud)

    X_repr = np.append(norm_repr, fraud_repr, axis=0)
    y_norm = np.zeros(norm_repr.shape[0])
    y_fraud = np.ones(fraud_repr.shape[0])
    y_repr = np.append(y_norm, y_fraud)

    return X_repr, y_repr, norm_repr
