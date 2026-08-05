import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Keep in sync with app/train/eval FEATURES order
FEATURES = [
    'temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'precipitation',
    'wind_speed_10m', 'surface_pressure', 'uv_index', 'dew_point_2m', 'pm10'
]


def add_features(X):
    """Feature engineering on base 9 columns in FEATURES order.
    Returns an augmented array with extra nonlinear/interaction terms.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # Columns by index for readability
    T = X[:, 0]   # temperature_2m
    RH = X[:, 1]  # relative_humidity_2m
    AT = X[:, 2]  # apparent_temperature
    PR = X[:, 3]  # precipitation
    WS = X[:, 4]  # wind_speed_10m
    SP = X[:, 5]  # surface_pressure
    UV = X[:, 6]  # uv_index
    DP = X[:, 7]  # dew_point_2m
    P10 = X[:, 8] # pm10

    temp_dew_spread = T - DP
    rh_pm10 = RH * P10
    wind_inv = 1.0 / (1.0 + np.maximum(0.0, WS))
    precip_bin = (PR > 0).astype(float)
    app_minus_temp = AT - T
    uv2 = UV * UV
    log_pm10 = np.log1p(np.maximum(0.0, P10))
    press_center = SP - 1013.0

    extra = np.column_stack([
        temp_dew_spread,
        rh_pm10,
        wind_inv,
        precip_bin,
        app_minus_temp,
        uv2,
        log_pm10,
        press_center,
    ])
    return np.column_stack([X, extra])


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that augments base FEATURES with engineered ones."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_features(X)
