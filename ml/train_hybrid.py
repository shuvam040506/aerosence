"""
Train a hybrid PM2.5 regressor (RandomForest + ExtraTrees average).

This reinstates the original concept (two diverse tree ensembles) instead of
the previous single HistGradientBoosting model. We keep a lightweight
`TransformedTargetRegressor` (log1p/expm1) around the hybrid for improved
stability on skewed PM2.5 targets.

Modes:
1) External CSV (recommended):
    python ml/train_hybrid.py --csv path/to/data.csv --target pm25
    The CSV should contain the 9 input features and a PM2.5 target column.
    Column names can be auto-detected from common aliases (see below).

2) Synthetic (fallback if --csv is not provided):
    Generates a plausible synthetic dataset to keep this project self-contained.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score
try:
    import pandas as pd
except Exception:
    pd = None

# Ensure project root is on sys.path so 'ml' is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features import FeatureAdder, FEATURES

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# FEATURES imported from ml.features

# Common aliases to auto-detect columns
ALIASES = {
    'temperature_2m': ['temperature_2m', 'temperature', 'temp', 'temp_c', 't2m'],
    'relative_humidity_2m': ['relative_humidity_2m', 'relative_humidity', 'humidity', 'rh'],
    'apparent_temperature': ['apparent_temperature', 'feels_like', 'apparent_temp', 'feelslike'],
    'precipitation': ['precipitation', 'precip', 'prcp'],
    'wind_speed_10m': ['wind_speed_10m', 'wind_speed', 'wind', 'ws10'],
    'surface_pressure': ['surface_pressure', 'pressure', 'msl_pressure', 'press'],
    'uv_index': ['uv_index', 'uv', 'uvi'],
    'dew_point_2m': ['dew_point_2m', 'dew_point', 'dewpoint', 'td'],
    'pm10': ['pm10', 'PM10', 'pm_10']
}

TARGET_ALIASES = ['pm25', 'PM2_5', 'PM2.5', 'PM25', 'pm2_5', 'pm_2_5']


# add_features imported from ml.features


class HybridForest:
    """Average predictions of a RandomForest and an ExtraTrees model.

    Kept intentionally simple (no meta-learner) for robustness & quick training.
    Exposes sklearn-like fit/predict so it can live inside other wrappers.
    """
    def __init__(self,
                 n_estimators_rf=300,
                 n_estimators_et=300,
                 random_state=42,
                 max_depth=None,
                 n_jobs=-1):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators_rf,
            random_state=random_state,
            n_jobs=n_jobs,
            max_depth=max_depth,
            oob_score=False,
        )
        self.et = ExtraTreesRegressor(
            n_estimators=n_estimators_et,
            random_state=random_state + 17 if random_state is not None else None,
            n_jobs=n_jobs,
            max_depth=max_depth,
        )
        # Store top-level params explicitly for sklearn's cloning
        self.n_estimators_rf = n_estimators_rf
        self.n_estimators_et = n_estimators_et
        self.random_state = random_state
        self.max_depth = max_depth
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.et.fit(X, y)
        return self

    def predict(self, X):
        prf = self.rf.predict(X)
        pet = self.et.predict(X)
        return 0.5 * (np.asarray(prf) + np.asarray(pet))

    # --- sklearn estimator API helpers ---
    def get_params(self, deep=True):
        params = {
            'n_estimators_rf': self.n_estimators_rf,
            'n_estimators_et': self.n_estimators_et,
            'random_state': self.random_state,
            'max_depth': self.max_depth,
            'n_jobs': self.n_jobs,
        }
        if deep:
            # include child estimator params namespaced
            params.update({f'rf__{k}': v for k, v in self.rf.get_params(deep=True).items()})
            params.update({f'et__{k}': v for k, v in self.et.get_params(deep=True).items()})
        return params

    def set_params(self, **params):
        # Top-level params
        for p in ['n_estimators_rf', 'n_estimators_et', 'random_state', 'max_depth', 'n_jobs']:
            if p in params:
                setattr(self, p, params[p])
        # Rebuild base estimators if structural params changed
        self.rf.set_params(**{k[4:]: v for k, v in params.items() if k.startswith('rf__')})
        self.et.set_params(**{k[4:]: v for k, v in params.items() if k.startswith('et__')})
        return self


def build_model():
    hybrid = HybridForest(random_state=RANDOM_SEED)
    # Log-transform the target to stabilize variance / heavy tails
    ttr = TransformedTargetRegressor(
        regressor=hybrid,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    return Pipeline([
        ('feat', FeatureAdder()),
        ('reg', ttr),
    ])


def autodetect_columns(df, target_name=None):
    mapping = {}
    cols = {c.lower(): c for c in df.columns}
    # features
    for feat in FEATURES:
        found = None
        for ali in ALIASES.get(feat, []):
            if ali.lower() in cols:
                found = cols[ali.lower()]
                break
        if not found and feat.lower() in cols:
            found = cols[feat.lower()]
        if not found:
            raise ValueError(f"Missing required feature column for '{feat}'. Tried aliases: {ALIASES.get(feat)}")
        mapping[feat] = found
    # target
    tgt = None
    if target_name:
        # use provided target if exists
        if target_name in df.columns:
            tgt = target_name
        elif target_name.lower() in cols:
            tgt = cols[target_name.lower()]
        else:
            raise ValueError(f"Target column '{target_name}' not found in CSV.")
    else:
        for cand in TARGET_ALIASES:
            if cand.lower() in cols:
                tgt = cols[cand.lower()]
                break
        if not tgt:
            raise ValueError("Could not autodetect target column for PM2.5. Provide --target.")
    return mapping, tgt


def load_external_csv(path, target=None, limit=None):
    if pd is None:
        raise RuntimeError("pandas is required to load CSVs. Add 'pandas' to requirements and install it.")
    df = pd.read_csv(path)
    mapping, tgt = autodetect_columns(df, target_name=target)
    use_cols = [mapping[f] for f in FEATURES] + [tgt]
    df = df[use_cols].dropna()
    if limit and limit > 0:
        df = df.sample(n=min(limit, len(df)), random_state=RANDOM_SEED)
    X = df[[mapping[f] for f in FEATURES]].to_numpy(dtype=float)
    y = df[tgt].to_numpy(dtype=float)
    print('[columns]', {f: mapping[f] for f in FEATURES}, ' target:', tgt)
    return X, y


def generate_synthetic(N=4000):
    temp = np.random.normal(18, 10, N)           # °C
    rh = np.clip(np.random.normal(60, 20, N), 5, 100)    # %
    app_temp = temp + np.random.normal(0, 2, N)  # feels-like
    precip = np.abs(np.random.normal(0.8, 1.2, N))
    wind = np.abs(np.random.normal(4, 2.5, N))
    press = np.random.normal(1013, 10, N)
    uv = np.clip(np.random.normal(4, 2, N), 0, 11)
    dew = temp - ((100 - rh) / 5) + np.random.normal(0, 1, N)
    pm10 = np.abs(np.random.normal(35, 25, N))

    true_pm25 = 0.45 * pm10 * (1 + (rh - 50) / 200) * (1 - np.tanh(precip / 10)) * (1 - np.tanh(wind / 8))
    true_pm25 += 0.015 * np.maximum(0, 20 - temp)  # colder inversions
    true_pm25 = np.maximum(0, true_pm25)
    noise = np.random.normal(0, 5, N)
    y = np.maximum(0, true_pm25 + noise)

    X = np.column_stack([temp, rh, app_temp, precip, wind, press, uv, dew, pm10])
    return X, y


def main():
    parser = argparse.ArgumentParser(description='Train hybrid PM2.5 model')
    parser.add_argument('--csv', type=str, help='Path to CSV with features and pm25 target')
    parser.add_argument('--target', type=str, help='Target column name (PM2.5). Autodetected if not provided')
    parser.add_argument('--limit', type=int, default=0, help='Optional row limit for quick runs')
    args = parser.parse_args()

    if args.csv:
        X, y = load_external_csv(args.csv, target=args.target, limit=args.limit)
    else:
        X, y = generate_synthetic()

    model = build_model()
    # Train/valid split
    N = len(X)
    idx = np.arange(N)
    np.random.shuffle(idx)
    train = idx[: int(0.8 * N)]
    valid = idx[int(0.8 * N):]

    model.fit(X[train], y[train])
    pred = model.predict(X[valid])
    r2 = float(r2_score(y[valid], pred))
    print('R2:', r2)

    here = Path(__file__).parent
    joblib.dump(model, here / 'pm25_pipeline.pkl')
    print('Saved pipeline to:', here / 'pm25_pipeline.pkl')

    # Persist simple metrics for inspection
    try:
        import json
        with open(here / 'last_metrics.json', 'w', encoding='utf-8') as f:
            json.dump({'r2': r2, 'n_train': int(len(train)), 'n_valid': int(len(valid))}, f)
        print('Saved metrics to:', here / 'last_metrics.json')
    except Exception as _:
        pass


if __name__ == '__main__':
    main()
