import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
# Ensure project root on sys.path for 'ml' package import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features import FeatureAdder, FEATURES

class HybridForest:
    def __init__(self, random_state=42, n_estimators_rf=300, n_estimators_et=300, max_depth=None, n_jobs=-1):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators_rf,
            random_state=random_state,
            n_jobs=n_jobs,
            max_depth=max_depth,
        )
        self.et = ExtraTreesRegressor(
            n_estimators=n_estimators_et,
            random_state=random_state + 17 if random_state is not None else None,
            n_jobs=n_jobs,
            max_depth=max_depth,
        )
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
        return 0.5 * (self.rf.predict(X) + self.et.predict(X))

    def get_params(self, deep=True):
        params = {
            'n_estimators_rf': self.n_estimators_rf,
            'n_estimators_et': self.n_estimators_et,
            'random_state': self.random_state,
            'max_depth': self.max_depth,
            'n_jobs': self.n_jobs,
        }
        if deep:
            params.update({f'rf__{k}': v for k, v in self.rf.get_params(deep=True).items()})
            params.update({f'et__{k}': v for k, v in self.et.get_params(deep=True).items()})
        return params

    def set_params(self, **params):
        for p in ['n_estimators_rf', 'n_estimators_et', 'random_state', 'max_depth', 'n_jobs']:
            if p in params:
                setattr(self, p, params[p])
        self.rf.set_params(**{k[4:]: v for k, v in params.items() if k.startswith('rf__')})
        self.et.set_params(**{k[4:]: v for k, v in params.items() if k.startswith('et__')})
        return self


def make_pipeline():
    hybrid = HybridForest()
    ttr = TransformedTargetRegressor(
        regressor=hybrid,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    return Pipeline([
        ('feat', FeatureAdder()),
        ('reg', ttr)
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=str(Path(__file__).parent.parent / 'data' / 'pm25_weather_kaggleish.csv'))
    ap.add_argument('--target', type=str, default='pm25')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--n_jobs_cv', type=int, default=1, help='Unused (kept for CLI compatibility)')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[args.target].to_numpy(dtype=float)

    cv = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    # Manual CV to avoid joblib/loky multiprocessing issues on Windows
    scores = []
    for train_idx, test_idx in cv.split(X):
        pipe = make_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        scores.append(r2_score(y[test_idx], pred))
    scores = np.array(scores, dtype=float)
    r2_mean = float(scores.mean())
    r2_std = float(scores.std())
    acc_pct = r2_mean * 100
    print(f'CV R2 mean={r2_mean:.4f}, std={r2_std:.4f}, folds={args.folds}, n={len(df)}')
    print(f'Accuracy (R²) = {acc_pct:.2f}%')

if __name__ == '__main__':
    main()
