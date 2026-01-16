import numpy as np
import joblib
import pandas as pd

CLUSTER_FEATURES = ["NetQuantity", "NetRevenue", "NumTransactions", "NumUniqueCustomers"]

# Load scalers
SCALER = joblib.load("models/clustering_scaler.pkl")
reg_scaler = joblib.load("models/regression_scaler.pkl")
try:
    reg_scaler_tgt = joblib.load("models/regression_scaler_tgtlog1p.pkl")
except Exception:
    reg_scaler_tgt = None


def signed_log1p(x):
    """Sign-preserving log1p: sign(x) * log1p(|x|)."""
    a = np.asarray(x, dtype=float)
    return np.sign(a) * np.log1p(np.abs(a))


def signed_expm1(x):
    """Inverse of signed_log1p: sign(y) * expm1(|y|)."""
    a = np.asarray(x, dtype=float)
    return np.sign(a) * (np.expm1(np.abs(a)))


def prepare_features_from_json(record: dict) -> np.ndarray:
    """Return scaled clustering features as a 2D numpy array."""
    values = {f: float(record.get(f, 0.0)) for f in CLUSTER_FEATURES}
    df = pd.DataFrame([values], columns=CLUSTER_FEATURES)
    return SCALER.transform(df)


def prepare_regression_features_from_json(record: dict) -> np.ndarray:
    """Prepare regression features (apply signed log to revenues and scale).

    The scaler expects revenue columns suffixed with `_log1p`, so we rename
    them after applying the signed transform.
    """
    features = ['NetRevenue', 'NetRevenue_LastMonth', 'NetRevenue_MA3', 'Month', 'ProductFrequency']
    values = {f: float(record.get(f, 0.0)) for f in features}
    df = pd.DataFrame([values], columns=features)

    for c in ['NetRevenue', 'NetRevenue_LastMonth', 'NetRevenue_MA3']:
        df[c] = signed_log1p(df[c].values)

    df = df.rename(columns={
        'NetRevenue': 'NetRevenue_log1p',
        'NetRevenue_LastMonth': 'NetRevenue_LastMonth_log1p',
        'NetRevenue_MA3': 'NetRevenue_MA3_log1p'
    })

    try:
        feature_order = list(reg_scaler.feature_names_in_)
        df = df[feature_order]
    except Exception:
        pass

    return reg_scaler.transform(df)