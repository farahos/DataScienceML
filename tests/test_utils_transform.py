import sys
import os
import numpy as np
import pytest

# ensure project root is importable for pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import prepare_regression_features_from_json, reg_scaler


def test_prepare_regression_features_shape():
    sample = {
        "NetRevenue": 100.0,
        "NetRevenue_LastMonth": 90.0,
        "NetRevenue_MA3": 95.0,
        "Month": 5,
        "ProductFrequency": 2
    }
    arr = prepare_regression_features_from_json(sample)
    assert arr.ndim == 2
    # number of features should equal scaler expected features if available
    try:
        expected = len(reg_scaler.feature_names_in_)
        assert arr.shape[1] == expected
    except Exception:
        # fallback: expect 5 features
        assert arr.shape[1] == 5


def test_prepare_regression_values_consistent():
    sample = {
        "NetRevenue": -12.5,
        "NetRevenue_LastMonth": -10.0,
        "NetRevenue_MA3": -11.0,
        "Month": 6,
        "ProductFrequency": 3
    }
    arr1 = prepare_regression_features_from_json(sample)
    # calling twice should produce same scaled array
    arr2 = prepare_regression_features_from_json(sample)
    assert np.allclose(arr1, arr2)
