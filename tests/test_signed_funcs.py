import sys
import os
import math
import numpy as np

# ensure project root is importable for pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import signed_log1p, signed_expm1


def test_signed_log1p_positive():
    x = np.array([0.0, 1.0, 10.0])
    y = signed_log1p(x)
    assert y[0] == 0.0
    assert np.isclose(y[1], math.log1p(1.0))
    assert np.isclose(y[2], math.log1p(10.0))


def test_signed_log1p_negative():
    x = np.array([-1.0, -5.5])
    y = signed_log1p(x)
    assert np.isclose(y[0], -math.log1p(1.0))
    assert np.isclose(y[1], -math.log1p(5.5))


def test_signed_expm1_inverse():
    arr = np.array([-2.5, -1.0, 0.0, 1.0, 3.2])
    inv = signed_expm1(arr)
    # inverse of signed_log1p should approximately recover magnitude ordering
    recov = signed_log1p(inv)
    assert np.allclose(recov, arr, atol=1e-8)
