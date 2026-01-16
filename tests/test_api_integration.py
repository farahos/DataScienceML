import sys
import os
import json

# ensure project root is importable for pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app


def test_predict_all_endpoint():
    client = app.test_client()
    payload = {
        "ProductGroup": "X",
        "NetRevenue": -12.5,
        "NetQuantity": 5,
        "NumTransactions": 2,
        "NumUniqueCustomers": 2,
        "NetRevenue_LastMonth": -10.0,
        "NetRevenue_MA3": -11.0,
        "Month": 6,
        "ProductFrequency": 3
    }
    resp = client.post('/predict_all?debug=1', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    # Expect certain keys
    assert 'next_month_revenue' in data
    assert 'models' in data
    assert 'debug' in data
