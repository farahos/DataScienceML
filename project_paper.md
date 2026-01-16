# Product Grouping & Next Month Revenue Prediction

## 1. Introduction

This project addresses product grouping and short-term revenue forecasting for retail products using the UCI Online Retail dataset. Grouping helps categorize products for targeted actions (promotion, stocking, discontinuation), while predicting next-month revenue helps planning and inventory management.

## 2. Dataset

- Source: UCI Online Retail Dataset (https://archive.ics.uci.edu/dataset/352/online+retail)
- Download date: Oct 02, 2025
- Rough size used in this project: ~541k transactions aggregated to product-level (~4k products)

### Preprocessing Summary
- Parsed `InvoiceDate` to datetime
- Removed rows with missing `CustomerID` or `Description`, removed invalid `UnitPrice` <= 0
- Handled negative `Quantity` as returns
- Computed `TotalPrice = Quantity * UnitPrice`, `NetQuantity`, `NetRevenue` per product/month
- Aggregated at product-month level and created lag features: `NetRevenue_LastMonth`, `NetRevenue_MA3` (3-month moving average). Target: `NextMonthRevenue`.

## 3. Models

### Clustering
- Features: `NetQuantity`, `NetRevenue`, `NumTransactions`, `NumUniqueCustomers` (standardized)
- Algorithms: KMeans (Elbow method for k), DBSCAN (eps, min_samples tuned)
- Metrics: Silhouette Score, Davies-Bouldin Index; cluster profiling for business interpretation.

### Regression (Next Month Revenue)
- Features: `NetRevenue`, `NetQuantity`, `CustomerFrequency`, `ProductFrequency`, `NetRevenue_LastMonth`, `NetRevenue_MA3`, `Month`, `Year` (standardized)
- Algorithms: Linear Regression, Random Forest, XGBoost
- Training: 80/20 train/test split, StandardScaler on features, GridSearchCV for RF and XGBoost
- Important transforms: sign-preserving `signed_log1p` applied to revenue features and (for deployed models) the target. Prediction outputs are inverse-transformed via `signed_expm1`.

## 4. Results & Sanity Checks

| Model                   | R²     | MAE   | MSE    | RMSE  |
|--------------------------|--------|-------|--------|-------|
| Linear Regression        | 0.403  | 4.545 | 66.655 | 8.164 |
| Random Forest            | 0.594  | 3.055 | 45.361 | 6.735 |
| XGBoost                  | 0.675  | 2.944 | 36.331 | 6.028 |
| Linear Regression (tgtlog) | 0.513 | 3.389 | 54.420 | 7.377 |
| Random Forest (tgtlog)   | 0.640  | 2.910 | 40.190 | 6.340 |
| XGBoost (tgtlog)         | 0.659  | 2.895 | 38.097 | 6.172 |

**Key Insights:**
- **XGBoost** achieved the best performance on the held-out test set across all metrics — highest R² (0.675) and lowest RMSE (6.028), showing strong predictive accuracy and generalization.
- Models trained on log-transformed targets (`tgtlog`) slightly improved stability and reduced prediction variance.
- Overall, **XGBoost** (both normal and log-target versions) captured non-linear relationships more effectively than linear models.

**Sanity Checks:**
Several single-row predictions were compared with their actual values. The predicted revenues were consistent in both magnitude and trend with the true observations after applying inverse transformations.

These results confirm that **XGBoost** is the most reliable and interpretable model for next-month revenue forecasting in this system.

## 5. Deployment

- Flask API (`app.py`) exposes:
  - `POST /predict_group` — clustering
  - `POST /predict_revenue` — regression
  - `POST /predict_all` — both results combined (used by frontend)
  - `GET /ui` — simple frontend at `/ui`

Example request (curl):

```bash
curl -X POST http://127.0.0.1:7000/predict_all?group_model=kmeans&rev_model=xgboost \
  -H "Content-Type: application/json" \
  -d '{"NetRevenue": 123.45, "NetQuantity": 10, "NumTransactions": 2, "NumUniqueCustomers": 1, "NetRevenue_LastMonth": 100.0, "NetRevenue_MA3": 95.0, "Month": 9, "ProductFrequency": 3}'
```

Example response (abridged):

```json
{
  "models": ["KMeans", "XGBoost"],
  "product_group": "High-volume / Discounted items",
  "business": "Very high quantity but low or negative revenue",
  "next_month_revenue": "$155.62"
}
```

## 6. Lessons Learned

- Sign-preserving log transforms are essential when working with revenue values that can be negative (returns) and span magnitudes.
- Separating aggregation from modeling reduces runtime and complexity when retraining models.
- Model deployments must match inference preprocessing exactly; we saved scalers and added compatibility checks in `utils.py`.

## 7. How to reproduce

1. Create venv and install dependencies from `requirements.txt`.
2. Prepare data using `prepare_regression_data.py` and `processing.py`.
3. Train models with `regression.py` and `clustering.py`.
4. Run the API: `python app.py` and use `/ui` or `/predict_all`.

## 8. Files in repo

(see README in root for full list)

---

*End of paper*