# Reflection — Product Grouping & Next‑Month Revenue Prediction (Final)

## Introduction

This project tackled two practical problems often faced by retail analytics teams:
- Product grouping: cluster products by sales/return behavior to improve assortment, pricing and promotion decisions.
- Short‑term forecasting: predict each product's next‑month net revenue to support inventory planning and margin management.

Both problems matter because better segmentation and accurate short‑horizon forecasts reduce stockouts, improve marketing ROI, and protect margins when returns or discounts are common.

This final reflection summarizes the dataset, preprocessing, modeling experiments, test results, deployment, and lessons learned. It combines earlier notes and recent engineering updates into a concise 3–5 page narrative suitable for submission.

## Dataset and Aggregation

- Source: UCI Online Retail (public). Raw dataset contains ≈541k transactions covering ≈4.3k unique products (StockCode).
- Core fields used: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.

Aggregation strategy:
- Clustering input: one row per StockCode with aggregated sums and counts (NetRevenue, NetQuantity, NumTransactions, NumUniqueCustomers, ProductFrequency).
- Regression input: product×month granularity (StockCode, Year, Month) with NetRevenue, NetQuantity and lag features (LastMonth and MA3). Target is NextMonthRevenue (shifted −1).

Preprocessing highlights:
- Cleaned InvoiceDate, removed entries with UnitPrice ≤ 0 or missing critical fields, and dropped duplicates.
- Handled quantity/unit price extremes with IQR capping instead of removal to preserve data coverage.
- Engineered frequency features: unique invoices per customer and per product; computed NetRevenue sales/returns breakdowns to capture negative revenues from returns.

Persistence: processed CSVs, scaler objects and model artifacts are saved under `dataset/processed/`, `models/`, and result images under `regression_results/` and `clustering_results/` for reproducibility.

## Feature Engineering and Scaling

- Row-level features: TotalPrice, LogTotalPrice (when positive), NetQuantity, NetRevenue, Sales/Returns breakdowns.
- Lag features for regression: NetRevenue_LastMonth (t−1) and NetRevenue_MA3 (3‑month moving average, shifted).
- Categorical encoding: Month used as numeric cyclical indicator where helpful (kept as integer for tree models).
- Scaling: numeric features standardized using `StandardScaler` fit on the training split. Scalers are saved to `models/` and loaded at inference.

Special handling for negatives: returned products create negative NetRevenue values. Applying plain log1p would produce NaNs. To solve this, I implemented a sign‑preserving transform:
- signed_log1p(x) = sign(x) * log1p(|x|)
- signed_expm1(y) = sign(y) * expm1(|y|)

This allowed log‑like variance stabilization while preserving sign information for both features and the target. The final deployed regression models were trained on the signed target (`NextMonthRevenue` transformed) and predictions are inverse‑transformed at inference.

## Models and Training

Algorithms tried (representative two explained here):

1) KMeans (clustering)
- Goal: partition products into business‑meaningful groups (e.g., high‑volume discounted, best‑sellers, niche low‑volume).
- Input: standardized NetQuantity, NetRevenue (signed log where appropriate), NumTransactions, NumUniqueCustomers.
- Selection: elbow method (inertia) for k, validated with Silhouette Score and Davies‑Bouldin Index. Clusters profiled by feature means and example SKUs.

2) XGBoost (regression)
- Goal: predict NextMonthRevenue at product×month level.
- Inputs: scaled NetRevenue (current, last month, MA3), Month, ProductFrequency, NetQuantity, NumTransactions.
- Training: 80/20 train/test split; grid search (n_estimators, learning_rate, max_depth) for hyperparameter tuning; StandardScaler fit on training features. The target was transformed with signed_log1p for stability; models then invert predictions with signed_expm1.

Baselines: Random Forest and Linear Regression were trained for comparison. Models saved under `models/` (regression_scaler.pkl, random_forest_regressor.joblib, xgboost_regressor.joblib). To keep API backward compatible the final artifacts were saved using the original filenames while inference code performs the inverse transform.

## Results & Sanity Checks

Final test set metrics (held‑out test split) — exact values from the final evaluation:

| Model                       | R²    | MAE   | MSE    | RMSE  |
|-----------------------------|-------:|------:|-------:|------:|
| Linear Regression           | 0.403  | 4.545 | 66.655 | 8.164 |
| Random Forest               | 0.594  | 3.055 | 45.361 | 6.735 |
| XGBoost                     | 0.675  | 2.944 | 36.331 | 6.028 |
| Linear Regression (tgtlog)  | 0.513  | 3.389 | 54.420 | 7.377 |
| Random Forest (tgtlog)      | 0.640  | 2.910 | 40.190 | 6.340 |
| XGBoost (tgtlog)            | 0.659  | 2.895 | 38.097 | 6.172 |

Interpretation:
- XGBoost (no target transform) achieved the highest R² and lowest RMSE on the held‑out test set in final reported numbers. Target‑log variants (trained on signed_log transformed target) improved linear model performance substantially and brought tree models closer together; these variants demonstrate that variance‑stabilizing transforms can help simpler models but tree ensembles already capture nonlinear structure well.

Sanity checks performed:
- Visual inspection of actual vs predicted plots for all models (`regression_results/actual_vs_predicted_*.png`).
- Single‑row prediction smoke tests against the Flask API (`/predict_all?debug=1`) to verify inputs/outputs, scaled arrays and inverse‑transformed predictions.
- Random cluster samples inspected manually for business interpretability (example SKUs per cluster printed during analysis).

## Deployment (API & UI)

- App: Flask app (`app.py`) exposing endpoints used during testing and for the UI.
- Key endpoints:
  - GET `/ui` — HTML form UI (WTForms + Jinja2) at `templates/forms.html` for manual inputs and visualization.
  - POST `/predict_group?model=kmeans|dbscan` — returns cluster label and business text for given numeric features.
  - POST `/predict_revenue?model=random_forest|xgboost` — returns next‑month revenue prediction.
  - POST `/predict_all?group_model=...&rev_model=...` — returns both clustering and regression outputs in one JSON payload.

Inference contract and compatibility:
- `utils.py` prepares DataFrames with named columns, applies `signed_log1p` to revenue fields, renames revenue columns to the `_log1p` suffix expected by the saved scaler, reorders columns to match `reg_scaler.feature_names_in_` when available, and returns scaled arrays for model inputs.
- `app.py` loads scalers and model artifacts from `models/`, calls model.predict, and inverse‑transforms regression predictions with `signed_expm1` so frontend and API consumers always receive revenue in original scale.
- To avoid breaking older clients, the API now returns canonical JSON keys (`models`, `product_group`, `business`, `next_month_revenue`) while also populating compatibility/fallback keys used by earlier UI versions.

Example request (POST /predict_all with JSON body):

```json
{
  "NetRevenue": 130.33,
  "NetQuantity": 1000,
  "NumTransactions": 8,
  "NumUniqueCustomers": 5,
  "NetRevenue_LastMonth": 290,
  "NetRevenue_MA3": 1000,
  "Month": 10,
  "ProductFrequency": 17
}
```

Example abridged response:

```json
{
  "models": {"group_model": "kmeans", "rev_model": "xgboost"},
  "product_group": "High-volume / Discounted items",
  "business": "Verify pricing and promotions; high returns detected.",
  "next_month_revenue": 1234.56,
  "next_month_revenue_formatted": "$1,234.56"
}
```

Frontend notes:
- `templates/forms.html` was updated to prefer canonical keys while keeping fallbacks for older clients. Chart.js visualizations were updated to persist between predictions, and a responsive footer matching the header gradient was added for polish.

## Lessons Learned & Challenges

- Data quality: missing CustomerID and invalid UnitPrice required conservative cleaning and IQR capping to preserve useful patterns while reducing noise.
- Negative revenues from returns: naive log transforms break on negative values. The sign‑preserving signed_log1p/signed_expm1 pattern worked well and is a low‑risk way to stabilize variance while preserving sign.
- Inference reproducibility: saving scaler and model artifacts separately can cause feature ordering and name mismatches. Building DataFrames with named columns and reordering to `feature_names_in_` fixes this; an even better long‑term fix is to persist a single sklearn Pipeline or TransformedTargetRegressor.
- API/UI compatibility: small naming mismatches broke the frontend during early testing. Returning canonical keys plus compatibility fallbacks kept the API stable for new and legacy clients.
- Testing: adding unit tests for transforms and a `/predict_all` smoke test dramatically reduced debugging cycles during deployment.

## Next Steps & Recommendations

1. Persist full preprocessing+estimator pipelines (scikit-learn Pipeline or TransformedTargetRegressor) to avoid future ordering mistakes and make versioning simpler.
2. Add CI checks that ensure required model files are present and unit tests run on push.
3. Expand features: holiday flags, promotion indicators, categorical embeddings (product/category), and time‑aware cross‑validation (time folds) for more robust forecasting.
4. Consider LightGBM/XGBoost with time folds or sequence models for longer horizon forecasting; explore calibration methods for models that occasionally predict negative revenue.

## Engineering notes (recent updates)

- Implemented `signed_log1p` and `signed_expm1` and applied them in `utils.py` and during model training/inference.
- Reordered DataFrame columns during preprocessing to match scaler expectations; saved scalers and models under `models/`.
- Updated `app.py` and `templates/forms.html` for API key compatibility and UI stability; added tests under `tests/` (unit + integration) and verified they pass locally.

---

This reflection summarizes technical choices, empirical outcomes and practical learnings. The project now provides end‑to‑end artifacts (processed datasets, trained models, plots, API and UI) ready for further iteration and production hardening.