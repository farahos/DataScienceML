import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib


# Load processed dataset
agg = pd.read_csv('dataset/processed/product_revenue_dataset.csv')


# Helpers: sign-preserving log1p and inverse
def signed_log1p(x):
    a = np.asarray(x, dtype=float)
    return np.sign(a) * np.log1p(np.abs(a))


def signed_expm1(x):
    a = np.asarray(x, dtype=float)
    return np.sign(a) * (np.expm1(np.abs(a)))


# Create log-features for revenue inputs
for col in ('NetRevenue', 'NetRevenue_LastMonth', 'NetRevenue_MA3'):
    if col not in agg.columns:
        raise KeyError(f"Missing expected column: {col}")
    agg[col + '_log1p'] = signed_log1p(agg[col].values)

features = [
    'NetRevenue_log1p', 'NetRevenue_LastMonth_log1p', 'NetRevenue_MA3_log1p', 'Month', 'ProductFrequency'
]

# Inputs and targets
X = agg[features]
y = agg['NextMonthRevenue'].values
y_log = signed_log1p(y)

# Train-test split (single split used for both original and target-log models)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, _, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Scale features (X only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train helper: performs GridSearch and returns best estimator and predictions
def train_and_predict(model_cls, params, Xtr, ytr, Xte):
    grid = GridSearchCV(model_cls, params, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    preds = best.predict(Xte)
    return best, preds


# Training: baseline models (predict original target)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
xgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 7]}

rf_best, rf_pred = train_and_predict(RandomForestRegressor(random_state=42), rf_params, X_train_scaled, y_train, X_test_scaled)
xgb_best, xgb_pred = train_and_predict(XGBRegressor(objective='reg:squarederror', random_state=42), xgb_params, X_train_scaled, y_train, X_test_scaled)


# Training: target-log models (predict log(target))
lr_t = LinearRegression()
lr_t.fit(X_train_scaled, y_train_log)
lr_t_pred_log = lr_t.predict(X_test_scaled)

rf_best_t, rf_t_pred_log = train_and_predict(RandomForestRegressor(random_state=42), rf_params, X_train_scaled, y_train_log, X_test_scaled)
xgb_best_t, xgb_t_pred_log = train_and_predict(XGBRegressor(objective='reg:squarederror', random_state=42), xgb_params, X_train_scaled, y_train_log, X_test_scaled)

# Inverse log predictions back to original scale for evaluation
lr_t_pred = signed_expm1(lr_t_pred_log)
rf_t_pred = signed_expm1(rf_t_pred_log)
xgb_t_pred = signed_expm1(xgb_t_pred_log)


# Evaluation helper
def evaluate_model(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'Model': name, 'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}


# Compare baseline vs target-log (on original scale)
results = []
results.append(evaluate_model('Linear Regression', y_test, lr_pred))
results.append(evaluate_model('Random Forest', y_test, rf_pred))
results.append(evaluate_model('XGBoost', y_test, xgb_pred))

results.append(evaluate_model('Linear Regression (tgtlog)', y_test, lr_t_pred))
results.append(evaluate_model('Random Forest (tgtlog)', y_test, rf_t_pred))
results.append(evaluate_model('XGBoost (tgtlog)', y_test, xgb_t_pred))

results_df = pd.DataFrame(results)
results_df.to_csv('regression_results/model_performance_compare.csv', index=False)
print('\nModel performance comparison saved to regression_results/model_performance_compare.csv')
print(results_df)


# Overwrite original model filenames with target-log versions (API will use these)
os.makedirs('models', exist_ok=True)
joblib.dump(rf_best_t, 'models/random_forest_regressor.joblib')
joblib.dump(xgb_best_t, 'models/xgboost_regressor.joblib')
joblib.dump(scaler, 'models/regression_scaler.pkl')
joblib.dump(lr_t, 'models/linear_regression.joblib')
print('\n✅ Overwrote models/regression scaler with target-log trained artifacts.')


# Sanity check (single-row)
i = 3
x_one = X_test_scaled[[i]]
print(f"\nSanity check (row {i}): actual={y_test[i]:,.2f}")
print(f"  Baseline RF : {rf_best.predict(x_one)[0]:,.2f}")
print(f"  Target-log RF (inv): {signed_expm1(rf_best_t.predict(x_one))[0]:,.2f}")
