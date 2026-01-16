# -----------------------------
# Imports and setup
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1.Load dataset
# -----------------------------
df = pd.read_csv('dataset/raw/online_retail.csv')
# === INITIAL SNAPSHOT ===
print("Initial data head:")
print(df.head())
print("Initial data info:")
print(df.info())
print("initial missing values per column:")
print(df.isnull().sum())
print("Initial data description:")
print(df.describe(include='all'))
print("Data types:")
print(df.dtypes)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')


# -----------------------------
# 2. Data Cleaning
# -----------------------------
# Start with a copy of the original dataframe
df_clean = df.copy()

# 2a). Replace 'Unspecified' in Country with NaN
df_clean["Country"] = df_clean["Country"].replace({"Unspecified": np.nan})

# 2b). Remove duplicates
df_clean = df_clean.drop_duplicates()
# print(f"Dropped duplicates: {df.shape} → {df_clean.shape}")

# 2c). Remove rows with missing UnitPrice
# Only remove rows with UnitPrice <= 0, do NOT filter Quantity here
# (so we can separate sales and returns later)
df_clean = df_clean[df_clean['UnitPrice'] > 0]

# 2d). Remove rows with missing Description
df_clean = df_clean[df_clean['Description'].notnull()]

# 2e). Remove rows with missing CustomerID
df_clean = df_clean[df_clean['CustomerID'].notnull()]

# 2f). Reset index
df_clean = df_clean.reset_index(drop=True)
# print("Data after cleaning head:")
# print(df_clean.head())

# 3. Outliers (IQR capping for Quantity and UnitPrice)
def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

low_qty, high_qty = iqr_bounds(df_clean["Quantity"])
low_price, high_price = iqr_bounds(df_clean["UnitPrice"])
df_clean["Quantity"] = df_clean["Quantity"].clip(lower=low_qty, upper=high_qty)
df_clean["UnitPrice"] = df_clean["UnitPrice"].clip(lower=low_price, upper=high_price)


# -----------------------------
# 4. One-hot encoding for Country
# -----------------------------
df_clean = pd.get_dummies(df_clean, columns=["Country"], drop_first=False, dtype="int")


# -----------------------------
# 5. Feature engineering
# -----------------------------
# 5a). TotalPrice and LogTotalPrice
df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]
# Compute LogTotalPrice, handle -inf/NaN
df_clean["LogTotalPrice"] = np.log1p(df_clean["TotalPrice"])
# Replace -inf, inf, NaN with 0 (or another value if preferred)
df_clean["LogTotalPrice"] = df_clean["LogTotalPrice"].replace([np.inf, -np.inf], 0)
df_clean["LogTotalPrice"] = df_clean["LogTotalPrice"].fillna(0)

# -----------------------------
# 5b). Separate Sales and Returns
# -----------------------------
df_clean['Sales'] = df_clean['Quantity'].apply(lambda x: x if x > 0 else 0)
df_clean['Returns'] = df_clean['Quantity'].apply(lambda x: abs(x) if x < 0 else 0)

df_clean['Revenue_Sales'] = df_clean['Sales'] * df_clean['UnitPrice']
df_clean['Revenue_Returns'] = df_clean['Returns'] * df_clean['UnitPrice']

# Optional: Net Sales (combined)
df_clean['NetQuantity'] = df_clean['Sales'] - df_clean['Returns']
df_clean['NetRevenue'] = df_clean['Revenue_Sales'] - df_clean['Revenue_Returns']

# 5c). Date features
df_clean["Year"] = df_clean["InvoiceDate"].dt.year
df_clean["Month"] = df_clean["InvoiceDate"].dt.month
df_clean["Day"] = df_clean["InvoiceDate"].dt.day

# 5d). Customer and product frequency features
df_clean["CustomerFrequency"] = df_clean.groupby("CustomerID")["InvoiceNo"].transform("nunique")
df_clean["ProductFrequency"] = df_clean.groupby("StockCode")["InvoiceNo"].transform("nunique")
# print("Data after feature engineering head:")
# print(df_clean.head())


# -----------------------------
# 6. Feature scaling (numeric features only; keep dummies unscaled)
# -----------------------------
numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns.to_list()
exclude = [c for c in df_clean.columns if c.startswith("Country_")]
num_features_to_scale = [c for c in numeric_cols if c not in exclude]
# Remove rows with NaN or Inf in features to be scaled
df_clean = df_clean[~df_clean[num_features_to_scale].isnull().any(axis=1)]
df_clean = df_clean[~df_clean[num_features_to_scale].isin([np.inf, -np.inf]).any(axis=1)]
scaler = StandardScaler()
df_clean[num_features_to_scale] = scaler.fit_transform(df_clean[num_features_to_scale])


# -----------------------------
# 7. Save cleaned dataset
# -----------------------------
df_clean.to_csv('dataset/raw/cleaned_product_grouping.csv', index=False)
print("Cleaned dataset saved as 'dataset/raw/cleaned_product_grouping.csv'.")