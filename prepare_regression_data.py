import pandas as pd

# Load raw cleaned grouping CSV
df = pd.read_csv('cleaned_product_grouping.csv')

# Ensure InvoiceDate is datetime and extract Year/Month
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month

# Aggregate per product (StockCode) by Year/Month
agg = df.groupby(['StockCode', 'Year', 'Month']).agg({
    'NetRevenue': 'sum',
    'NetQuantity': 'sum',
    'CustomerFrequency': 'mean',
    'ProductFrequency': 'mean'
}).reset_index()

# Create lag and moving-average features
agg['NetRevenue_LastMonth'] = agg.groupby('StockCode')['NetRevenue'].shift(1)
agg['NetRevenue_MA3'] = agg.groupby('StockCode')['NetRevenue'].rolling(3).mean().shift(1).reset_index(0, drop=True)
agg['NextMonthRevenue'] = agg.groupby('StockCode')['NetRevenue'].shift(-1)

# Drop rows with NA introduced by shifting/rolling
agg = agg.dropna()

# Strip any accidental whitespace in column names to ensure consistency
agg.columns = [c.strip() for c in agg.columns]

# Save processed dataset
out_path = 'dataset/processed/product_revenue_dataset.csv'
agg.to_csv(out_path, index=False)
print(f"✅ Aggregated dataset saved as '{out_path}'")