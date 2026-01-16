import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from elbowK.elbow import find_best_k
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

# Load cleaned dataset (with Sales & Returns)
df = pd.read_csv('dataset/raw/cleaned_product_grouping.csv')
print("Cleaned data head:")
print(df.head())

# -----------------------------
# Aggregate to product-level features
# -----------------------------
df_products = df.groupby("StockCode").agg({
    "Description": "first",
    "Sales": "sum",
    "Returns": "sum",
    "NetQuantity": "sum",
    "Revenue_Sales": "sum",
    "Revenue_Returns": "sum",
    "NetRevenue": "sum",
    "InvoiceNo": "nunique",
    "CustomerID": "nunique"
}).reset_index()

df_products.rename(columns={
    "InvoiceNo": "NumTransactions",
    "CustomerID": "NumUniqueCustomers"
}, inplace=True)

print(df_products.head())
print(df_products.shape)

# -----------------------------
# Select features for clustering
# -----------------------------
FEATURES = ["NetQuantity", "NetRevenue", "NumTransactions", "NumUniqueCustomers"]
X = df_products[FEATURES].copy()
print("\nSelected features head:")
print(X.head())

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled shape:", X_scaled.shape)

# -----------------------------
# Elbow method to find optimal k
# -----------------------------
print("\n=== ELBOW METHOD ===")
best_k = find_best_k(X_scaled, max_k=10, save_plot=True)
print(f"Optimal k: {best_k}")
print("Elbow plot saved as 'clustering_results/elbow_plot.png'.")

# -----------------------------
# Fit K-Means with chosen k
# -----------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42)
df_products['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)
print("\nClustered data head:")
print(df_products.head())

# -----------------------------
# Fit DBSCAN with chosen k
# -----------------------------

dbscan = DBSCAN(eps=0.5, min_samples=5)
df_products['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)
print("\nClustered data head:")
print(df_products.head())

# -----------------------------
# Evaluate clustering
# -----------------------------
# KMeans evaluation
kmeans_sil = silhouette_score(X_scaled, df_products['KMeans_Cluster'])
kmeans_db = davies_bouldin_score(X_scaled, df_products['KMeans_Cluster'])
print(f"KMeans - Silhouette Score: {kmeans_sil:.4f}, Davies-Bouldin Index: {kmeans_db:.4f}")

# DBSCAN evaluation (only if more than 1 cluster found)
if len(set(df_products['DBSCAN_Cluster'])) > 1:
    dbscan_sil = silhouette_score(X_scaled, df_products['DBSCAN_Cluster'])
    dbscan_db = davies_bouldin_score(X_scaled, df_products['DBSCAN_Cluster'])
    print(f"DBSCAN - Silhouette Score: {dbscan_sil:.4f}, Davies-Bouldin Index: {dbscan_db:.4f}")
else:
    print("DBSCAN did not find more than one cluster.")

# -----------------------------
# Cluster centers (original units)
# -----------------------------
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers_original, columns=FEATURES)
centers_df.index.name = "Cluster"

print("\n=== CLUSTER CENTERS (Original Units) ===")
print(centers_df.round(2))

# -----------------------------
# Sanity check: 3 products from 2 random clusters for each algorithm
# -----------------------------
print("\n=== SAMPLE PRODUCTS FROM 2 RANDOM KMeans CLUSTERS ===")
kmeans_clusters = list(df_products['KMeans_Cluster'].unique())
for cluster in random.sample(kmeans_clusters, min(2, len(kmeans_clusters))):
    sample = df_products[df_products['KMeans_Cluster'] == cluster].sample(n=min(3, (df_products['KMeans_Cluster'] == cluster).sum()), random_state=42)
    print(f"\nKMeans Cluster {cluster} samples:")
    print(sample[['StockCode', 'Description'] + FEATURES + ["KMeans_Cluster"]])

print("\n=== SAMPLE PRODUCTS FROM 2 RANDOM DBSCAN CLUSTERS ===")
dbscan_clusters = list(df_products['DBSCAN_Cluster'].unique())
for cluster in random.sample(dbscan_clusters, min(2, len(dbscan_clusters))):
    sample = df_products[df_products['DBSCAN_Cluster'] == cluster].sample(n=min(3, (df_products['DBSCAN_Cluster'] == cluster).sum()), random_state=42)
    print(f"\nDBSCAN Cluster {cluster} samples:")
    print(sample[['StockCode', 'Description'] + FEATURES + ["DBSCAN_Cluster"]])

# -----------------------------
# Cluster profiling: mean of each feature per cluster for each algorithm to help business interpretation.
# -----------------------------
print("\n=== CLUSTER PROFILES (Feature Means by Cluster) ===")
print(df_products.groupby('KMeans_Cluster')[FEATURES].mean().round(2))
print(df_products.groupby('DBSCAN_Cluster')[FEATURES].mean().round(2))

# -----------------------------
# Save clustered dataset
# -----------------------------
df_products.to_csv("dataset/processed/products_labeled_clusters.csv", index=False)
print(f"\nSaved clustered dataset → dataset/processed/products_labeled_clusters.csv")

# After fitting scaler, kmeans, dbscan
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(kmeans, "models/kmeans_model.joblib")
joblib.dump(dbscan, "models/dbscan_model.joblib")

# -----------------------------
# Cluster visualization with PCA
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans visualization
plt.figure(figsize=(8,6))
for cluster in range(best_k):
    plt.scatter(
        X_pca[df_products['KMeans_Cluster'] == cluster, 0],
        X_pca[df_products['KMeans_Cluster'] == cluster, 1],
        label=f'KMeans Cluster {cluster}', alpha=0.6
    )
plt.title('KMeans Clusters (PCA 2D)')
plt.legend()
plt.tight_layout()
plt.savefig('clustering_results/kmeans_cluster_pca_plot.png')
plt.show()

# DBSCAN visualization
plt.figure(figsize=(8,6))
for cluster in set(df_products['DBSCAN_Cluster']):
    plt.scatter(
        X_pca[df_products['DBSCAN_Cluster'] == cluster, 0],
        X_pca[df_products['DBSCAN_Cluster'] == cluster, 1],
        label=f'DBSCAN Cluster {cluster}', alpha=0.6
    )
plt.title('DBSCAN Clusters (PCA 2D)')
plt.legend()
plt.tight_layout()
plt.savefig('clustering_results/dbscan_cluster_pca_plot.png')
plt.show()