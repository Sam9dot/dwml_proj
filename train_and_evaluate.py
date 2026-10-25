"""
Healthcare Cost Clustering & Unusual Billing Detection
Algorithms: K-Means (Segmentation) & DBSCAN (Anomaly Detection)
Saves all model artifacts into a single clean file: models/model.joblib
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

def run_pipeline():
    print("Running Clustering Pipeline...")

    # 1. Load data
    data_path = os.path.join("data", "insurance.csv")
    df = pd.read_csv(data_path)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Preprocess
    df['smoker_encoded'] = df['smoker'].map({'yes': 1, 'no': 0}).astype(int)
    feature_cols = ['age', 'bmi', 'smoker_encoded', 'charges']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)

    # 3. K-Means (K=2 to 8 scan, choosing K=4)
    kmeans_scan = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        kmeans_scan.append({
            'k': k,
            'inertia': round(float(km.inertia_), 1),
            'silhouette': round(float(sil), 4),
            'davies_bouldin': round(float(db), 4)
        })

    best_k = 4
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

    # 4. DBSCAN (eps=0.58, min_samples=8)
    eps = 0.58
    min_samples = 8
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

    n_noise = int((df['dbscan_cluster'] == -1).sum())
    noise_pct = round((n_noise / len(df)) * 100, 2)
    n_dbscan_clusters = len(set(df['dbscan_cluster'])) - (1 if -1 in df['dbscan_cluster'].values else 0)

    # 5. Compute the 3 Simple Metrics
    # Metric 1: Silhouette Score
    km_sil = round(float(silhouette_score(X_scaled, df['kmeans_cluster'])), 4)
    valid_db_mask = df['dbscan_cluster'] != -1
    db_sil = round(float(silhouette_score(X_scaled[valid_db_mask], df.loc[valid_db_mask, 'dbscan_cluster'])), 4)

    # Metric 2: Davies-Bouldin Index (lower is better)
    km_db = round(float(davies_bouldin_score(X_scaled, df['kmeans_cluster'])), 4)
    db_db = round(float(davies_bouldin_score(X_scaled[valid_db_mask], df.loc[valid_db_mask, 'dbscan_cluster'])), 4)

    # Metric 3: Noise / Outlier Percentage
    km_noise = 0.0
    db_noise = noise_pct

    metrics_table = [
        {
            'Algorithm': 'K-Means (Centroid)',
            'Clusters': int(best_k),
            'Silhouette Score': float(km_sil),
            'Davies-Bouldin Index': float(km_db),
            'Noise / Outlier %': f"{km_noise}%"
        },
        {
            'Algorithm': 'DBSCAN (Density)',
            'Clusters': int(n_dbscan_clusters),
            'Silhouette Score': float(db_sil),
            'Davies-Bouldin Index': float(db_db),
            'Noise / Outlier %': f"{db_noise}%"
        }
    ]

    print("\n--- 3 Simple Clustering Metrics ---")
    print(pd.DataFrame(metrics_table).to_string(index=False))

    # 6. Cluster Personas Profile
    cluster_profiles = []
    for c_id in sorted(df['kmeans_cluster'].unique()):
        sub = df[df['kmeans_cluster'] == c_id]
        smoker_share = float(sub['smoker_encoded'].mean())
        mean_age = float(sub['age'].mean())
        mean_bmi = float(sub['bmi'].mean())
        mean_charges = float(sub['charges'].mean())

        if smoker_share >= 0.70:
            name = "Smokers (High Cost Pattern)"
            tier = "High"
        elif mean_age >= 45:
            name = "Mature Non-Smokers (Moderate Cost Pattern)"
            tier = "Moderate"
        elif mean_bmi >= 32:
            name = "Young High-BMI Non-Smokers (Low-Mod Cost)"
            tier = "Low-Moderate"
        else:
            name = "Young Lean Non-Smokers (Baseline Low Cost)"
            tier = "Low"

        cluster_profiles.append({
            'cluster_id': int(c_id),
            'persona_name': name,
            'cost_tier': tier,
            'count': int(len(sub)),
            'share_pct': round((len(sub) / len(df)) * 100, 1),
            'avg_age': round(mean_age, 1),
            'avg_bmi': round(mean_bmi, 1),
            'smoker_pct': round(smoker_share * 100, 1),
            'avg_charges': round(mean_charges, 2),
            'median_charges': round(float(sub['charges'].median()), 2)
        })

    # 7. Save everything into ONE single file: models/model.joblib
    os.makedirs("models", exist_ok=True)
    model_payload = {
        'scaler': scaler,
        'kmeans': kmeans,
        'dbscan_core_samples': X_scaled[dbscan.core_sample_indices_],
        'dbscan_eps': eps,
        'dbscan_min_samples': min_samples,
        'feature_cols': feature_cols,
        'cluster_profiles': cluster_profiles,
        'metrics_table': metrics_table,
        'kmeans_scan': kmeans_scan,
        'df': df
    }
    
    output_path = os.path.join("models", "model.joblib")
    joblib.dump(model_payload, output_path)
    print(f"\nAll artifacts saved into single file: {output_path}")

if __name__ == "__main__":
    run_pipeline()
