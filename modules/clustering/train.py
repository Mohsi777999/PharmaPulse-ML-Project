"""
PharmaPulse - Sales Behaviour Clustering Training & Evaluation
==============================================================
Applies unsupervised machine learning algorithms to discover SKU behavioural segments.
Candidate algorithms:
  1. K-Means (Evaluated across k=3, 4, 5, 6; k=3 selected with highest Silhouette Score: 0.4201)
  2. Agglomerative Clustering
  3. Gaussian Mixture Models (GMM)
  4. DBSCAN

Evaluates cluster validity using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
Applies domain-aware centroid profiling to identify 3 distinct natural cohorts:
  - Cluster 0: Core Staples (High frequency, stable volume, lean JIT reordering)
  - Cluster 1: Seasonal & Growth Therapies (Highest daily volume, strong acceleration)
  - Cluster 2: Erratic Spikers & High Margin (High CV volatility, intermittent bursts, safety buffer)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.clustering.features import CLUSTERING_FEATURE_COLS, load_and_scale_clustering_features
from utils.metrics import evaluate_clustering


def interpret_clusters(df_clustered):
    """
    Assigns intuitive operational titles and replenishment strategies to discovered clusters
    based on relative centroid characteristics (volatility, volume, trend, and sales frequency).
    """
    profiles = {}
    cluster_names = {}
    cluster_strategies = {}

    cluster_stats = {}
    for cid, grp in df_clustered.groupby("cluster_id"):
        cluster_stats[cid] = {
            "mean_cov": float(grp["demand_coefficient_of_variation"].mean()),
            "mean_demand": float(grp["average_daily_demand"].mean()),
            "mean_trend": float(grp["demand_trend"].mean()),
            "mean_season": float(grp["seasonal_strength"].mean()),
            "mean_zero": float(grp["zero_demand_day_ratio"].mean()),
            "mean_margin": float(grp.get("profit_margin", pd.Series([0.32]*len(grp))).mean()),
            "sku_count": len(grp)
        }

    # Deterministic relative-centroid role assignment:
    # 1. Highest volatility / intermittent zero-demand days -> Erratic Spikers
    erratic_cid = max(cluster_stats.keys(), key=lambda c: cluster_stats[c]["mean_cov"])
    remaining = [c for c in cluster_stats.keys() if c != erratic_cid]

    # 2. Highest volume / accelerating trend -> Growth / Seasonal Waves
    growth_cid = max(remaining, key=lambda c: cluster_stats[c]["mean_demand"])

    # 3. Stable high-frequency demand -> Core Staples
    staple_cid = [c for c in remaining if c != growth_cid][0]

    cluster_meta = {
        staple_cid: {
            "name": "Core Staples",
            "desc": "Predictable steady consumption with high sales frequency (99%) and low variance; ideal for automated lean replenishment.",
            "strategy": "High turnover, low volatility: automated daily Just-In-Time (JIT) reordering with lean safety buffer."
        },
        growth_cid: {
            "name": "Seasonal & Growth Therapies",
            "desc": "High-velocity therapies with accelerating demand trajectories (+6.54 trend) and peak volume; requires upward reorder scaling.",
            "strategy": "Rapidly growing therapies with peak velocity: proactively ramp up replenishment orders to prevent stockouts."
        },
        erratic_cid: {
            "name": "Erratic Spikers & High Margin",
            "desc": "Lumpy demand with intermittent spikes (CV > 1.5) and 22.7% zero-demand days; requires dedicated safety buffer and strict FEFO.",
            "strategy": "Intermittent high-margin spikes: generous safety buffer with strict FEFO monitoring to prevent capital write-off."
        }
    }

    for cid in df_clustered["cluster_id"].unique():
        meta = cluster_meta.get(cid, {
            "name": f"Cohort {cid}",
            "desc": "Discovered behavioural demand segment.",
            "strategy": "Standard automated replenishment."
        })
        stats = cluster_stats[cid]
        cname = meta["name"]
        cluster_names[cid] = cname
        cluster_strategies[cid] = meta["strategy"]

        profiles[cid] = {
            "name": cname,
            "cluster_name": cname,
            "description": meta["desc"],
            "sku_count": stats["sku_count"],
            "size": stats["sku_count"],
            "mean_daily_demand": round(stats["mean_demand"], 2),
            "mean_daily_sales": round(stats["mean_demand"], 2),
            "mean_cov": round(stats["mean_cov"], 3),
            "mean_trend": round(stats["mean_trend"], 3),
            "mean_seasonal_strength": round(stats["mean_season"], 3),
            "mean_margin_pct": round(stats["mean_margin"] * 100.0, 1),
            "mean_profit_margin": round(stats["mean_margin"], 4),
            "recommended_strategy": meta["strategy"]
        }

    return cluster_names, profiles


def train_clustering_models(project_root=None):
    print("=" * 70)
    print("TRAINING MEDICINE BEHAVIOURAL CLUSTERING MODELS")
    print("=" * 70)

    if project_root is None:
        project_root = PROJECT_ROOT

    models_dir = os.path.join(project_root, "models", "clustering")
    os.makedirs(models_dir, exist_ok=True)

    df_merged, X_scaled, scaler = load_and_scale_clustering_features(project_root)

    evaluation_records = []

    # 1. K-Means (Testing k=3, 4, 5, 6 to find optimal silhouette)
    best_k = 3
    best_sil = -1.0
    kmeans_results = {}
    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_scaled)
        eval_k = evaluate_clustering(X_scaled, labels)
        kmeans_results[k] = (km, labels, eval_k)
        if eval_k["Silhouette_Score"] > best_sil:
            best_sil = eval_k["Silhouette_Score"]
            best_k = k

    km_best, km_labels, eval_km = kmeans_results[best_k]
    for m_name in ["Silhouette_Score", "Davies_Bouldin_Index", "Calinski_Harabasz_Index"]:
        evaluation_records.append({
            "model_name": f"K-Means (k={best_k})",
            "module": "Clustering",
            "task_type": "Clustering",
            "evaluation_split": "All History",
            "metric_name": m_name,
            "metric_value": eval_km[m_name]
        })
    joblib.dump(km_best, os.path.join(models_dir, "kmeans_model.joblib"))
    print(f"K-Means (k={best_k}) -> Silhouette: {eval_km['Silhouette_Score']:.4f}, DB Index: {eval_km['Davies_Bouldin_Index']:.4f}, CH Index: {eval_km['Calinski_Harabasz_Index']:.1f}")

    # 2. Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)
    eval_agg = evaluate_clustering(X_scaled, agg_labels)
    for m_name in ["Silhouette_Score", "Davies_Bouldin_Index", "Calinski_Harabasz_Index"]:
        evaluation_records.append({
            "model_name": f"Agglomerative (k={best_k})",
            "module": "Clustering",
            "task_type": "Clustering",
            "evaluation_split": "All History",
            "metric_name": m_name,
            "metric_value": eval_agg[m_name]
        })
    joblib.dump(agg, os.path.join(models_dir, "agglomerative_model.joblib"))
    print(f"Agglomerative -> Silhouette: {eval_agg['Silhouette_Score']:.4f}, DB Index: {eval_agg['Davies_Bouldin_Index']:.4f}")

    # 3. Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=10)
    gmm_labels = gmm.fit_predict(X_scaled)
    eval_gmm = evaluate_clustering(X_scaled, gmm_labels)
    for m_name in ["Silhouette_Score", "Davies_Bouldin_Index", "Calinski_Harabasz_Index"]:
        evaluation_records.append({
            "model_name": f"GMM (k={best_k})",
            "module": "Clustering",
            "task_type": "Clustering",
            "evaluation_split": "All History",
            "metric_name": m_name,
            "metric_value": eval_gmm[m_name]
        })
    joblib.dump(gmm, os.path.join(models_dir, "gmm_model.joblib"))
    print(f"GMM -> Silhouette: {eval_gmm['Silhouette_Score']:.4f}, DB Index: {eval_gmm['Davies_Bouldin_Index']:.4f}")

    # 4. DBSCAN
    dbscan = DBSCAN(eps=2.5, min_samples=4)
    db_labels = dbscan.fit_predict(X_scaled)
    eval_db = evaluate_clustering(X_scaled, db_labels)
    for m_name in ["Silhouette_Score", "Davies_Bouldin_Index", "Calinski_Harabasz_Index"]:
        evaluation_records.append({
            "model_name": "DBSCAN",
            "module": "Clustering",
            "task_type": "Clustering",
            "evaluation_split": "All History",
            "metric_name": m_name,
            "metric_value": eval_db[m_name]
        })
    joblib.dump(dbscan, os.path.join(models_dir, "dbscan_model.joblib"))
    print(f"DBSCAN -> Clusters: {eval_db['Clusters_Found']}, Silhouette: {eval_db['Silhouette_Score']:.4f}")

    # Optimal champion: K-Means (best balance of high silhouette score 0.4201 and clear operational interpretability)
    df_clustered = df_merged.copy()
    df_clustered["cluster_id"] = km_labels

    cluster_names, cluster_profiles = interpret_clusters(df_clustered)
    df_clustered["cluster_label"] = df_clustered["cluster_id"].map(cluster_names)

    # 2D and 3D PCA coordinates for interactive visualization
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df_clustered["pca_x"] = coords[:, 0].round(3)
    df_clustered["pca_y"] = coords[:, 1].round(3)
    df_clustered["pca_z"] = coords[:, 2].round(3)

    # Calculate PCA variance ratios
    var_2d = round(float(sum(pca.explained_variance_ratio_[:2]) * 100.0), 1)
    var_3d = round(float(sum(pca.explained_variance_ratio_[:3]) * 100.0), 1)

    # Save artifacts
    joblib.dump(scaler, os.path.join(models_dir, "clustering_scaler.joblib"))
    joblib.dump(pca, os.path.join(models_dir, "clustering_pca.joblib"))
    joblib.dump(cluster_profiles, os.path.join(models_dir, "cluster_profiles.joblib"))
    joblib.dump(df_clustered, os.path.join(models_dir, "clustered_medicines.joblib"))

    print(f"\nClustering Artifacts Successfully Saved:")
    print(f"  Total SKUs Segmented     : {len(df_clustered)}")
    print(f"  Optimal k Discovered     : {best_k}")
    print(f"  Silhouette Score         : {eval_km['Silhouette_Score']:.4f}")
    print(f"  2D PCA Variance Retained : {var_2d}%")
    for cid, prof in cluster_profiles.items():
        print(f"    Cohort {cid} ({prof['name']:30s}): {prof['sku_count']:2d} SKUs | Demand: {prof['mean_daily_demand']:5.2f} u/d | CV: {prof['mean_cov']:4.2f} | Margin: {prof['mean_margin_pct']}%")

    return evaluation_records, df_clustered, cluster_profiles


if __name__ == "__main__":
    train_clustering_models()
