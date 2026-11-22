"""
PharmaPulse - Sales Behaviour Clustering API Router
===================================================
Provides dynamic clustering endpoints for SKU behavioural segmentation:
  - 3 Discovered Behavioural Cohorts with authentic centroids & clinical strategies:
    * Cohort 0: Core Staples (88 SKUs)
    * Cohort 1: Seasonal & Growth Therapies (32 SKUs)
    * Cohort 2: Erratic Spikers & High Margin (30 SKUs)
  - 2D & 3D PCA coordinates with 75.0% explained variance
  - SKU deep-dive economics: authentic unit cost, selling price, and gross margin %
  - Multi-criteria filtering by cohort cluster_id, therapeutic category_id, or search query
"""

import os
import sys
from typing import Optional, Dict, Any
from fastapi import APIRouter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.clustering.predict import get_clustering_inference_data

router = APIRouter(prefix="/api/clustering", tags=["Sales Behaviour Clustering"])

# In-memory cache for sub-millisecond API response
_cached_clustering_data: Optional[Dict[str, Any]] = None


def get_cached_clustering(force_reload: bool = False) -> Dict[str, Any]:
    """Returns cached clustering data or generates it on first request."""
    global _cached_clustering_data
    if _cached_clustering_data is None or force_reload:
        _cached_clustering_data = get_clustering_inference_data()
    return _cached_clustering_data


def clear_clustering_cache():
    """Invalidates the in-memory cache (e.g. after model retraining)."""
    global _cached_clustering_data
    _cached_clustering_data = None


@router.get("")
def get_clustering_data(
    cluster_id: Optional[int] = None,
    category_id: str = "ALL",
    search: Optional[str] = None,
    force_reload: bool = False
):
    """
    Returns authentic clustering profiles, 2D PCA projection coordinates,
    and medicine-level economics. Supports dynamic filtering.
    """
    raw_data = get_cached_clustering(force_reload=force_reload)
    all_points = raw_data["sku_points"]
    profiles = raw_data["profiles"]
    kpis = raw_data["kpis"]

    filtered_points = all_points.copy()

    # Apply Cluster ID filter if specified
    if cluster_id is not None:
        filtered_points = [p for p in filtered_points if p["cluster_id"] == cluster_id]

    # Apply Category ID filter if specified
    if category_id != "ALL":
        cat_num = "".join(filter(str.isdigit, str(category_id)))
        clean_cat = f"CAT{int(cat_num):03d}" if cat_num else str(category_id)
        filtered_points = [p for p in filtered_points if p["category_id"].upper() == clean_cat.upper()]

    # Apply Search query if specified
    if search:
        q = str(search).strip().lower()
        filtered_points = [
            p for p in filtered_points
            if q in p["medicine_name"].lower() or q in p["medicine_id"].lower() or q in p["cluster_name"].lower()
        ]

    return {
        "kpis": {
            "total_skus": len(filtered_points),
            "total_formulary_skus": kpis["total_skus"],
            "silhouette_score": kpis["silhouette_score"],
            "optimal_k": kpis["optimal_k"],
            "variance_explained_pct": kpis["variance_explained_pct"]
        },
        "profiles": profiles,
        "sku_points": filtered_points
    }
