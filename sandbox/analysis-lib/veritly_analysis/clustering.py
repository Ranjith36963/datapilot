"""
Clustering — find customer segments and natural groupings.

Algorithms: K-Means, DBSCAN, HDBSCAN, Hierarchical, GMM.
Includes optimal-k detection via elbow and silhouette methods.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from veritly_analysis.utils import (
    load_data,
    save_data,
    safe_json_serialize,
    setup_logging,
    get_numeric_columns,
    upload_result,
)


logger = setup_logging("clustering")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_features(df: pd.DataFrame, features: Optional[List[str]] = None):
    """Prepare and scale numeric features for clustering."""
    cols = features or get_numeric_columns(df)
    X = df[cols].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, cols, scaler


def _profile_clusters(df: pd.DataFrame, labels: np.ndarray, features: List[str]) -> list:
    """Build a profile for each cluster: size, centroid, distinguishing features."""
    df_work = df.copy()
    df_work["__cluster__"] = labels
    overall_means = {f: float(df_work[f].mean()) for f in features if pd.api.types.is_numeric_dtype(df_work[f])}

    profiles = []
    for label in sorted(set(labels)):
        if label == -1:
            continue  # noise
        mask = df_work["__cluster__"] == label
        cluster_df = df_work[mask]
        size = int(mask.sum())

        centroid = {}
        diffs = []
        for f in features:
            if not pd.api.types.is_numeric_dtype(df_work[f]):
                continue
            cm = float(cluster_df[f].mean())
            centroid[f] = round(cm, 4)
            om = overall_means.get(f, 0)
            diff_pct = round((cm - om) / abs(om) * 100, 2) if om != 0 else 0.0
            diffs.append({
                "feature": f,
                "cluster_mean": round(cm, 4),
                "overall_mean": round(om, 4),
                "diff_pct": diff_pct,
            })

        diffs.sort(key=lambda x: abs(x["diff_pct"]), reverse=True)

        # Auto-label suggestion based on top distinguishing feature
        if diffs:
            top = diffs[0]
            direction = "High" if top["diff_pct"] > 0 else "Low"
            label_suggestion = f"{direction} {top['feature'].replace('_', ' ').title()} Group"
        else:
            label_suggestion = f"Cluster {label}"

        profiles.append({
            "cluster": int(label),
            "size": size,
            "centroid": centroid,
            "distinguishing_features": diffs[:5],
            "label_suggestion": label_suggestion,
        })

    return profiles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_clusters(
    file_path: str,
    n_clusters: Union[int, str] = "auto",
    algorithm: str = "kmeans",
    features: Optional[List[str]] = None,
) -> dict:
    """
    Cluster data into groups.

    Algorithms: kmeans, dbscan, hdbscan, hierarchical, gmm.
    """
    try:
        df = load_data(file_path)
        X_scaled, feat_cols, scaler = _prepare_features(df, features)

        logger.info(f"Clustering via {algorithm}: {X_scaled.shape[0]} rows, {X_scaled.shape[1]} features")

        # Determine n_clusters for methods that need it
        k = n_clusters
        if isinstance(k, str) and k == "auto":
            opt = optimal_clusters(file_path, max_k=10, features=features)
            k = opt.get("recommended_k", 3)

        if algorithm == "kmeans":
            model = KMeans(n_clusters=int(k), random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
        elif algorithm == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X_scaled)
        elif algorithm == "hdbscan":
            try:
                import hdbscan as hdb
                model = hdb.HDBSCAN(min_cluster_size=15)
                labels = model.fit_predict(X_scaled)
            except ImportError:
                return {"status": "error", "message": "hdbscan not installed"}
        elif algorithm == "hierarchical":
            model = AgglomerativeClustering(n_clusters=int(k))
            labels = model.fit_predict(X_scaled)
        elif algorithm == "gmm":
            from sklearn.mixture import GaussianMixture
            model = GaussianMixture(n_components=int(k), random_state=42)
            labels = model.fit_predict(X_scaled)
        else:
            return {"status": "error", "message": f"Unknown algorithm: {algorithm}"}

        unique_labels = [l for l in set(labels) if l != -1]
        actual_k = len(unique_labels)

        # Cluster sizes
        sizes = []
        for lab in sorted(unique_labels):
            cnt = int((labels == lab).sum())
            sizes.append({"cluster": int(lab), "size": cnt, "pct": round(cnt / len(labels) * 100, 2)})

        # Metrics (need ≥2 clusters)
        metrics = {}
        if actual_k >= 2:
            non_noise = labels != -1
            if non_noise.sum() > actual_k:
                metrics["silhouette_score"] = round(float(silhouette_score(X_scaled[non_noise], labels[non_noise])), 4)
                metrics["calinski_harabasz"] = round(float(calinski_harabasz_score(X_scaled[non_noise], labels[non_noise])), 2)
                metrics["davies_bouldin"] = round(float(davies_bouldin_score(X_scaled[non_noise], labels[non_noise])), 4)

        # Cluster profiles
        profiles = _profile_clusters(df, labels, feat_cols)

        # Save data with cluster labels
        from pathlib import Path
        out_path = str(Path(file_path).parent / "clustered_data.csv")
        df_out = df.copy()
        df_out["cluster"] = labels
        save_data(df_out, out_path)

        result = {
            "status": "success",
            "algorithm": algorithm,
            "n_clusters": actual_k,
            "cluster_sizes": sizes,
            "metrics": metrics,
            "cluster_profiles": profiles,
            "cluster_assignments": labels.tolist(),
            "output_path": out_path,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def optimal_clusters(
    file_path: str,
    max_k: int = 10,
    features: Optional[List[str]] = None,
) -> dict:
    """
    Find optimal number of clusters using elbow and silhouette methods.
    """
    try:
        df = load_data(file_path)
        X_scaled, _, _ = _prepare_features(df, features)
        max_k = min(max_k, len(X_scaled) - 1)

        inertias = []
        sil_scores = []

        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(float(km.inertia_))
            sil_scores.append(float(silhouette_score(X_scaled, labels)))

        # Elbow: biggest drop in inertia
        diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        elbow_k = diffs.index(max(diffs)) + 2 if diffs else 2

        # Silhouette: highest score
        sil_k = sil_scores.index(max(sil_scores)) + 2

        recommended = sil_k  # silhouette is generally more reliable

        return safe_json_serialize({
            "status": "success",
            "elbow_k": elbow_k,
            "silhouette_k": sil_k,
            "recommended_k": recommended,
            "elbow_scores": inertias,
            "silhouette_scores": sil_scores,
        })

    except Exception as e:
        return {"status": "error", "message": str(e)}


def describe_clusters(df: pd.DataFrame, cluster_column: str) -> dict:
    """Profile each cluster with statistics."""
    if cluster_column not in df.columns:
        return {"status": "error", "message": f"Column '{cluster_column}' not found"}

    num_cols = get_numeric_columns(df)
    num_cols = [c for c in num_cols if c != cluster_column]
    labels = df[cluster_column].values
    profiles = _profile_clusters(df, labels, num_cols)

    return safe_json_serialize({"status": "success", "cluster_profiles": profiles})


def cluster_and_upload(file_path: str, output_name: str = "clustering.json", **kwargs) -> dict:
    """Convenience function: find_clusters + upload."""
    result = find_clusters(file_path, **kwargs)
    upload_result(result, output_name)
    return result
