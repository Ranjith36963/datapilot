"""
Dimensionality reduction — PCA, Truncated SVD, Factor Analysis, t-SNE, UMAP.

Provides explained variance, component loadings, and 2D visualization data.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..utils.helpers import load_data, save_data, setup_logging, get_numeric_columns
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.dimensionality")


def reduce_dimensions(
    file_path: str,
    method: str = "pca",
    n_components: Union[int, str] = "auto",
    features: Optional[List[str]] = None,
) -> dict:
    """
    Dimensionality reduction.

    Methods: pca, truncated_svd, factor_analysis, tsne, umap.
    n_components: int or 'auto' (retains 95% variance for PCA).
    """
    try:
        df = load_data(file_path)
        cols = features or get_numeric_columns(df)
        X = df[cols].copy()
        for c in X.columns:
            X[c] = X[c].fillna(X[c].median())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_orig = X_scaled.shape[1]

        logger.info(f"Reducing {n_orig} dims via {method}")

        if method == "pca":
            from sklearn.decomposition import PCA

            if n_components == "auto":
                pca_full = PCA().fit(X_scaled)
                cum = np.cumsum(pca_full.explained_variance_ratio_)
                n_comp = int(np.searchsorted(cum, 0.95) + 1)
                n_comp = max(1, min(n_comp, n_orig))
            else:
                n_comp = int(n_components)

            pca = PCA(n_components=n_comp)
            X_reduced = pca.fit_transform(X_scaled)
            evr = pca.explained_variance_ratio_.tolist()
            cum_var = np.cumsum(evr).tolist()

            components_info = []
            for i in range(n_comp):
                loadings = pca.components_[i]
                top_idx = np.argsort(np.abs(loadings))[::-1][:5]
                top_feats = [{"feature": cols[j], "loading": round(float(loadings[j]), 4)} for j in top_idx]
                components_info.append({
                    "component": i + 1,
                    "variance_explained": round(float(evr[i]), 4),
                    "top_features": top_feats,
                })

        elif method == "truncated_svd":
            from sklearn.decomposition import TruncatedSVD
            n_comp = int(n_components) if n_components != "auto" else min(n_orig - 1, 10)
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            X_reduced = svd.fit_transform(X_scaled)
            evr = svd.explained_variance_ratio_.tolist()
            cum_var = np.cumsum(evr).tolist()
            components_info = []

        elif method == "factor_analysis":
            from sklearn.decomposition import FactorAnalysis
            n_comp = int(n_components) if n_components != "auto" else min(n_orig, 5)
            fa = FactorAnalysis(n_components=n_comp, random_state=42)
            X_reduced = fa.fit_transform(X_scaled)
            evr = []
            cum_var = []
            components_info = []

        elif method == "tsne":
            from sklearn.manifold import TSNE
            n_comp = int(n_components) if n_components != "auto" else 2
            tsne = TSNE(n_components=n_comp, random_state=42, perplexity=min(30, len(X_scaled) - 1))
            X_reduced = tsne.fit_transform(X_scaled)
            evr = []
            cum_var = []
            components_info = []

        elif method == "umap":
            try:
                import umap
                n_comp = int(n_components) if n_components != "auto" else 2
                reducer = umap.UMAP(n_components=n_comp, random_state=42)
                X_reduced = reducer.fit_transform(X_scaled)
            except ImportError:
                return {"status": "error", "message": "umap-learn not installed"}
            evr = []
            cum_var = []
            components_info = []
        else:
            return {"status": "error", "message": f"Unknown method: {method}"}

        # Save reduced data
        from pathlib import Path
        red_df = pd.DataFrame(X_reduced, columns=[f"comp_{i+1}" for i in range(X_reduced.shape[1])])
        out_path = str(Path(file_path).parent / "reduced_data.csv")
        save_data(red_df, out_path)

        result = {
            "status": "success",
            "method": method,
            "original_dimensions": n_orig,
            "reduced_dimensions": X_reduced.shape[1],
            "explained_variance_ratio": evr,
            "cumulative_variance": cum_var,
            "components": components_info,
            "output_path": out_path,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def pca_analysis(file_path: str, n_components: Optional[int] = None) -> dict:
    """PCA with full loadings analysis."""
    return reduce_dimensions(file_path, method="pca", n_components=n_components or "auto")


def visualize_2d(file_path: str, method: str = "pca", color_by: Optional[str] = None) -> dict:
    """Reduce to 2D and return scatter-ready data."""
    try:
        df = load_data(file_path)
        result = reduce_dimensions(file_path, method=method, n_components=2)
        if result.get("status") != "success":
            return result

        out_path = result.get("output_path", "")
        red_df = pd.read_csv(out_path) if out_path else None

        points = []
        if red_df is not None:
            for i in range(len(red_df)):
                pt: Dict[str, Any] = {"x": float(red_df.iloc[i, 0]), "y": float(red_df.iloc[i, 1])}
                if color_by and color_by in df.columns:
                    pt["color"] = safe_json_serialize(df[color_by].iloc[i])
                points.append(pt)

        result["points"] = points[:5000]  # cap for JSON size
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def reduce_and_upload(file_path: str, output_name: str = "dimensionality.json", **kwargs) -> dict:
    """Convenience function: reduce_dimensions + upload."""
    result = reduce_dimensions(file_path, **kwargs)
    upload_result(result, output_name)
    return result
