from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rasterio.io import DatasetReader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class AutoClassificationResult:
    zone_arr: np.ndarray
    profile: Dict[str, Any]
    stats: Dict[str, Any]
    feature_names: List[str]


def _validate_rasters(rasters: Dict[str, DatasetReader]) -> None:
    if not rasters:
        raise ValueError("No rasters were provided for automatic classification.")

    first_ds = None
    for attr, ds in rasters.items():
        if ds.count != 1:
            raise ValueError(f"Raster '{attr}' must have exactly 1 band.")
        if first_ds is None:
            first_ds = ds
            continue

        if ds.crs != first_ds.crs:
            raise ValueError(f"Raster '{attr}' CRS differs from the reference raster.")
        if ds.transform != first_ds.transform:
            raise ValueError(f"Raster '{attr}' transform differs from the reference raster.")
        if ds.width != first_ds.width or ds.height != first_ds.height:
            raise ValueError(f"Raster '{attr}' shape differs from the reference raster.")


def _stack_valid_pixels(
    rasters: Dict[str, DatasetReader]
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], List[str], Dict[str, np.ndarray]]:
    """
    Build feature matrix X from aligned rasters using only pixels valid in all rasters.

    Returns
    -------
    X : np.ndarray
        Shape (n_valid_pixels, n_features)
    valid_mask : np.ndarray
        Shape (height, width), True where all rasters are valid
    shape : tuple
        (height, width)
    feature_names : list[str]
        Attribute names in the same order as columns of X
    raw_arrays : dict[str, np.ndarray]
        Raw masked arrays filled with np.nan, useful for debugging
    """
    _validate_rasters(rasters)

    feature_names = list(rasters.keys())
    first_ds = rasters[feature_names[0]]
    h, w = first_ds.height, first_ds.width

    arrays = {}
    valid_masks = []

    for attr in feature_names:
        band = rasters[attr].read(1, masked=True).astype("float64")
        arr = band.filled(np.nan)

        valid = (~band.mask) & np.isfinite(arr)
        arrays[attr] = arr
        valid_masks.append(valid)

    valid_mask = np.logical_and.reduce(valid_masks)

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise ValueError("No pixels are valid across all rasters for automatic classification.")

    X = np.column_stack([arrays[attr][valid_mask] for attr in feature_names])

    return X, valid_mask, (h, w), feature_names, arrays


def _choose_k_by_silhouette(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    n_init: int = 10,
) -> Tuple[int, Dict[int, float]]:
    """
    Choose k by silhouette score.
    """
    n_samples = X_scaled.shape[0]
    if n_samples < 3:
        raise ValueError("Too few valid samples to compute silhouette for automatic clustering.")

    # avoid impossible k
    k_max_eff = min(k_max, n_samples - 1)
    if k_min > k_max_eff:
        raise ValueError(
            f"Invalid silhouette search interval: k_min={k_min}, k_max={k_max}, "
            f"but only {n_samples} valid samples are available."
        )

    scores: Dict[int, float] = {}
    best_k = None
    best_score = -np.inf

    for k in range(k_min, k_max_eff + 1):
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
        )
        labels = model.fit_predict(X_scaled)

        # silhouette requires at least 2 clusters actually present
        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(X_scaled, labels)
        scores[k] = float(score)

        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        raise ValueError("Failed to determine k automatically using silhouette.")

    return int(best_k), scores


def run_auto_classification(
    rasters: Dict[str, DatasetReader],
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    n_init: int = 10,
    dst_nodata_id: int = 0,
) -> AutoClassificationResult:
    """
    Automatic zone generation using raster-first KMeans clustering.

    Parameters
    ----------
    rasters : dict[str, DatasetReader]
        Aligned single-band rasters, one per attribute.
    k : int or None
        If provided, fixed number of zones.
        If None, choose k using silhouette.
    k_min, k_max : int
        Search interval for silhouette when k is None.
    random_state : int
        Random seed for reproducibility.
    n_init : int
        KMeans n_init.
    dst_nodata_id : int
        Zone id used for invalid / outside pixels.

    Returns
    -------
    AutoClassificationResult
    """
    X, valid_mask, (h, w), feature_names, _ = _stack_valid_pixels(rasters)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose k
    silhouette_scores = None
    if k is None:
        k_chosen, silhouette_scores = _choose_k_by_silhouette(
            X_scaled,
            k_min=k_min,
            k_max=k_max,
            random_state=random_state,
            n_init=n_init,
        )
    else:
        if k < 2:
            raise ValueError("Automatic classification requires k >= 2.")
        k_chosen = int(k)

    # Final clustering
    model = KMeans(
        n_clusters=k_chosen,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(X_scaled)

    # Shift labels to start at 1
    zone_values = labels.astype(np.int32) + 1

    # Rebuild raster
    zone_arr = np.full((h, w), dst_nodata_id, dtype=np.int32)
    zone_arr[valid_mask] = zone_values

    # Use first raster as reference for profile
    first_ds = rasters[feature_names[0]]
    profile = {
        "driver": "GTiff",
        "dtype": "int32",
        "count": 1,
        "crs": first_ds.crs,
        "transform": first_ds.transform,
        "width": first_ds.width,
        "height": first_ds.height,
        "nodata": dst_nodata_id,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }

    # Cluster sizes
    unique_zones, counts = np.unique(zone_values, return_counts=True)
    cluster_sizes = {int(z): int(c) for z, c in zip(unique_zones, counts)}

    stats = {
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "total_pixels": int(h * w),
        "valid_pixels": int(valid_mask.sum()),
        "invalid_pixels": int((~valid_mask).sum()),
        "k_selected": int(k_chosen),
        "k_source": "user_defined" if k is not None else "silhouette",
        "silhouette_scores": silhouette_scores,
        "cluster_sizes_pixels": cluster_sizes,
    }

    return AutoClassificationResult(
        zone_arr=zone_arr,
        profile=profile,
        stats=stats,
        feature_names=feature_names,
    )