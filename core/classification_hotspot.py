from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rasterio.io import DatasetReader
from sklearn.preprocessing import StandardScaler


@dataclass
class HotspotClassificationResult:
    zone_arr: np.ndarray
    profile: Dict[str, Any]
    stats: Dict[str, Any]
    feature_names: List[str]


def _validate_rasters(rasters: Dict[str, DatasetReader]) -> None:
    if not rasters:
        raise ValueError("No rasters were provided for hotspot classification.")

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
        raise ValueError("No pixels are valid across all rasters for hotspot classification.")

    X = np.column_stack([arrays[attr][valid_mask] for attr in feature_names])

    return X, valid_mask, (h, w), feature_names, arrays


def _classify_value_by_levels(value: float, levels: List[Dict[str, Any]]) -> Tuple[str, float]:
    """
    Returns (level_name, score) for one value according to a technical library.
    """
    for lvl in levels:
        cond = True
        if "min" in lvl and lvl["min"] is not None:
            cond &= value >= lvl["min"]
        if "max" in lvl and lvl["max"] is not None:
            cond &= value < lvl["max"]
        if cond:
            return lvl["level"], float(lvl["score"])

    raise ValueError(f"Value {value} did not match any hotspot level rule.")


def run_hotspot_library(
    rasters: Dict[str, DatasetReader],
    selected_attributes: List[str],
    classification_library: Dict[str, Any],
    dst_nodata_id: int = 0,
) -> HotspotClassificationResult:
    """
    Hotspot mode based on a technical recommendation library
    provided directly in the JSON payload.

    Expected structure:
    {
        "attributes": {
            "pH": {
                "levels": [
                    {"level": "baixo", "min": None, "max": 5.5, "score": 0},
                    {"level": "ideal", "min": 5.5, "max": 6.5, "score": 1},
                    {"level": "alto", "min": 6.5, "max": None, "score": 0}
                ]
            },
            ...
        }
    }

    Automatic zone logic:
    - top_1 = all selected attributes classified at maximum score
    - top_2 = intermediate score (> 0 and < max_score)
    - top_3 = score == 0
    """
    rasters_sel = {k: v for k, v in rasters.items() if k in selected_attributes}
    X, valid_mask, (h, w), feature_names, _ = _stack_valid_pixels(rasters_sel)

    library = classification_library.get("attributes")
    if not library:
        raise ValueError("classification_library must contain an 'attributes' key.")

    for attr in feature_names:
        if attr not in library:
            raise ValueError(f"Attribute '{attr}' not found in classification_library.")

        if "levels" not in library[attr]:
            raise ValueError(f"Attribute '{attr}' must contain a 'levels' list.")

    scores = np.zeros(X.shape[0], dtype=float)
    max_score = 0.0

    for i, attr in enumerate(feature_names):
        levels = library[attr]["levels"]
        attr_scores = np.array(
            [_classify_value_by_levels(v, levels)[1] for v in X[:, i]],
            dtype=float
        )
        scores += attr_scores
        max_attr_score = max(float(lvl["score"]) for lvl in levels)
        max_score += max_attr_score

    zone_values = np.full(X.shape[0], dst_nodata_id, dtype=np.int32)

    # top_1 = atende totalmente
    zone_values[scores == max_score] = 1

    # top_3 = não atende
    zone_values[scores == 0] = 3

    # top_2 = atende parcialmente
    zone_values[(scores > 0) & (scores < max_score)] = 2

    zone_arr = np.full((h, w), dst_nodata_id, dtype=np.int32)
    zone_arr[valid_mask] = zone_values

    first_ds = rasters_sel[feature_names[0]]
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

    unique_zones, counts = np.unique(zone_values, return_counts=True)
    cluster_sizes = {int(z): int(c) for z, c in zip(unique_zones, counts)}

    stats = {
        "hotspot_mode": "library",
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "total_pixels": int(h * w),
        "valid_pixels": int(valid_mask.sum()),
        "invalid_pixels": int((~valid_mask).sum()),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_max_possible": float(max_score),
        "zone_sizes_pixels": cluster_sizes,
    }

    return HotspotClassificationResult(
        zone_arr=zone_arr,
        profile=profile,
        stats=stats,
        feature_names=feature_names,
    )


def _apply_target_rule(values: np.ndarray, operator: str, threshold: float) -> np.ndarray:
    if operator == ">=":
        return values >= threshold
    elif operator == ">":
        return values > threshold
    elif operator == "<=":
        return values <= threshold
    elif operator == "<":
        return values < threshold
    elif operator == "==":
        return values == threshold
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def run_hotspot_target(
    rasters: Dict[str, DatasetReader],
    selected_attributes: List[str],
    target_rules: List[Dict[str, Any]],
    negative_split_quantile: float = 0.5,
    dst_nodata_id: int = 0,
) -> HotspotClassificationResult:
    """
    Hotspot mode based on user-defined target criteria.
    top_1 = satisfies all criteria
    top_2 = does not satisfy all, but is closer to target
    top_3 = farther from target
    """
    rasters_sel = {k: v for k, v in rasters.items() if k in selected_attributes}
    X, valid_mask, (h, w), feature_names, _ = _stack_valid_pixels(rasters_sel)

    attr_to_idx = {a: i for i, a in enumerate(feature_names)}

    # build positive mask
    positive = np.ones(X.shape[0], dtype=bool)
    target_vector = np.zeros(len(feature_names), dtype=float)

    for rule in target_rules:
        attr = rule["attribute"]
        if attr not in attr_to_idx:
            raise ValueError(f"Target rule attribute '{attr}' not found in selected rasters.")

        idx = attr_to_idx[attr]
        thr = float(rule["value"])
        positive &= _apply_target_rule(X[:, idx], rule["operator"], thr)
        target_vector[idx] = thr

    zone_values = np.full(X.shape[0], dst_nodata_id, dtype=np.int32)

    # top_1 = satisfies all rules
    zone_values[positive] = 1

    negative = ~positive
    n_neg = int(negative.sum())

    if n_neg > 0:
        # standardize data + target vector for fair Euclidean distance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        target_scaled = scaler.transform(target_vector.reshape(1, -1))[0]

        dists = np.linalg.norm(X_scaled - target_scaled, axis=1)

        neg_dists = dists[negative]
        cut = float(np.quantile(neg_dists, negative_split_quantile))

        # top_2 = closer negatives
        zone_values[negative & (dists <= cut)] = 2

        # top_3 = farther negatives
        zone_values[negative & (dists > cut)] = 3

    zone_arr = np.full((h, w), dst_nodata_id, dtype=np.int32)
    zone_arr[valid_mask] = zone_values

    first_ds = rasters_sel[feature_names[0]]
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

    unique_zones, counts = np.unique(zone_values, return_counts=True)
    cluster_sizes = {int(z): int(c) for z, c in zip(unique_zones, counts)}

    stats = {
        "hotspot_mode": "target",
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "total_pixels": int(h * w),
        "valid_pixels": int(valid_mask.sum()),
        "invalid_pixels": int((~valid_mask).sum()),
        "n_positive_pixels": int(positive.sum()),
        "n_negative_pixels": int(n_neg),
        "negative_split_quantile": float(negative_split_quantile),
        "zone_sizes_pixels": cluster_sizes,
    }

    return HotspotClassificationResult(
        zone_arr=zone_arr,
        profile=profile,
        stats=stats,
        feature_names=feature_names,
    )