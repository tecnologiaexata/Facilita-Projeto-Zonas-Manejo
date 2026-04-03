from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
from rasterio.io import DatasetReader
from rasterio.features import geometry_mask


@dataclass
class ThresholdPreview:
    attribute: str
    units: str
    stats: Dict[str, float]
    n_valid: int
    suggested_classes: List[Dict[str, Any]]


def _inside_mask(aoi: gpd.GeoDataFrame, ds: DatasetReader) -> np.ndarray:
    geoms = [g for g in aoi.geometry if g is not None]
    if not geoms:
        raise ValueError("AOI has no valid geometries.")
    return geometry_mask(
        geometries=geoms,
        out_shape=(ds.height, ds.width),
        transform=ds.transform,
        invert=True,
        all_touched=False,
    )


def compute_threshold_preview(
    aoi: gpd.GeoDataFrame,
    ds: DatasetReader,
    attribute: str,
    units: str = "same_as_raster",
    n_classes: int = 3,
    sample_max: int = 250_000,
    seed: int = 42,
) -> ThresholdPreview:
    """
    Preview stats inside AOI for a single-band continuous raster.
    Returns quartiles + suggested thresholds (tercis/quartis).
    """
    if ds.count != 1:
        raise ValueError("threshold preview expects single-band raster.")

    band = ds.read(1, masked=True).astype("float64")
    inside = _inside_mask(aoi, ds)

    valid = inside & (~band.mask) & np.isfinite(band.filled(np.nan))
    vals = band.filled(np.nan)[valid]

    n_valid = int(vals.size)
    if n_valid == 0:
        raise ValueError("No valid pixels inside AOI for threshold preview.")

    # Optional sampling for speed
    if n_valid > sample_max:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_valid, size=sample_max, replace=False)
        vals_use = vals[idx]
    else:
        vals_use = vals

    # Stats
    q = np.percentile(vals_use, [0, 25, 50, 75, 100])
    stats = {
        "min": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "max": float(q[4]),
        "mean": float(np.mean(vals_use)),
        "std": float(np.std(vals_use, ddof=1)) if vals_use.size > 1 else 0.0,
    }

    # Suggested classes: tercis (3) ou quartis (4) por padrão
    suggested_classes: List[Dict[str, Any]] = []
    if n_classes == 3:
        p = np.percentile(vals_use, [33.3333, 66.6667])
        suggested_classes = [
            {"id": 1, "label": "Baixa", "max": float(p[0])},
            {"id": 2, "label": "Media", "min": float(p[0]), "max": float(p[1])},
            {"id": 3, "label": "Alta", "min": float(p[1])},
        ]
    elif n_classes == 4:
        p = np.percentile(vals_use, [25, 50, 75])
        suggested_classes = [
            {"id": 1, "label": "Muito baixa", "max": float(p[0])},
            {"id": 2, "label": "Baixa", "min": float(p[0]), "max": float(p[1])},
            {"id": 3, "label": "Media", "min": float(p[1]), "max": float(p[2])},
            {"id": 4, "label": "Alta", "min": float(p[2])},
        ]
    else:
        raise ValueError("n_classes must be 3 or 4 (for now).")

    return ThresholdPreview(
        attribute=attribute,
        units=units,
        stats=stats,
        n_valid=n_valid,
        suggested_classes=suggested_classes,
    )