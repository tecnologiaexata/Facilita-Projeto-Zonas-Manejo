from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.io import DatasetReader
from rasterio.features import geometry_mask


@dataclass
class ThresholdClass:
    id: int
    label: str
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class ThresholdResult:
    zone_arr: np.ndarray                 # int32 (H, W), 0 = nodata/outside
    profile: Dict[str, Any]              # raster profile suitable for writing GeoTIFF
    legend: Dict[int, str]               # class_id -> label
    stats: Dict[str, Any]                # quick debug stats


def _validate_classes(classes: List[ThresholdClass]) -> None:
    if not classes:
        raise ValueError("threshold: 'classes' must be a non-empty list.")

    ids = [c.id for c in classes]
    if len(ids) != len(set(ids)):
        raise ValueError("threshold: class ids must be unique.")

    # Basic min/max sanity
    for c in classes:
        if c.min is not None and c.max is not None and c.min >= c.max:
            raise ValueError(
                f"threshold: invalid class range for id={c.id} ('min' must be < 'max')."
            )


def _build_inside_mask(aoi: gpd.GeoDataFrame, ds: DatasetReader) -> np.ndarray:
    """
    Returns a boolean mask (H, W) where True indicates pixels INSIDE the AOI.
    Assumes AOI CRS matches ds.crs (enforced in alignment).
    """
    geoms = [geom for geom in aoi.geometry if geom is not None]
    if not geoms:
        raise ValueError("AOI has no valid geometries for masking.")

    inside = geometry_mask(
        geometries=geoms,
        out_shape=(ds.height, ds.width),
        transform=ds.transform,
        invert=True,        # True inside polygons
        all_touched=False,  # conservative boundary (avoids fattening)
    )
    return inside


def run_threshold_classification(
    aoi: gpd.GeoDataFrame,
    ds: DatasetReader,
    classes: List[Dict[str, Any]] | List[ThresholdClass],
    dst_nodata_id: int = 0,
) -> ThresholdResult:
    """
    Classify a single continuous raster into discrete zones using user-defined thresholds.

    Parameters
    ----------
    aoi : GeoDataFrame
        AOI polygons. Must be in same CRS as ds.
    ds : DatasetReader
        Single-band raster aligned to AOI grid (same CRS, transform, resolution, bounds).
    classes : list
        Either list of dicts (from JSON) or list of ThresholdClass.
        Each item: {id, label, min?, max?}
    dst_nodata_id : int
        Zone id used for nodata/outside AOI.

    Returns
    -------
    ThresholdResult
    """
    # Normalize input classes
    if classes and isinstance(classes[0], dict):
        parsed = [ThresholdClass(**c) for c in classes]  # type: ignore[arg-type]
    else:
        parsed = classes  # type: ignore[assignment]

    _validate_classes(parsed)

    if ds.count != 1:
        raise ValueError("threshold: expected a single-band raster (one tif per attribute).")

    # Read raster band as masked array (respects src nodata)
    band = ds.read(1, masked=True)

    # Convert to float for comparisons; keep mask
    data = band.astype("float64")
    valid = ~data.mask

    # AOI mask
    inside = _build_inside_mask(aoi, ds)

    # Initialize zones with nodata id
    zone_arr = np.full((ds.height, ds.width), dst_nodata_id, dtype=np.int32)

    # Pixels eligible for classification
    eligible = inside & valid & np.isfinite(data.filled(np.nan))

    # Apply classes in order given (you control order in JSON)
    # Convention:
    #   - min is inclusive if present
    #   - max is exclusive if present
    filled = data.filled(np.nan)

    classified_any = np.zeros_like(zone_arr, dtype=bool)

    for c in parsed:
        cond = eligible.copy()
        if c.min is not None:
            cond &= (filled >= c.min)
        if c.max is not None:
            cond &= (filled < c.max)

        zone_arr[cond] = int(c.id)
        classified_any |= cond

    # Optional: warn-like stats (we return them; you can log later)
    total_inside = int(inside.sum())
    total_eligible = int(eligible.sum())
    total_classified = int(classified_any.sum())
    unclassified_inside_valid = int((eligible & ~classified_any).sum())

    legend = {int(c.id): c.label for c in parsed}

    profile = {
        "driver": "GTiff",
        "dtype": "int32",
        "count": 1,
        "crs": ds.crs,
        "transform": ds.transform,
        "width": ds.width,
        "height": ds.height,
        "nodata": dst_nodata_id,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }

    stats = {
        "total_pixels": int(ds.width * ds.height),
        "inside_pixels": total_inside,
        "eligible_pixels": total_eligible,
        "classified_pixels": total_classified,
        "unclassified_inside_valid_pixels": unclassified_inside_valid,
        "nodata_or_outside_pixels": int((zone_arr == dst_nodata_id).sum()),
    }

    return ThresholdResult(zone_arr=zone_arr, profile=profile, legend=legend, stats=stats)


def write_zone_raster(
    out_path: str,
    zone_arr: np.ndarray,
    profile: Dict[str, Any],
) -> str:
    """
    Convenience writer for debugging / intermediate artifacts.
    """
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(zone_arr.astype(profile["dtype"]), 1)
    return out_path