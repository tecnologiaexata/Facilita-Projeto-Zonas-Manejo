from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.io import DatasetReader


@dataclass
class StatisticsResult:
    zones_gdf: gpd.GeoDataFrame   # original zones enriched by zone_id stats
    stats_gdf: gpd.GeoDataFrame   # one row per zone_id
    summary: Dict[str, Any]


def _valid_geom_mask(gdf: gpd.GeoDataFrame):
    return (~gdf.geometry.is_empty) & (gdf.geometry.notna())


def _recompute_area(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["area_m2"] = gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10000.0
    return gdf


def _zonal_stats_single_raster(
    zones_gdf: gpd.GeoDataFrame,
    ds: DatasetReader,
    prefix: str,
) -> gpd.GeoDataFrame:
    """
    Compute per-zone stats for one aligned single-band raster.

    Assumptions:
    - zones_gdf and ds are in the same CRS
    - ds is aligned to the AOI grid
    """
    if ds.count != 1:
        raise ValueError(f"Raster '{prefix}' must have exactly 1 band.")

    out = zones_gdf.copy()
    band = ds.read(1, masked=True).astype("float64")

    means = []
    medians = []
    stds = []
    mins = []
    maxs = []
    n_pixels = []

    for geom in out.geometry:
        mask = geometry_mask(
            geometries=[geom],
            out_shape=(ds.height, ds.width),
            transform=ds.transform,
            invert=True,
            all_touched=False,
        )

        valid = mask & (~band.mask) & np.isfinite(band.filled(np.nan))
        vals = band.filled(np.nan)[valid]

        if vals.size == 0:
            means.append(np.nan)
            medians.append(np.nan)
            stds.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
            n_pixels.append(0)
            continue

        means.append(float(np.mean(vals)))
        medians.append(float(np.median(vals)))
        stds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
        mins.append(float(np.min(vals)))
        maxs.append(float(np.max(vals)))
        n_pixels.append(int(vals.size))

    out[f"{prefix}_n_pixels"] = n_pixels
    out[f"{prefix}_mean"] = means
    out[f"{prefix}_median"] = medians
    out[f"{prefix}_std"] = stds
    out[f"{prefix}_min"] = mins
    out[f"{prefix}_max"] = maxs

    return out


def compute_zone_statistics(
    zones_gdf: gpd.GeoDataFrame,
    rasters: Dict[str, DatasetReader],
    zone_field: str = "zone_id",
) -> StatisticsResult:
    """
    Compute per-zone statistics from final smoothed polygons and aligned rasters.

    Parameters
    ----------
    zones_gdf : GeoDataFrame
        Final zone polygons (can contain multiple polygons per zone_id).
    rasters : dict[str, DatasetReader]
        Aligned rasters by attribute name.
    zone_field : str
        Zone grouping field, default "zone_id".

    Returns
    -------
    StatisticsResult
    """
    if zones_gdf.empty:
        raise ValueError("zones_gdf is empty.")

    if zone_field not in zones_gdf.columns:
        raise ValueError(f"zone_field '{zone_field}' not found in zones_gdf.")

    if not rasters:
        raise ValueError("No rasters were provided for statistics.")

    work = zones_gdf.copy()
    work["geometry"] = work.geometry.buffer(0)
    work = work[_valid_geom_mask(work)].copy()

    if work.empty:
        raise ValueError("No valid geometries found in zones_gdf.")

    # Aggregate final geometry by zone_id for thematic stats
    stats_gdf = work.dissolve(by=zone_field, as_index=False)
    stats_gdf = stats_gdf.explode(index_parts=False).reset_index(drop=True)
    stats_gdf = stats_gdf[_valid_geom_mask(stats_gdf)].copy()
    stats_gdf = _recompute_area(stats_gdf)

    # Zonal stats for each raster
    for attr, ds in rasters.items():
        if stats_gdf.crs != ds.crs:
            raise ValueError(
                f"CRS mismatch for raster '{attr}': zones={stats_gdf.crs}, raster={ds.crs}"
            )
        stats_gdf = _zonal_stats_single_raster(stats_gdf, ds, prefix=attr)

    # Merge stats back into original polygon layer by zone_id
    stat_cols = [c for c in stats_gdf.columns if c not in ["geometry"]]
    zones_enriched = work.merge(
        stats_gdf[stat_cols],
        on=zone_field,
        how="left"
    )

    summary = {
        "n_input_polygons": int(len(zones_gdf)),
        "n_final_polygons": int(len(zones_enriched)),
        "n_zones": int(stats_gdf[zone_field].nunique()),
        "zone_ids": sorted(stats_gdf[zone_field].unique().tolist()),
        "total_area_ha": float(stats_gdf["area_ha"].sum()),
        "attributes": list(rasters.keys()),
    }

    return StatisticsResult(
        zones_gdf=zones_enriched,
        stats_gdf=stats_gdf,
        summary=summary,
    )