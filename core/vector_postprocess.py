from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import geopandas as gpd
import numpy as np


@dataclass
class MinAreaMergeResult:
    gdf: gpd.GeoDataFrame
    stats: Dict[str, Any]


def _recompute_area(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["area_m2"] = gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10000.0
    return gdf


def _shared_boundary_length(geom_a, geom_b) -> float:
    try:
        inter = geom_a.boundary.intersection(geom_b.boundary)
        return float(inter.length)
    except Exception:
        return 0.0


def _pick_best_neighbor(
    idx_small: int,
    gdf: gpd.GeoDataFrame,
) -> int:
    """
    Pick the best neighbor for a small polygon:
    1) touching neighbors
    2) intersecting neighbors
    3) nearest neighbor

    Preference: largest shared boundary length.
    """
    geom_small = gdf.loc[idx_small, "geometry"]

    others = gdf.drop(index=idx_small).copy()
    if others.empty:
        raise ValueError("Cannot merge: there is only one polygon in the dataset.")

    # 1) touches
    candidates = others[others.geometry.touches(geom_small)].copy()

    # 2) intersects
    if candidates.empty:
        candidates = others[others.geometry.intersects(geom_small)].copy()

    # 3) fallback: nearest
    if candidates.empty:
        dists = others.geometry.distance(geom_small)
        idx_nearest = dists.idxmin()
        return int(idx_nearest)

    # Pick by largest shared boundary
    candidates["shared_len"] = candidates.geometry.apply(
        lambda g: _shared_boundary_length(geom_small, g)
    )

    # If all shared lengths are zero (weird topology), fallback to nearest among candidates
    if float(candidates["shared_len"].max()) <= 0:
        dists = candidates.geometry.distance(geom_small)
        idx_best = dists.idxmin()
        return int(idx_best)

    idx_best = candidates["shared_len"].idxmax()
    return int(idx_best)


def enforce_min_polygon_area(
    gdf: gpd.GeoDataFrame,
    min_area_ha: float,
    zone_field: str = "zone_id",
    max_iters: int = 10000,
) -> MinAreaMergeResult:
    """
    Merge polygons smaller than min_area_ha into neighboring polygons.

    Rules:
    - Area is evaluated per polygon/component
    - Neighbor preference: largest shared boundary
    - Fallback: nearest polygon
    - The absorbed polygon inherits the neighbor's zone_field value
    """
    if gdf.empty:
        raise ValueError("Input GeoDataFrame is empty.")

    if "geometry" not in gdf.columns:
        raise ValueError("Input GeoDataFrame has no geometry column.")

    if min_area_ha <= 0:
        raise ValueError("min_area_ha must be > 0.")

    work = gdf.copy()
    work = work.reset_index(drop=True)
    work = _recompute_area(work)

    if zone_field not in work.columns:
        raise ValueError(f"zone_field '{zone_field}' not found in GeoDataFrame.")

    n_initial = len(work)
    merged_count = 0
    iters = 0

    while True:
        iters += 1
        if iters > max_iters:
            raise RuntimeError(
                f"Maximum iterations reached while enforcing min polygon area ({max_iters})."
            )

        small = work[work["area_ha"] < min_area_ha]

        if small.empty:
            break

        # Take the smallest polygon first (more deterministic than "first row")
        idx_small = small["area_ha"].idxmin()
        idx_target = _pick_best_neighbor(idx_small, work)

        geom_small = work.loc[idx_small, "geometry"]
        geom_target = work.loc[idx_target, "geometry"]

        # Merge geometry into target polygon
        merged_geom = geom_target.union(geom_small)

        work.loc[idx_target, "geometry"] = merged_geom

        # Keep target's zone_id (small polygon is absorbed)
        work = work.drop(index=idx_small).reset_index(drop=True)

        # Clean potential geometry issues and recompute area
        work["geometry"] = work.geometry.buffer(0)
        work = work[~work.geometry.is_empty & work.geometry.notnull()].copy()
        work = _recompute_area(work)

        merged_count += 1

    # explode again just in case any union created multipart
    work = work.explode(index_parts=False).reset_index(drop=True)
    work = work[~work.geometry.is_empty & work.geometry.notnull()].copy()
    work["geometry"] = work.geometry.buffer(0)
    work = _recompute_area(work)

    # Reassign poly_id after merges
    work["poly_id"] = np.arange(1, len(work) + 1, dtype=int)

    stats = {
        "n_initial_polygons": int(n_initial),
        "n_final_polygons": int(len(work)),
        "merged_polygons": int(merged_count),
        "min_area_ha_applied": float(min_area_ha),
        "total_area_ha": float(work["area_ha"].sum()),
    }

    return MinAreaMergeResult(gdf=work, stats=stats)