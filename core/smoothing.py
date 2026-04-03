from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, polygonize
from topojson import Topology


@dataclass
class SmoothingResult:
    gdf: gpd.GeoDataFrame
    stats: Dict[str, Any]


# ----------------------------------
# Basic geometry helpers
# ----------------------------------
def _fix_geom(geom):
    if geom is None or geom.is_empty:
        return geom
    try:
        return geom.buffer(0)
    except Exception:
        return geom


def _valid_geom_mask(gdf: gpd.GeoDataFrame):
    return gdf.geometry.apply(lambda geom: geom is not None and not geom.is_empty)


def _recompute_area(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["area_m2"] = gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10000.0
    return gdf


def _get_area_contour(area_gdf: gpd.GeoDataFrame):
    if area_gdf.empty:
        raise ValueError("AOI GeoDataFrame is empty.")
    if area_gdf.crs is None:
        raise ValueError("AOI CRS is undefined.")
    area = area_gdf.copy()
    area["geometry"] = area.geometry.apply(_fix_geom)
    area = area[_valid_geom_mask(area)].copy()
    if area.empty:
        raise ValueError("AOI has no valid geometries.")
    return unary_union(area.geometry)


def remove_internal_overlaps(
    zones: gpd.GeoDataFrame,
    sort_fields: list[str] | None = None
) -> gpd.GeoDataFrame:
    """
    Remove residual overlaps between polygons by sequential planarization.

    Earlier polygons in the chosen order keep their geometry;
    later polygons lose the overlapping portion.

    This is intentionally simple and deterministic, suitable for
    very small overlap artifacts created by Chaikin smoothing.
    """
    gdf = zones.copy()
    gdf["geometry"] = gdf.geometry.apply(_fix_geom)
    gdf = gdf[_valid_geom_mask(gdf)].copy()

    if gdf.empty:
        return gdf

    # deterministic order
    if sort_fields is None:
        sort_fields = []
        if "zone_id" in gdf.columns:
            sort_fields.append("zone_id")
        if "area_ha" in gdf.columns:
            sort_fields.append("area_ha")

    if sort_fields:
        ascending = [True] * len(sort_fields)
        # if area_ha is present, make it descending
        ascending = [False if f == "area_ha" else True for f in sort_fields]
        gdf = gdf.sort_values(sort_fields, ascending=ascending).reset_index(drop=True)
    else:
        gdf = gdf.reset_index(drop=True)

    cleaned_geoms = []
    occupied = None

    for geom in gdf.geometry:
        geom = _fix_geom(geom)

        if occupied is None:
            cleaned = geom
            occupied = geom
        else:
            cleaned = _fix_geom(geom.difference(occupied))
            if cleaned is not None and not cleaned.is_empty:
                occupied = _fix_geom(occupied.union(cleaned))

        cleaned_geoms.append(cleaned)

    gdf["geometry"] = cleaned_geoms
    gdf = gdf[_valid_geom_mask(gdf)].copy()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf["geometry"] = gdf.geometry.apply(_fix_geom)
    gdf = gdf[_valid_geom_mask(gdf)].copy()
    gdf = _recompute_area(gdf)

    return gdf

# ----------------------------------
# Topological smoothing
# ----------------------------------
def smooth_zones_topology(
    zones: gpd.GeoDataFrame,
    area_gdf: gpd.GeoDataFrame,
    id_field: str = "poly_id",
    epsilon: float = 8.0,
) -> gpd.GeoDataFrame:
    """
    Topological simplification:
    - preserves shared boundaries between adjacent polygons
    - reconstructs polygon faces
    - clips result back to original AOI contour
    - reassigns attributes based on largest area overlap with original polygons
    """
    if zones.empty:
        raise ValueError("Input zones GeoDataFrame is empty.")

    if id_field not in zones.columns:
        raise ValueError(f"id_field '{id_field}' not found in zones.")

    gdf = zones.copy()
    gdf["geometry"] = gdf.geometry.apply(_fix_geom)
    gdf = gdf[_valid_geom_mask(gdf)].copy()

    if gdf.empty:
        raise ValueError("No valid geometries available for topological smoothing.")

    if gdf.crs is None:
        raise ValueError("Zones CRS is undefined. Topological smoothing requires a projected CRS.")

    if area_gdf.crs != gdf.crs:
        area_gdf = area_gdf.to_crs(gdf.crs)

    # True outer contour from AOI, not from pixelated zones
    area_contour = _get_area_contour(area_gdf)
    area_boundary = area_contour.boundary

    # Topological simplification
    topo = Topology(gdf)
    topo_simplified = topo.toposimplify(epsilon=epsilon)
    simp = topo_simplified.to_gdf()

    simp["geometry"] = simp.geometry.apply(_fix_geom)
    simp = simp[_valid_geom_mask(simp)].copy()

    if simp.empty:
        raise ValueError("Topological simplification produced no valid geometries.")

    # Build shared linework + force original AOI boundary
    linework = unary_union(list(simp.boundary) + [area_boundary])

    # Rebuild polygon faces from simplified linework
    polys = list(polygonize(linework))
    rebuilt = gpd.GeoDataFrame(geometry=polys, crs=gdf.crs)

    if rebuilt.empty:
        raise ValueError("Topological smoothing failed to rebuild polygons.")

    rebuilt["geometry"] = rebuilt.geometry.apply(lambda geom: _fix_geom(geom.intersection(area_contour)))
    rebuilt = rebuilt[_valid_geom_mask(rebuilt)].copy()

    if rebuilt.empty:
        raise ValueError("No polygons remained after clipping rebuilt faces to AOI contour.")

    # Reassign attributes by largest intersection area with original zones
    original = gdf.copy()
    original["_orig_idx"] = np.arange(len(original))

    rebuilt["_new_idx"] = np.arange(len(rebuilt))
    inter = gpd.overlay(
        rebuilt[["_new_idx", "geometry"]],
        original,
        how="intersection",
        keep_geom_type=False
    )

    inter = inter[_valid_geom_mask(inter)].copy()
    if inter.empty:
        raise ValueError("No overlap found between rebuilt polygons and original zones.")

    inter["int_area"] = inter.geometry.area
    winner = inter.sort_values("int_area", ascending=False).drop_duplicates("_new_idx")

    attrs = [c for c in original.columns if c != "geometry"]
    rebuilt = rebuilt.merge(
        winner[["_new_idx"] + attrs],
        on="_new_idx",
        how="left"
    )

    rebuilt = rebuilt.drop(columns=["_new_idx"], errors="ignore")

    # dissolve by id_field so same polygon id returns as one object if split during rebuild
    out = rebuilt.dissolve(by=id_field, as_index=False)
    out["geometry"] = out.geometry.apply(_fix_geom)
    out = out[_valid_geom_mask(out)].copy()

    # explode again to keep separate components explicit
    out = out.explode(index_parts=False).reset_index(drop=True)
    out = out[_valid_geom_mask(out)].copy()
    out = _recompute_area(out)

    return out


# ----------------------------------
# Chaikin smoothing
# ----------------------------------
def chaikin_ring(coords, iterations: int = 2, offset: float = 0.25):
    pts = list(coords)
    if len(pts) < 4:
        return pts

    for _ in range(iterations):
        new_pts = []
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            q = (
                (1 - offset) * p0[0] + offset * p1[0],
                (1 - offset) * p0[1] + offset * p1[1],
            )
            r = (
                offset * p0[0] + (1 - offset) * p1[0],
                offset * p0[1] + (1 - offset) * p1[1],
            )
            new_pts.extend([q, r])

        if new_pts[0] != new_pts[-1]:
            new_pts.append(new_pts[0])

        pts = new_pts

    return pts


def _smooth_polygon(poly: Polygon, iterations: int = 2, offset: float = 0.25) -> Polygon:
    if poly.is_empty:
        return poly

    ext = chaikin_ring(list(poly.exterior.coords), iterations=iterations, offset=offset)
    ints = [chaikin_ring(list(r.coords), iterations=iterations, offset=offset) for r in poly.interiors]
    return Polygon(ext, ints)


def smooth_per_polygon(
    zones: gpd.GeoDataFrame,
    iterations: int = 2,
    offset: float = 0.25
) -> gpd.GeoDataFrame:
    gdf = zones.copy()
    new_geoms = []

    for geom in gdf.geometry:
        if isinstance(geom, Polygon):
            new_geoms.append(_fix_geom(_smooth_polygon(geom, iterations, offset)))
        elif isinstance(geom, MultiPolygon):
            parts = [_fix_geom(_smooth_polygon(p, iterations, offset)) for p in geom.geoms]
            new_geoms.append(_fix_geom(MultiPolygon(parts)))
        else:
            new_geoms.append(geom)

    gdf["geometry"] = new_geoms
    gdf = gdf[_valid_geom_mask(gdf)].copy()
    return gdf


# ----------------------------------
# Gap filling after geometric smoothing
# ----------------------------------
def _extract_polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(_extract_polygon_parts(g))
        return out
    return []


def _shared_boundary_length(geom_a, geom_b) -> float:
    try:
        inter = geom_a.boundary.intersection(geom_b.boundary)
        return float(inter.length)
    except Exception:
        return 0.0


def fill_gaps_by_boundary(
    zones: gpd.GeoDataFrame,
    area_gdf: gpd.GeoDataFrame,
    min_gap_area_ha: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Fill gaps against the true AOI contour, not against union(zones).
    """
    gdf = zones.copy()
    gdf["geometry"] = gdf.geometry.apply(_fix_geom)
    gdf = gdf[_valid_geom_mask(gdf)].copy()

    if gdf.empty:
        raise ValueError("No valid geometries available for gap filling.")

    if area_gdf.crs != gdf.crs:
        area_gdf = area_gdf.to_crs(gdf.crs)

    area_contour = _get_area_contour(area_gdf)
    union_now = unary_union(gdf.geometry)

    gaps = _fix_geom(area_contour.difference(union_now))
    gap_parts = _extract_polygon_parts(gaps)

    if not gap_parts:
        return _recompute_area(gdf)

    min_gap_m2 = min_gap_area_ha * 10000.0

    for gap in gap_parts:
        if gap.is_empty or gap.area <= min_gap_m2:
            continue

        candidates = gdf[gdf.geometry.touches(gap)].copy()

        if candidates.empty:
            candidates = gdf[gdf.geometry.intersects(gap)].copy()

        if candidates.empty:
            dists = gdf.geometry.distance(gap.centroid)
            idx_best = dists.idxmin()
        else:
            candidates["shared_len"] = candidates.geometry.apply(lambda gg: _shared_boundary_length(gap, gg))
            if float(candidates["shared_len"].max()) <= 0:
                dists = candidates.geometry.distance(gap.centroid)
                idx_best = dists.idxmin()
            else:
                idx_best = candidates["shared_len"].idxmax()

        gdf.loc[idx_best, "geometry"] = _fix_geom(gdf.loc[idx_best, "geometry"].union(gap))

    gdf["geometry"] = gdf.geometry.apply(_fix_geom)
    gdf = gdf[_valid_geom_mask(gdf)].copy()
    gdf = _recompute_area(gdf)

    return gdf


# ----------------------------------
# Full smoothing pipeline
# ----------------------------------
def smooth_and_fill(
    zones: gpd.GeoDataFrame,
    area_gdf: gpd.GeoDataFrame,
    id_field: str = "poly_id",
    epsilon: float = 8.0,
    chaikin_iterations: int = 2,
    chaikin_offset: float = 0.25,
    min_gap_area_ha: float = 0.0,
    reset_poly_id: bool = True,
) -> SmoothingResult:
    """
    Full smoothing pipeline using AOI contour as the true external boundary.
    """
    if zones.empty:
        raise ValueError("Input zones GeoDataFrame is empty.")

    original = zones.copy()
    original["geometry"] = original.geometry.apply(_fix_geom)
    original = original[_valid_geom_mask(original)].copy()

    if original.empty:
        raise ValueError("No valid geometries available for smoothing.")

    if area_gdf.crs != original.crs:
        area_gdf = area_gdf.to_crs(original.crs)

    area_contour = _get_area_contour(area_gdf)

    # Step 1: topological smoothing
    topo = smooth_zones_topology(
        zones=original,
        area_gdf=area_gdf,
        id_field=id_field,
        epsilon=epsilon,
    )

    # Step 2: per-polygon Chaikin
    smooth = smooth_per_polygon(
        topo,
        iterations=chaikin_iterations,
        offset=chaikin_offset,
    )

    # Step 3: clip everything back to original AOI contour
    smooth["geometry"] = smooth.geometry.apply(lambda geom: _fix_geom(geom.intersection(area_contour)))
    smooth = smooth[_valid_geom_mask(smooth)].copy()

    # Step 4: fill residual gaps against AOI contour
    filled = fill_gaps_by_boundary(
        smooth,
        area_gdf=area_gdf,
        min_gap_area_ha=min_gap_area_ha
    )

    # Step 5: final planarization to remove residual overlaps
    filled = remove_internal_overlaps(
        filled,
        sort_fields=["zone_id", "area_ha"] if "zone_id" in filled.columns else ["area_ha"]
    )
    
    # final cleanup
    filled = filled.explode(index_parts=False).reset_index(drop=True)
    filled["geometry"] = filled.geometry.apply(_fix_geom)
    filled = filled[_valid_geom_mask(filled)].copy()
    filled = _recompute_area(filled)

    if reset_poly_id:
        filled["poly_id"] = np.arange(1, len(filled) + 1, dtype=int)

    stats = {
        "n_input_polygons": int(len(zones)),
        "n_output_polygons": int(len(filled)),
        "epsilon": float(epsilon),
        "chaikin_iterations": int(chaikin_iterations),
        "chaikin_offset": float(chaikin_offset),
        "min_gap_area_ha": float(min_gap_area_ha),
        "total_area_ha": float(filled["area_ha"].sum()) if "area_ha" in filled.columns else None,
    }

    return SmoothingResult(gdf=filled, stats=stats)