from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize, shapes
from shapely.geometry import shape


@dataclass
class PolygonizeResult:
    gdf: gpd.GeoDataFrame
    stats: Dict[str, Any]


def raster_to_polygons(
    zone_arr: np.ndarray,
    transform,
    crs,
    nodata_id: int = 0,
) -> PolygonizeResult:
    """
    Convert a classified raster (integer zone ids) into polygons.

    Parameters
    ----------
    zone_arr : np.ndarray
        2D array of integer zone ids.
    transform : affine.Affine
        Raster transform.
    crs : rasterio.crs.CRS or anything accepted by GeoPandas
        CRS of the raster.
    nodata_id : int
        Zone id representing nodata / outside AOI. Will be excluded.

    Returns
    -------
    PolygonizeResult
        GeoDataFrame with one row per contiguous polygon.
    """
    if zone_arr.ndim != 2:
        raise ValueError("zone_arr must be a 2D array.")

    if not np.issubdtype(zone_arr.dtype, np.integer):
        raise ValueError("zone_arr must contain integer class ids.")

    records = []

    for geom, value in shapes(zone_arr.astype(np.int32), transform=transform):
        zone_id = int(value)

        if zone_id == nodata_id:
            continue

        geom_shp = shape(geom)

        if geom_shp.is_empty:
            continue

        records.append({
            "zone_id": zone_id,
            "geometry": geom_shp
        })

    if not records:
        raise ValueError("No valid polygons were generated from zone raster.")

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    # explode multipart just in case
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # remove invalid/empty geometries
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.area > 0].copy()

    if gdf.empty:
        raise ValueError("Polygonization produced no valid geometries after cleanup.")

    gdf["poly_id"] = np.arange(1, len(gdf) + 1, dtype=int)
    gdf["area_m2"] = gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10000.0

    stats = {
        "n_polygons": int(len(gdf)),
        "zone_ids": sorted(gdf["zone_id"].unique().tolist()),
        "total_area_ha": float(gdf["area_ha"].sum())
    }

    return PolygonizeResult(gdf=gdf, stats=stats)


def write_polygons(
    gdf: gpd.GeoDataFrame,
    out_path: str,
    driver: str = "GPKG",
    layer: str = "zones"
) -> str:
    """
    Save polygons to disk.

    Parameters
    ----------
    gdf : GeoDataFrame
    out_path : str
    driver : str
        Default GPKG. Use 'ESRI Shapefile' if needed.
    layer : str
        Only used for GPKG.
    """
    if driver.upper() == "GPKG":
        gdf.to_file(out_path, driver=driver, layer=layer)
    else:
        gdf.to_file(out_path, driver=driver)

    return out_path


def polygons_to_raster(
    gdf: gpd.GeoDataFrame,
    *,
    zone_field: str,
    reference_profile: Dict[str, Any],
    nodata_id: int = 0,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Rasterize final polygons back to the AOI/reference raster grid.

    This is useful when the final published TIFF must match the post-processed
    vector result instead of the intermediate classified raster.
    """
    if gdf.empty:
        raise ValueError("Cannot rasterize an empty GeoDataFrame.")

    if zone_field not in gdf.columns:
        raise ValueError(f"zone_field '{zone_field}' not found in polygons.")

    height = int(reference_profile["height"])
    width = int(reference_profile["width"])
    transform = reference_profile["transform"]

    shapes_to_burn = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        zone_id = int(row[zone_field])
        shapes_to_burn.append((geom, zone_id))

    if not shapes_to_burn:
        raise ValueError("No valid polygons were available for rasterization.")

    zone_arr = rasterize(
        shapes=shapes_to_burn,
        out_shape=(height, width),
        transform=transform,
        fill=nodata_id,
        dtype=np.int32,
        all_touched=False,
    )

    profile = {
        "driver": "GTiff",
        "dtype": "int32",
        "count": 1,
        "crs": reference_profile["crs"],
        "transform": transform,
        "width": width,
        "height": height,
        "nodata": nodata_id,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }

    return zone_arr, profile
