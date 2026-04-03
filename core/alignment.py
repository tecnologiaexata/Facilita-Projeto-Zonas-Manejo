from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import os
import re
import time
import unicodedata

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.io import DatasetReader
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask

from core.io import LoadedInputs


@dataclass
class AlignmentResult:
    rasters: Dict[str, DatasetReader]              # attribute -> opened aligned dataset
    raster_paths: Dict[str, Path]                  # attribute -> new (or original) path
    report: Dict[str, Any]                         # what happened


def _sanitize_filename_part(value: str, fallback: str = "atributo") -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode(
        "ascii", "ignore"
    ).decode("ascii")
    text = re.sub(r"[^\w\s.-]", "_", text)
    text = re.sub(r"[/\\]+", "_", text)
    text = re.sub(r"[\s_-]+", "_", text).strip("._")
    return text or fallback


# -----------------------------
# Grid utilities
# -----------------------------
def _safe_unlink(path: Path, retries: int = 5, sleep_s: float = 0.2) -> None:
    """
    Remove file with retries (Windows can keep handles briefly).
    """
    if not path.exists():
        return
    last_err = None
    for _ in range(retries):
        try:
            path.unlink()
            return
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"Could not delete existing file: {path}. Last error: {last_err}") from last_err


def _aoi_bounds_in_crs(aoi: gpd.GeoDataFrame, target_crs: CRS) -> Tuple[float, float, float, float]:
    g = aoi.to_crs(target_crs)
    minx, miny, maxx, maxy = g.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _snap_bounds_to_grid(
    bounds: Tuple[float, float, float, float],
    cell: float
) -> Tuple[float, float, float, float]:
    """
    Expand bounds outward so that min/max align with the target grid.
    """
    minx, miny, maxx, maxy = bounds

    minx_s = np.floor(minx / cell) * cell
    miny_s = np.floor(miny / cell) * cell
    maxx_s = np.ceil(maxx / cell) * cell
    maxy_s = np.ceil(maxy / cell) * cell

    return float(minx_s), float(miny_s), float(maxx_s), float(maxy_s)


def _target_transform_and_shape(
    bounds: Tuple[float, float, float, float],
    cell: float
) -> Tuple[Affine, int, int]:
    """
    Create a north-up affine transform and (height, width) for given bounds and cell size.
    """
    minx, miny, maxx, maxy = bounds
    width = int(round((maxx - minx) / cell))
    height = int(round((maxy - miny) / cell))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid target grid shape derived from bounds={bounds} and cell={cell}.")

    transform = Affine(cell, 0.0, minx, 0.0, -cell, maxy)
    return transform, height, width


def _resampling_for(
    src_res: Tuple[float, float],
    target_cell: float
) -> Resampling:
    """
    For continuous data only:
      - if downsampling (src finer than target) -> average
      - if upsampling (src coarser than target) -> bilinear
    """
    # rasterio returns (xres, yres), both positive
    src_cell = float(np.mean([abs(src_res[0]), abs(src_res[1])]))
    if src_cell < target_cell:   # going coarser
        return Resampling.average
    else:                        # going finer or equal
        return Resampling.bilinear


# -----------------------------
# Checks
# -----------------------------
def check_alignment(
    inputs: LoadedInputs,
    target_cell_m: float = 10.0,
    require_full_aoi_coverage: bool = True
) -> None:
    """
    Strict check:
    - All rasters must match AOI CRS
    - All rasters must match target_cell_m (within tolerance)
    - All rasters must share same transform/shape after snapping AOI bounds to grid (i.e., already aligned)
    - Optionally: each raster must cover the AOI bounds

    Raises ValueError with clear message if any requirement fails.
    """
    aoi_crs = inputs.aoi.crs
    if aoi_crs is None:
        raise ValueError("AOI CRS is None; cannot perform alignment checks.")

    target_crs = CRS.from_user_input(aoi_crs)
    raw_bounds = _aoi_bounds_in_crs(inputs.aoi, target_crs)
    snapped_bounds = _snap_bounds_to_grid(raw_bounds, target_cell_m)
    target_transform, target_h, target_w = _target_transform_and_shape(snapped_bounds, target_cell_m)

    # Small tolerances to avoid float issues
    cell_tol = 1e-6

    for attr, ds in inputs.rasters.items():
        # CRS
        if ds.crs is None or CRS.from_user_input(ds.crs) != target_crs:
            raise ValueError(
                f"Raster '{attr}' CRS does not match AOI CRS.\n"
                f"  raster CRS: {ds.crs}\n"
                f"  aoi CRS:    {target_crs}\n"
                "Use auto_fix alignment or re-export rasters in the same CRS as the AOI."
            )

        # Resolution
        xres, yres = ds.res
        if abs(abs(xres) - target_cell_m) > cell_tol or abs(abs(yres) - target_cell_m) > cell_tol:
            raise ValueError(
                f"Raster '{attr}' resolution does not match target {target_cell_m} m.\n"
                f"  raster res: {ds.res}\n"
                f"  target:     ({target_cell_m}, {target_cell_m})\n"
                "Use auto_fix alignment or re-export rasters at the target resolution."
            )

        # Transform + shape
        if ds.transform != target_transform or ds.width != target_w or ds.height != target_h:
            raise ValueError(
                f"Raster '{attr}' grid does not match target grid derived from AOI.\n"
                f"  raster transform: {ds.transform}\n"
                f"  target transform: {target_transform}\n"
                f"  raster shape:     (h={ds.height}, w={ds.width})\n"
                f"  target shape:     (h={target_h}, w={target_w})\n"
                "Use auto_fix alignment (crop/resample) or re-export rasters already aligned."
            )

        # Coverage (optional)
        if require_full_aoi_coverage:
            rb = ds.bounds
            minx, miny, maxx, maxy = snapped_bounds
            covers = (rb.left <= minx) and (rb.bottom <= miny) and (rb.right >= maxx) and (rb.top >= maxy)
            if not covers:
                raise ValueError(
                    f"Raster '{attr}' does not fully cover AOI bounds.\n"
                    f"  raster bounds: {rb}\n"
                    f"  aoi bounds:    {snapped_bounds}\n"
                    "Use auto_fix alignment (will fill missing with nodata) or provide a raster with full coverage."
                )


# -----------------------------
# Auto-fix
# -----------------------------
def align_to_aoi_grid(
    inputs: LoadedInputs,
    target_cell_m: float = 10.0,
    out_dir: Path | str = "outputs/temp",
    dst_nodata: float = -9999.0,
    fill_missing_with_nodata: bool = True,
    logger: Optional[logging.Logger] = None,
) -> AlignmentResult:
    """
    Auto-fix alignment:
    - Reproject rasters to AOI CRS if needed
    - Resample to target_cell_m
    - Crop/warp to AOI bounds snapped to grid
    - Writes aligned rasters to out_dir and returns opened datasets + report

    Note:
    - Continuous attributes only (bilinear/average).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_crs = CRS.from_user_input(inputs.aoi.crs)
    raw_bounds = _aoi_bounds_in_crs(inputs.aoi, target_crs)
    snapped_bounds = _snap_bounds_to_grid(raw_bounds, target_cell_m)
    target_transform, target_h, target_w = _target_transform_and_shape(snapped_bounds, target_cell_m)

    report: Dict[str, Any] = {
        "mode": "auto_fix",
        "target": {
            "crs": str(target_crs),
            "cell_m": target_cell_m,
            "bounds": snapped_bounds,
            "shape": [target_h, target_w],
        },
        "rasters": {}
    }

    if logger:
        logger.info(
            "Alignment | target grid | crs=%s | cell_m=%s | bounds=%s | shape=(h=%s,w=%s) | out_dir=%s",
            target_crs,
            target_cell_m,
            snapped_bounds,
            target_h,
            target_w,
            out_dir,
        )

    aligned_paths: Dict[str, Path] = {}
    aligned_datasets: Dict[str, DatasetReader] = {}

    # Create AOI mask on the target grid (optional use later in pipeline)
    # We don't apply mask here; we just align to AOI extent.
    # Masking by AOI polygon can be done after classification.
    for attr, src in inputs.rasters.items():
        src_crs = CRS.from_user_input(src.crs)
        src_res = src.res
        resampling = _resampling_for(src_res, target_cell_m)

        attr_filename = _sanitize_filename_part(attr)
        out_path = out_dir / f"{attr_filename}_aligned_{int(target_cell_m)}m.tif"

        raster_report = {
            "input_path": str(inputs.raster_paths.get(attr, "")),
            "output_path": str(out_path),
            "changed_crs": str(src_crs) != str(target_crs),
            "src_crs": str(src_crs),
            "dst_crs": str(target_crs),
            "src_res": tuple(map(float, src_res)),
            "dst_res": (target_cell_m, target_cell_m),
            "resampling": resampling.name,
            "filled_missing_with_nodata": fill_missing_with_nodata,
            "dst_nodata": dst_nodata,
        }

        if logger:
            logger.info(
                "Alignment | raster start | attribute=%s | src_crs=%s | src_res=%s | resampling=%s | out=%s",
                attr,
                src_crs,
                tuple(map(float, src_res)),
                resampling.name,
                out_path,
            )

        # Prepare output profile
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "crs": target_crs,
            "transform": target_transform,
            "width": target_w,
            "height": target_h,
            "nodata": dst_nodata,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }

        # Reproject band 1 only (we assume single-band tifs per attribute)
        _safe_unlink(out_path)
        
        try:
            with rasterio.open(out_path, "w", **profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    src_nodata=src.nodata,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    dst_nodata=dst_nodata if fill_missing_with_nodata else None,
                    resampling=resampling,
                )
        except Exception:
            # Clean partial/corrupted output if something goes wrong
            try:
                _safe_unlink(out_path, retries=3, sleep_s=0.2)
            except Exception:
                pass
            raise

        # Reopen for downstream usage
        ds_aligned = rasterio.open(out_path)
        aligned_paths[attr] = out_path
        aligned_datasets[attr] = ds_aligned
        report["rasters"][attr] = raster_report
        if logger:
            logger.info(
                "Alignment | raster done | attribute=%s | aligned_path=%s | shape=(h=%s,w=%s)",
                attr,
                out_path,
                ds_aligned.height,
                ds_aligned.width,
            )

    if logger:
        logger.info(
            "Alignment | completed | rasters=%s",
            list(aligned_paths.keys()),
        )

    return AlignmentResult(rasters=aligned_datasets, raster_paths=aligned_paths, report=report)
