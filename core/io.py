from __future__ import annotations

import logging
import mimetypes
import re
import shutil
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import geopandas as gpd
import rasterio
from rasterio.io import DatasetReader

from core.blob_urls import build_blob_request_headers, is_http_url, resolve_blob_download_url
from core.models import AOIRef, RasterRef, ZonesRequest


CONTENT_TYPE_SUFFIX_MAP = {
    "application/geo+json": ".geojson",
    "application/geopackage+sqlite3": ".gpkg",
    "application/json": ".json",
    "application/vnd.google-earth.kml+xml": ".kml",
    "application/vnd.google-earth.kmz": ".kmz",
    "application/x-zip-compressed": ".zip",
    "application/zip": ".zip",
    "image/geotiff": ".tif",
    "image/tiff": ".tif",
}


# -----------------------------
# Result containers (clean API)
# -----------------------------
@dataclass
class DownloadedAsset:
    kind: str
    name: str
    source_url: str
    resolved_url: str
    local_path: Path


@dataclass
class LoadedInputs:
    aoi: gpd.GeoDataFrame
    aoi_path: Path
    aoi_reference: str
    rasters: Dict[str, DatasetReader]          # attribute -> opened dataset
    raster_paths: Dict[str, Path]              # attribute -> local path
    raster_references: Dict[str, str]          # attribute -> original path/url
    download_dir: Path
    downloaded_assets: List[DownloadedAsset]


# -----------------------------
# Helpers
# -----------------------------
def infer_utm_epsg_from_gdf(gdf: gpd.GeoDataFrame) -> int:
    """
    Infer UTM EPSG from AOI geometry (works best if gdf is in EPSG:4326).
    Returns EPSG code (326xx north, 327xx south).
    """
    gdf_ll = gdf.to_crs(4326)
    centroid = gdf_ll.unary_union.centroid
    lon, lat = centroid.x, centroid.y

    zone = int(floor((lon + 180) / 6) + 1)
    return (32700 + zone) if lat < 0 else (32600 + zone)


def _sanitize_name(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "_", text).strip("_")
    return text or "area"


def _ensure_path_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"{what} is a directory (expected file): {path}")


def _ensure_download_dir(cfg: ZonesRequest) -> Path:
    area_name = _sanitize_name(cfg.job.area_name)
    download_dir = Path("outputs") / "temp" / area_name / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir


def _guess_suffix_from_reference(reference: str) -> str:
    if not reference:
        return ""

    parsed = urlparse(reference) if is_http_url(reference) else None
    path_str = parsed.path if parsed else reference
    return Path(path_str).suffix.lower()


def _guess_suffix_from_content_type(content_type: str) -> str:
    cleaned = (content_type or "").split(";", 1)[0].strip().lower()
    if not cleaned:
        return ""

    return CONTENT_TYPE_SUFFIX_MAP.get(cleaned) or mimetypes.guess_extension(cleaned) or ""


def _download_to_file(
    *,
    url: str,
    dst_dir: Path,
    prefix: str,
    default_suffix: str = "",
) -> Tuple[Path, str]:
    resolved_url = str(resolve_blob_download_url(url) or url).strip()
    headers = {
        "User-Agent": "zonas-manejo-api/0.1",
        **build_blob_request_headers(resolved_url),
    }
    request = Request(resolved_url, headers=headers)

    try:
        with urlopen(request, timeout=60) as response:
            content_type = response.headers.get("Content-Type", "")
            suffix = (
                _guess_suffix_from_reference(resolved_url)
                or _guess_suffix_from_reference(url)
                or _guess_suffix_from_content_type(content_type)
                or default_suffix
                or ""
            )
            target = dst_dir / f"{prefix}_{uuid.uuid4().hex}{suffix}"
            with target.open("wb") as fh:
                shutil.copyfileobj(response, fh)
    except Exception as e:
        raise RuntimeError(f"Failed to download remote file: {url}. Error: {e}") from e

    return target, resolved_url


def _extract_vector_archive(path: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path

    extract_dir = path.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(path) as archive:
            archive.extractall(extract_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to extract AOI archive: {path}. Error: {e}") from e

    candidates: List[Path] = []
    for pattern in ("*.gpkg", "*.geojson", "*.json", "*.kml", "*.shp"):
        candidates.extend(sorted(extract_dir.rglob(pattern)))

    if not candidates:
        raise ValueError(
            f"AOI archive does not contain a supported vector file: {path}"
        )

    return candidates[0]


def _resolve_aoi_reference(
    aoi: AOIRef,
    download_dir: Path,
) -> Tuple[Path, str, Optional[DownloadedAsset]]:
    raw_reference = (aoi.url or aoi.id or "").strip()
    if not raw_reference:
        raise ValueError("AOI reference is empty.")

    if is_http_url(raw_reference):
        downloaded_path, resolved_url = _download_to_file(
            url=raw_reference,
            dst_dir=download_dir,
            prefix="aoi",
            default_suffix=".kml",
        )
        local_path = _extract_vector_archive(downloaded_path)
        asset = DownloadedAsset(
            kind="aoi",
            name="aoi",
            source_url=raw_reference,
            resolved_url=resolved_url,
            local_path=local_path,
        )
        return local_path, raw_reference, asset

    local_path = Path(raw_reference)
    return local_path, raw_reference, None


def _resolve_raster_reference(
    raster_ref: RasterRef,
    download_dir: Path,
) -> Tuple[Path, str, Optional[DownloadedAsset]]:
    raw_reference = (raster_ref.url or raster_ref.path or "").strip()
    if not raw_reference:
        raise ValueError(f"Raster reference is empty for attribute='{raster_ref.attribute}'.")

    if is_http_url(raw_reference):
        local_path, resolved_url = _download_to_file(
            url=raw_reference,
            dst_dir=download_dir,
            prefix=f"raster_{_sanitize_name(raster_ref.attribute)}",
            default_suffix=".tif",
        )
        asset = DownloadedAsset(
            kind="raster",
            name=raster_ref.attribute,
            source_url=raw_reference,
            resolved_url=resolved_url,
            local_path=local_path,
        )
        return local_path, raw_reference, asset

    return Path(raw_reference), raw_reference, None


def _load_aoi(aoi_path: Path, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Load AOI from vector file (KML/GPKG/SHP/etc).
    If AOI is KML (typically EPSG:4326), convert to inferred UTM for processing.
    """
    _ensure_path_exists(aoi_path, "AOI file")

    suffix = aoi_path.suffix.lower()

    try:
        if suffix == ".kml":
            gdf = gpd.read_file(aoi_path)
        else:
            gdf = gpd.read_file(aoi_path, layer=layer) if layer else gpd.read_file(aoi_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read AOI: {aoi_path}. Error: {e}") from e

    if gdf.empty:
        raise ValueError(f"AOI is empty: {aoi_path}")

    gdf = gdf[~gdf.geometry.isna()].copy()
    if gdf.empty:
        raise ValueError(f"AOI has no valid geometries after removing nulls: {aoi_path}")

    if gdf.crs is None:
        if suffix == ".kml":
            gdf = gdf.set_crs(4326, allow_override=True)
        else:
            raise ValueError(
                f"AOI has no CRS defined: {aoi_path}. Please ensure AOI has a valid CRS."
            )

    if gdf.crs.to_epsg() == 4326:
        utm_epsg = infer_utm_epsg_from_gdf(gdf)
        gdf = gdf.to_crs(utm_epsg)

    return gdf


def _open_raster(path: Path, attribute: str) -> DatasetReader:
    _ensure_path_exists(path, f"Raster for attribute='{attribute}'")

    try:
        ds = rasterio.open(path)
    except Exception as e:
        raise RuntimeError(f"Failed to open raster '{attribute}': {path}. Error: {e}") from e

    if ds.count < 1:
        ds.close()
        raise ValueError(f"Raster '{attribute}' has no bands: {path}")
    if ds.crs is None:
        ds.close()
        raise ValueError(f"Raster '{attribute}' has no CRS defined: {path}")
    if ds.transform is None:
        ds.close()
        raise ValueError(f"Raster '{attribute}' has no transform: {path}")

    return ds


def build_inputs_report(inputs: LoadedInputs) -> Dict[str, Any]:
    return {
        "download_dir": str(inputs.download_dir),
        "aoi": {
            "reference": inputs.aoi_reference,
            "local_path": str(inputs.aoi_path),
        },
        "rasters": {
            attr: {
                "reference": inputs.raster_references[attr],
                "local_path": str(path),
            }
            for attr, path in inputs.raster_paths.items()
        },
        "downloaded_assets": [
            {
                "kind": asset.kind,
                "name": asset.name,
                "source_url": asset.source_url,
                "resolved_url": asset.resolved_url,
                "local_path": str(asset.local_path),
            }
            for asset in inputs.downloaded_assets
        ],
    }


# -----------------------------
# Public API
# -----------------------------
def load_inputs(
    cfg: ZonesRequest,
    logger: Optional[logging.Logger] = None,
) -> LoadedInputs:
    """
    Load AOI + rasters from the config.

    Responsibilities:
    - Resolve local paths or public URLs
    - Download remote assets to a temp folder when necessary
    - Load AOI GeoDataFrame
    - Open rasters (do NOT align/resample here)

    Returns:
        LoadedInputs with AOI and a dict of opened raster datasets.

    Important:
    - Caller is responsible for closing rasters (use close_inputs()).
    """
    download_dir = _ensure_download_dir(cfg)
    if logger:
        logger.info("Inputs | preparing download directory at %s", download_dir)
    downloaded_assets: List[DownloadedAsset] = []

    aoi_path, aoi_reference, aoi_download = _resolve_aoi_reference(cfg.aoi, download_dir)
    if aoi_download is not None:
        downloaded_assets.append(aoi_download)
        if logger:
            logger.info(
                "Inputs | AOI downloaded | source=%s | resolved=%s | local=%s",
                aoi_download.source_url,
                aoi_download.resolved_url,
                aoi_download.local_path,
            )
    elif logger:
        logger.info("Inputs | AOI local path resolved to %s", aoi_path)
    aoi_gdf = _load_aoi(aoi_path, layer=cfg.aoi.layer)
    if logger:
        logger.info(
            "Inputs | AOI loaded | features=%s | crs=%s | bounds=%s",
            len(aoi_gdf),
            aoi_gdf.crs,
            tuple(map(float, aoi_gdf.total_bounds)),
        )

    rasters: Dict[str, DatasetReader] = {}
    raster_paths: Dict[str, Path] = {}
    raster_references: Dict[str, str] = {}

    seen = set()
    for raster_ref in cfg.rasters:
        if raster_ref.attribute in seen:
            raise ValueError(f"Duplicate raster attribute in request: '{raster_ref.attribute}'")
        seen.add(raster_ref.attribute)

        raster_path, raster_reference, raster_download = _resolve_raster_reference(
            raster_ref,
            download_dir,
        )
        if raster_download is not None:
            downloaded_assets.append(raster_download)
            if logger:
                logger.info(
                    "Inputs | raster downloaded | attribute=%s | source=%s | resolved=%s | local=%s",
                    raster_ref.attribute,
                    raster_download.source_url,
                    raster_download.resolved_url,
                    raster_download.local_path,
                )
        elif logger:
            logger.info(
                "Inputs | raster local path resolved | attribute=%s | local=%s",
                raster_ref.attribute,
                raster_path,
            )

        ds = _open_raster(raster_path, raster_ref.attribute)
        rasters[raster_ref.attribute] = ds
        raster_paths[raster_ref.attribute] = raster_path
        raster_references[raster_ref.attribute] = raster_reference
        if logger:
            logger.info(
                "Inputs | raster opened | attribute=%s | crs=%s | res=%s | shape=(h=%s,w=%s)",
                raster_ref.attribute,
                ds.crs,
                tuple(map(float, ds.res)),
                ds.height,
                ds.width,
            )

    if cfg.mode == "threshold" and len(rasters) != 1:
        raise ValueError("threshold mode requires exactly one raster.")

    if cfg.mode == "hotspot":
        mp = cfg.mode_params or {}
        selected = set(mp.get("selected_attributes", []))
        missing = sorted(selected - set(rasters.keys()))
        if missing:
            raise ValueError(f"hotspot mode missing rasters for attributes: {missing}")

    if logger:
        logger.info(
            "Inputs | completed | mode=%s | rasters=%s | downloaded_assets=%s",
            cfg.mode,
            list(rasters.keys()),
            len(downloaded_assets),
        )

    return LoadedInputs(
        aoi=aoi_gdf,
        aoi_path=aoi_path,
        aoi_reference=aoi_reference,
        rasters=rasters,
        raster_paths=raster_paths,
        raster_references=raster_references,
        download_dir=download_dir,
        downloaded_assets=downloaded_assets,
    )


def close_inputs(inputs: LoadedInputs) -> None:
    """Close all opened raster datasets."""
    for ds in inputs.rasters.values():
        try:
            ds.close()
        except Exception:
            pass
