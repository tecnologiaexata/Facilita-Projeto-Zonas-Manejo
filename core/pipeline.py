from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import re
import unicodedata

from core.models import ZonesRequest
from core.io import build_inputs_report, close_inputs, load_inputs
from core.alignment import check_alignment, align_to_aoi_grid
from core.threshold_preview import compute_threshold_preview
from core.classification_threshold import run_threshold_classification, write_zone_raster
from core.classification_hotspot import run_hotspot_library, run_hotspot_target
from core.classification_auto import run_auto_classification
from core.polygonize import polygons_to_raster, raster_to_polygons, write_polygons
from core.vector_postprocess import enforce_min_polygon_area
from core.smoothing import smooth_and_fill
from core.statistics import compute_zone_statistics


@dataclass
class PipelineResult:
    mode: str
    dry_run: bool
    status: str
    outputs: Dict[str, Any]
    reports: Dict[str, Any]


def _sanitize_name(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "_", text).strip("_")
    return text or "area"


def _ensure_output_dir(base_dir: str | Path) -> Path:
    out = Path(base_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _get_output_paths(cfg: ZonesRequest, mode: str) -> Dict[str, Path]:
    area_name = _sanitize_name(cfg.job.area_name)

    temp_dir = _ensure_output_dir(Path("outputs") / "temp" / area_name)
    final_dir = _ensure_output_dir(Path("outputs") / area_name)

    final_ext = "gpkg" if cfg.output.format == "gpkg" else "shp"

    return {
        "area_name": Path(area_name),
        "temp_dir": temp_dir,
        "final_dir": final_dir,
        "threshold_raster": temp_dir / f"zones_{mode}.tif",
        "published_raster": final_dir / f"zones_{mode}_final.tif",
        "polygons_raw": temp_dir / f"zones_{mode}_raw.gpkg",
        "polygons_minarea": temp_dir / f"zones_{mode}_minarea.gpkg",
        "polygons_smoothed": temp_dir / f"zones_{mode}_smoothed.gpkg",
        "zones_internal": temp_dir / f"zones_{mode}_internal.gpkg",
        "zones_final": final_dir / f"zones_{mode}_final.{final_ext}",
    }


def _write_vector(gdf, out_path: Path, file_format: str, layer: str) -> str:
    if file_format == "gpkg":
        write_polygons(gdf, str(out_path), driver="GPKG", layer=layer)
    elif file_format == "shp":
        write_polygons(gdf, str(out_path), driver="ESRI Shapefile")
    else:
        raise ValueError(f"Unsupported vector output format: {file_format}")
    return str(out_path)


def _add_hotspot_labels(gdf, hotspot_mode: str):
    gdf = gdf.copy()

    if hotspot_mode == "library":
        label_map = {
            1: ("top_1", "atende totalmente"),
            2: ("top_2", "atende parcialmente"),
            3: ("top_3", "nao atende"),
        }
    elif hotspot_mode == "target":
        label_map = {
            1: ("top_1", "atende ao alvo"),
            2: ("top_2", "nao atende, mas esta mais proximo do alvo"),
            3: ("top_3", "nao atende e esta mais distante do alvo"),
        }
    else:
        raise ValueError(f"Unsupported hotspot_mode: {hotspot_mode}")

    gdf["zone_rank"] = gdf["zone_id"].astype(int)
    gdf["zone_label"] = gdf["zone_id"].map(lambda z: label_map[int(z)][0])
    gdf["zone_meaning"] = gdf["zone_id"].map(lambda z: label_map[int(z)][1])
    gdf["hotspot_mode"] = hotspot_mode

    return gdf


def run_pipeline(
    cfg: ZonesRequest,
    alignment_mode: str = "auto_fix",   # "strict" or "auto_fix"
    target_cell_m: float = 10.0,
    export_intermediate: bool = True,
    export_final: bool = True,
    smoothing_params: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> PipelineResult:
    """
    Orchestrates the zones pipeline.

    Current implementation:
    - threshold mode fully supported
    - threshold dry_run supported
    - auto mode fully supported
    - hotspot mode fully supported
    """
    paths = _get_output_paths(cfg, cfg.mode)
    if logger:
        logger.info(
            "Pipeline | initialized | mode=%s | area_name=%s | temp_dir=%s | final_dir=%s",
            cfg.mode,
            cfg.job.area_name,
            paths["temp_dir"],
            paths["final_dir"],
        )

    if smoothing_params is None:
        smoothing_params = {
            "epsilon": 8.0,
            "chaikin_iterations": 2,
            "chaikin_offset": 0.25,
            "min_gap_area_ha": 0.0,
        }

    inputs = None
    aligned = None

    try:
        # ----------------------------------
        # 1) Load inputs
        # ----------------------------------
        inputs = load_inputs(cfg, logger=logger)
        inputs_report = build_inputs_report(inputs)
        if logger:
            logger.info("Pipeline | inputs loaded")

        # ----------------------------------
        # 2) Alignment
        # ----------------------------------
        alignment_report: Dict[str, Any] = {"mode": alignment_mode}

        if alignment_mode == "strict":
            check_alignment(inputs, target_cell_m=target_cell_m)
            aligned_rasters = inputs.rasters
            alignment_report["status"] = "passed_strict"
            alignment_report["details"] = None
            if logger:
                logger.info("Pipeline | alignment strict passed")
        elif alignment_mode == "auto_fix":
            aligned = align_to_aoi_grid(
                inputs,
                target_cell_m=target_cell_m,
                out_dir=paths["temp_dir"],
                logger=logger,
            )
            aligned_rasters = aligned.rasters
            alignment_report["status"] = "auto_fixed"
            alignment_report["details"] = aligned.report
            if logger:
                logger.info("Pipeline | alignment auto-fix completed")
        else:
            raise ValueError("alignment_mode must be 'strict' or 'auto_fix'.")

        # ----------------------------------
        # 3) Threshold mode
        # ----------------------------------
        if cfg.mode == "threshold":
            attr = cfg.mode_params["attribute"]
            ds = aligned_rasters[attr]
            if logger:
                logger.info("Pipeline | threshold mode | attribute=%s", attr)

            if cfg.dry_run:
                if logger:
                    logger.info("Pipeline | threshold dry-run | computing preview")
                preview = compute_threshold_preview(
                    aoi=inputs.aoi,
                    ds=ds,
                    attribute=attr,
                    units=cfg.mode_params.get("units", "same_as_raster"),
                    n_classes=3,
                )

                return PipelineResult(
                    mode=cfg.mode,
                    dry_run=True,
                    status="success",
                    outputs={
                        "preview": {
                            "attribute": preview.attribute,
                            "units": preview.units,
                            "stats": preview.stats,
                            "n_valid": preview.n_valid,
                            "suggested_classes": preview.suggested_classes,
                        }
                    },
                    reports={
                        "inputs": inputs_report,
                        "alignment": alignment_report,
                        "job": {
                            "area_name": cfg.job.area_name,
                            "temp_dir": str(paths["temp_dir"]),
                            "final_dir": str(paths["final_dir"]),
                        }
                    }
                )

            class_res = run_threshold_classification(
                aoi=inputs.aoi,
                ds=ds,
                classes=cfg.mode_params["classes"],
            )
            if logger:
                logger.info("Pipeline | threshold classification done | stats=%s", class_res.stats)

            classified_raster_path = None
            if export_intermediate:
                classified_raster_path = str(paths["threshold_raster"])
                write_zone_raster(
                    classified_raster_path,
                    class_res.zone_arr,
                    class_res.profile
                )

            poly_res = raster_to_polygons(
                zone_arr=class_res.zone_arr,
                transform=ds.transform,
                crs=ds.crs,
                nodata_id=0
            )
            if logger:
                logger.info("Pipeline | threshold polygonize done | stats=%s", poly_res.stats)

            polygons_raw_path = None
            if export_intermediate:
                polygons_raw_path = _write_vector(
                    poly_res.gdf,
                    paths["polygons_raw"],
                    "gpkg",
                    "zones_raw"
                )

            merged_res = enforce_min_polygon_area(
                gdf=poly_res.gdf,
                min_area_ha=cfg.user_choices.min_zone_area_ha,
                zone_field="zone_id"
            )
            if logger:
                logger.info("Pipeline | threshold min area merge done | stats=%s", merged_res.stats)

            polygons_minarea_path = None
            if export_intermediate:
                polygons_minarea_path = _write_vector(
                    merged_res.gdf,
                    paths["polygons_minarea"],
                    "gpkg",
                    "zones_minarea"
                )

            smooth_res = smooth_and_fill(
                zones=merged_res.gdf,
                area_gdf=inputs.aoi,
                id_field="poly_id",
                epsilon=smoothing_params.get("epsilon", 8.0),
                chaikin_iterations=smoothing_params.get("chaikin_iterations", 2),
                chaikin_offset=smoothing_params.get("chaikin_offset", 0.25),
                min_gap_area_ha=smoothing_params.get("min_gap_area_ha", 0.0),
            )
            if logger:
                logger.info("Pipeline | threshold smoothing done | stats=%s", smooth_res.stats)

            polygons_smoothed_path = None
            if export_intermediate:
                polygons_smoothed_path = _write_vector(
                    smooth_res.gdf,
                    paths["polygons_smoothed"],
                    "gpkg",
                    "zones_smoothed"
                )

            stats_res = compute_zone_statistics(
                zones_gdf=smooth_res.gdf,
                rasters=aligned_rasters,
                zone_field="zone_id"
            )
            if logger:
                logger.info("Pipeline | threshold statistics done | summary=%s", stats_res.summary)

            final_zones_path = None
            internal_polygons_path = None
            published_raster_path = None

            if export_final:
                final_zones_path = _write_vector(
                    stats_res.stats_gdf,
                    paths["zones_final"],
                    cfg.output.format,
                    "zones_final"
                )
                published_zone_arr, published_zone_profile = polygons_to_raster(
                    stats_res.stats_gdf,
                    zone_field="zone_id",
                    reference_profile=class_res.profile,
                )
                published_raster_path = str(paths["published_raster"])
                write_zone_raster(
                    published_raster_path,
                    published_zone_arr,
                    published_zone_profile,
                )
                if logger:
                    logger.info(
                        "Pipeline | threshold final exports written | vector=%s | tif=%s",
                        final_zones_path,
                        published_raster_path,
                    )

            if export_intermediate:
                internal_polygons_path = _write_vector(
                    stats_res.zones_gdf,
                    paths["zones_internal"],
                    "gpkg",
                    "zones_internal"
                )

            return PipelineResult(
                mode=cfg.mode,
                dry_run=False,
                status="success",
                outputs={
                    "final_gdf": stats_res.stats_gdf,
                    "internal_gdf": stats_res.zones_gdf,
                    "paths": {
                        "temp_dir": str(paths["temp_dir"]),
                        "final_dir": str(paths["final_dir"]),
                        "classified_raster": classified_raster_path,
                        "published_raster": published_raster_path,
                        "polygons_raw": polygons_raw_path,
                        "polygons_minarea": polygons_minarea_path,
                        "polygons_smoothed": polygons_smoothed_path,
                        "zones_final": final_zones_path,
                        "zones_internal": internal_polygons_path,
                    }
                },
                reports={
                    "inputs": inputs_report,
                    "job": {
                        "area_name": cfg.job.area_name,
                        "area_name_sanitized": str(paths["area_name"]),
                    },
                    "alignment": alignment_report,
                    "classification": class_res.stats,
                    "polygonize": poly_res.stats,
                    "min_area_merge": merged_res.stats,
                    "smoothing": smooth_res.stats,
                    "statistics": stats_res.summary,
                }
            )

        # ----------------------------------
        # 4) Auto mode
        # ----------------------------------
        elif cfg.mode == "auto":
            if logger:
                logger.info(
                    "Pipeline | auto mode | k=%s | min_zone_area_ha=%s",
                    cfg.user_choices.k,
                    cfg.user_choices.min_zone_area_ha,
                )
            auto_res = run_auto_classification(
                rasters=aligned_rasters,
                k=cfg.user_choices.k,
                k_min=2,
                k_max=8,
            )
            if logger:
                logger.info("Pipeline | auto classification done | stats=%s", auto_res.stats)

            first_ds = next(iter(aligned_rasters.values()))

            classified_raster_path = None
            if export_intermediate:
                classified_raster_path = str(paths["threshold_raster"])
                write_zone_raster(
                    classified_raster_path,
                    auto_res.zone_arr,
                    auto_res.profile
                )

            poly_res = raster_to_polygons(
                zone_arr=auto_res.zone_arr,
                transform=first_ds.transform,
                crs=first_ds.crs,
                nodata_id=0
            )
            if logger:
                logger.info("Pipeline | auto polygonize done | stats=%s", poly_res.stats)

            polygons_raw_path = None
            if export_intermediate:
                polygons_raw_path = _write_vector(
                    poly_res.gdf,
                    paths["polygons_raw"],
                    "gpkg",
                    "zones_raw"
                )

            merged_res = enforce_min_polygon_area(
                gdf=poly_res.gdf,
                min_area_ha=cfg.user_choices.min_zone_area_ha,
                zone_field="zone_id"
            )
            if logger:
                logger.info("Pipeline | auto min area merge done | stats=%s", merged_res.stats)

            polygons_minarea_path = None
            if export_intermediate:
                polygons_minarea_path = _write_vector(
                    merged_res.gdf,
                    paths["polygons_minarea"],
                    "gpkg",
                    "zones_minarea"
                )

            smooth_res = smooth_and_fill(
                zones=merged_res.gdf,
                area_gdf=inputs.aoi,
                id_field="poly_id",
                epsilon=smoothing_params.get("epsilon", 8.0),
                chaikin_iterations=smoothing_params.get("chaikin_iterations", 2),
                chaikin_offset=smoothing_params.get("chaikin_offset", 0.25),
                min_gap_area_ha=smoothing_params.get("min_gap_area_ha", 0.0),
            )
            if logger:
                logger.info("Pipeline | auto smoothing done | stats=%s", smooth_res.stats)

            polygons_smoothed_path = None
            if export_intermediate:
                polygons_smoothed_path = _write_vector(
                    smooth_res.gdf,
                    paths["polygons_smoothed"],
                    "gpkg",
                    "zones_smoothed"
                )

            stats_res = compute_zone_statistics(
                zones_gdf=smooth_res.gdf,
                rasters=aligned_rasters,
                zone_field="zone_id"
            )
            if logger:
                logger.info("Pipeline | auto statistics done | summary=%s", stats_res.summary)

            final_zones_path = None
            internal_polygons_path = None
            published_raster_path = None

            if export_final:
                final_zones_path = _write_vector(
                    stats_res.stats_gdf,
                    paths["zones_final"],
                    cfg.output.format,
                    "zones_final"
                )
                published_zone_arr, published_zone_profile = polygons_to_raster(
                    stats_res.stats_gdf,
                    zone_field="zone_id",
                    reference_profile=auto_res.profile,
                )
                published_raster_path = str(paths["published_raster"])
                write_zone_raster(
                    published_raster_path,
                    published_zone_arr,
                    published_zone_profile,
                )
                if logger:
                    logger.info(
                        "Pipeline | auto final exports written | vector=%s | tif=%s",
                        final_zones_path,
                        published_raster_path,
                    )

            if export_intermediate:
                internal_polygons_path = _write_vector(
                    stats_res.zones_gdf,
                    paths["zones_internal"],
                    "gpkg",
                    "zones_internal"
                )

            return PipelineResult(
                mode=cfg.mode,
                dry_run=False,
                status="success",
                outputs={
                    "final_gdf": stats_res.stats_gdf,
                    "internal_gdf": stats_res.zones_gdf,
                    "paths": {
                        "temp_dir": str(paths["temp_dir"]),
                        "final_dir": str(paths["final_dir"]),
                        "classified_raster": classified_raster_path,
                        "published_raster": published_raster_path,
                        "polygons_raw": polygons_raw_path,
                        "polygons_minarea": polygons_minarea_path,
                        "polygons_smoothed": polygons_smoothed_path,
                        "zones_final": final_zones_path,
                        "zones_internal": internal_polygons_path,
                    }
                },
                reports={
                    "inputs": inputs_report,
                    "job": {
                        "area_name": cfg.job.area_name,
                        "area_name_sanitized": str(paths["area_name"]),
                    },
                    "alignment": alignment_report,
                    "classification": auto_res.stats,
                    "polygonize": poly_res.stats,
                    "min_area_merge": merged_res.stats,
                    "smoothing": smooth_res.stats,
                    "statistics": stats_res.summary,
                }
            )


        # ----------------------------------
        # 5) Hotspot mode
        # ----------------------------------
        elif cfg.mode == "hotspot":
            hotspot_mode = cfg.mode_params.get("hotspot_mode")
            selected_attributes = cfg.mode_params.get("selected_attributes", [])
            if logger:
                logger.info(
                    "Pipeline | hotspot mode | hotspot_mode=%s | selected_attributes=%s | min_zone_area_ha=%s",
                    hotspot_mode,
                    selected_attributes,
                    cfg.user_choices.min_zone_area_ha,
                )

            if hotspot_mode == "library":
                hotspot_res = run_hotspot_library(
                    rasters=aligned_rasters,
                    selected_attributes=selected_attributes,
                    classification_library=cfg.mode_params["classification_library"],
                )

            elif hotspot_mode == "target":
                hotspot_res = run_hotspot_target(
                    rasters=aligned_rasters,
                    selected_attributes=selected_attributes,
                    target_rules=cfg.mode_params["target_rules"],
                    negative_split_quantile=cfg.mode_params.get("negative_split_quantile", 0.5),
                )

            else:
                raise ValueError("hotspot_mode inválido.")
            if logger:
                logger.info("Pipeline | hotspot classification done | stats=%s", hotspot_res.stats)

            first_ds = next(iter(aligned_rasters.values()))

            classified_raster_path = None
            if export_intermediate:
                classified_raster_path = str(paths["threshold_raster"])
                write_zone_raster(
                    classified_raster_path,
                    hotspot_res.zone_arr,
                    hotspot_res.profile
                )

            poly_res = raster_to_polygons(
                zone_arr=hotspot_res.zone_arr,
                transform=first_ds.transform,
                crs=first_ds.crs,
                nodata_id=0
            )
            if logger:
                logger.info("Pipeline | hotspot polygonize done | stats=%s", poly_res.stats)

            polygons_raw_path = None
            if export_intermediate:
                polygons_raw_path = _write_vector(
                    poly_res.gdf,
                    paths["polygons_raw"],
                    "gpkg",
                    "zones_raw"
                )

            merged_res = enforce_min_polygon_area(
                gdf=poly_res.gdf,
                min_area_ha=cfg.user_choices.min_zone_area_ha,
                zone_field="zone_id"
            )
            if logger:
                logger.info("Pipeline | hotspot min area merge done | stats=%s", merged_res.stats)

            polygons_minarea_path = None
            if export_intermediate:
                polygons_minarea_path = _write_vector(
                    merged_res.gdf,
                    paths["polygons_minarea"],
                    "gpkg",
                    "zones_minarea"
                )

            smooth_res = smooth_and_fill(
                zones=merged_res.gdf,
                area_gdf=inputs.aoi,
                id_field="poly_id",
                epsilon=smoothing_params.get("epsilon", 8.0),
                chaikin_iterations=smoothing_params.get("chaikin_iterations", 2),
                chaikin_offset=smoothing_params.get("chaikin_offset", 0.25),
                min_gap_area_ha=smoothing_params.get("min_gap_area_ha", 0.0),
            )
            if logger:
                logger.info("Pipeline | hotspot smoothing done | stats=%s", smooth_res.stats)

            polygons_smoothed_path = None
            if export_intermediate:
                polygons_smoothed_path = _write_vector(
                    smooth_res.gdf,
                    paths["polygons_smoothed"],
                    "gpkg",
                    "zones_smoothed"
                )

            stats_res = compute_zone_statistics(
                zones_gdf=smooth_res.gdf,
                rasters=aligned_rasters,
                zone_field="zone_id"
            )
            if logger:
                logger.info("Pipeline | hotspot statistics done | summary=%s", stats_res.summary)

            final_gdf = _add_hotspot_labels(stats_res.stats_gdf, hotspot_mode)
            internal_gdf = stats_res.zones_gdf.copy()

            final_zones_path = None
            internal_polygons_path = None
            published_raster_path = None

            if export_final:
                final_zones_path = _write_vector(
                    final_gdf,
                    paths["zones_final"],
                    cfg.output.format,
                    "zones_final"
                )
                published_zone_arr, published_zone_profile = polygons_to_raster(
                    final_gdf,
                    zone_field="zone_id",
                    reference_profile=hotspot_res.profile,
                )
                published_raster_path = str(paths["published_raster"])
                write_zone_raster(
                    published_raster_path,
                    published_zone_arr,
                    published_zone_profile,
                )
                if logger:
                    logger.info(
                        "Pipeline | hotspot final exports written | vector=%s | tif=%s",
                        final_zones_path,
                        published_raster_path,
                    )

            if export_intermediate:
                internal_polygons_path = _write_vector(
                    internal_gdf,
                    paths["zones_internal"],
                    "gpkg",
                    "zones_internal"
                )

            return PipelineResult(
                mode=cfg.mode,
                dry_run=False,
                status="success",
                outputs={
                    "final_gdf": final_gdf,
                    "internal_gdf": internal_gdf,
                    "paths": {
                        "temp_dir": str(paths["temp_dir"]),
                        "final_dir": str(paths["final_dir"]),
                        "classified_raster": classified_raster_path,
                        "published_raster": published_raster_path,
                        "polygons_raw": polygons_raw_path,
                        "polygons_minarea": polygons_minarea_path,
                        "polygons_smoothed": polygons_smoothed_path,
                        "zones_final": final_zones_path,
                        "zones_internal": internal_polygons_path,
                    }
                },
                reports={
                    "inputs": inputs_report,
                    "job": {
                        "area_name": cfg.job.area_name,
                        "area_name_sanitized": str(paths["area_name"]),
                    },
                    "alignment": alignment_report,
                    "classification": hotspot_res.stats,
                    "polygonize": poly_res.stats,
                    "min_area_merge": merged_res.stats,
                    "smoothing": smooth_res.stats,
                    "statistics": stats_res.summary,
                }
            )

        else:
            raise ValueError(f"Unsupported mode: {cfg.mode}")

    finally:
        if aligned is not None:
            for ds in aligned.rasters.values():
                try:
                    ds.close()
                except Exception:
                    pass

        if inputs is not None:
            close_inputs(inputs)
