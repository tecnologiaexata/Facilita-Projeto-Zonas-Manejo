from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from core.models import ZonesRequest
from core.pipeline import run_pipeline
from core.logger import get_logger
from core.publish import publish_zoneamento_raster

load_dotenv()


app = FastAPI(
    title="Zonas de Manejo API",
    description="API para geração de zonas de manejo nos modos threshold, auto e hotspot.",
    version="0.1.0",
)

logger = get_logger(name="zonas_manejo_api")


def _safe_log_path(area_name: str) -> Path:
    """
    Keep API-side logging simple:
    write one run log inside outputs/temp/{area_name}/run.log
    """
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", area_name).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "_", text).strip("_")
    area_name_sanitized = text or "area"

    return Path("outputs") / "temp" / area_name_sanitized / "run.log"


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "zonas_manejo_api",
    }


@app.post("/zones/generate")
def generate_zones(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single endpoint for all zone generation modes.

    Expected payload:
    - mode = threshold | auto | hotspot
    - AOI can be provided via local path (`aoi.id`) or public URL (`aoi.url`)
    - rasters can be provided via local path (`rasters[].path`) or public URL (`rasters[].url`)
    - validated by ZonesRequest
    """
    area_name_for_log = payload.get("job", {}).get("area_name", "area")
    run_logger = get_logger(
        name=f"zonas_manejo_api.{area_name_for_log}",
        log_file=_safe_log_path(area_name_for_log),
    )

    try:
        run_logger.info("Received /zones/generate request")

        cfg = ZonesRequest(**payload)

        run_logger.info(
            "Starting pipeline | mode=%s | area_name=%s | dry_run=%s | output_format=%s",
            cfg.mode,
            cfg.job.area_name,
            cfg.dry_run,
            cfg.output.format,
        )

        if cfg.mode == "threshold":
            run_logger.info(
                "Threshold mode | attribute=%s",
                cfg.mode_params.get("attribute") if cfg.mode_params else None,
            )

        elif cfg.mode == "auto":
            run_logger.info(
                "Auto mode | k=%s | min_zone_area_ha=%s",
                cfg.user_choices.k,
                cfg.user_choices.min_zone_area_ha,
            )

        elif cfg.mode == "hotspot":
            hotspot_mode = cfg.mode_params.get("hotspot_mode") if cfg.mode_params else None
            selected_attributes = cfg.mode_params.get("selected_attributes") if cfg.mode_params else None

            if hotspot_mode == "library":
                library_attrs = (
                    cfg.mode_params.get("classification_library", {}).get("attributes", {})
                    if cfg.mode_params else {}
                )
                run_logger.info(
                    "Hotspot mode | hotspot_mode=%s | selected_attributes=%s | library_attributes=%s",
                    hotspot_mode,
                    selected_attributes,
                    list(library_attrs.keys()),
                )

            elif hotspot_mode == "target":
                target_rules = cfg.mode_params.get("target_rules", []) if cfg.mode_params else []
                if isinstance(target_rules, list):
                    target_rule_attrs = [
                        rule.get("attribute")
                        for rule in target_rules
                        if isinstance(rule, dict) and rule.get("attribute")
                    ]
                elif isinstance(target_rules, dict):
                    target_rule_attrs = list(target_rules.keys())
                else:
                    target_rule_attrs = []

                run_logger.info(
                    "Hotspot mode | hotspot_mode=%s | selected_attributes=%s | target_rules=%s",
                    hotspot_mode,
                    selected_attributes,
                    target_rule_attrs,
                )

            else:
                run_logger.info(
                    "Hotspot mode | hotspot_mode=%s | selected_attributes=%s",
                    hotspot_mode,
                    selected_attributes,
                )

        result = run_pipeline(cfg=cfg, logger=run_logger)
        warnings: list[str] = []
        publication = None
        tif_url = None
        raster_id = None
        add_raster_payload = None
        external_response = None
        response_data = None

        if not cfg.dry_run:
            published_raster_path = (
                (result.outputs.get("paths") or {}).get("published_raster")
                or (result.outputs.get("paths") or {}).get("classified_raster")
            )
            publication = publish_zoneamento_raster(
                cfg,
                tif_path=published_raster_path,
                logger=run_logger,
            )
            warnings.extend(publication.warnings)
            tif_url = publication.tif_url
            raster_id = publication.raster_id
            add_raster_payload = publication.add_raster_payload
            external_response = publication.external_response
            if isinstance(external_response, dict) and isinstance(external_response.get("data"), dict):
                response_data = external_response.get("data")

            run_logger.info(
                "Publication finished | status=%s | tif_url=%s | raster_id=%s | warnings=%s",
                publication.status,
                tif_url,
                raster_id,
                warnings,
            )

        run_logger.info(
            "Pipeline finished successfully | mode=%s | area_name=%s",
            cfg.mode,
            cfg.job.area_name,
        )

        # do NOT return GeoDataFrames in API response
        response_payload = {
            "status": result.status,
            "mode": result.mode,
            "dry_run": result.dry_run,
            "reports": result.reports,
            "paths": result.outputs.get("paths"),
            "preview": result.outputs.get("preview"),
        }
        if publication is not None:
            response_payload.update(
                {
                    "publication": {
                        "status": publication.status,
                        "url_tif": tif_url,
                        "raster_id": raster_id,
                        "add_raster_payload": add_raster_payload,
                    },
                    "url_tif": tif_url,
                    "raster_id": raster_id,
                    "id": raster_id,
                    "add_raster_payload": add_raster_payload,
                    "external_response": external_response,
                }
            )

            if response_data is not None:
                response_payload["data"] = response_data

        if warnings:
            response_payload["warnings"] = warnings

        return response_payload

    except ValidationError as e:
        run_logger.error("Validation error: %s", e)
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid request payload.",
                "errors": e.errors(),
            },
        )

    except NotImplementedError as e:
        run_logger.error("Not implemented: %s", e)
        raise HTTPException(
            status_code=501,
            detail={"message": str(e)},
        )

    except Exception as e:
        tb = traceback.format_exc()
        run_logger.error("Unhandled error: %s", str(e))
        run_logger.error(tb)

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Internal pipeline error.",
                "error": str(e),
            },
        )
