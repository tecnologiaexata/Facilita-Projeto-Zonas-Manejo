from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError as UrlHTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from core.blob_urls import resolve_blob_download_url
from core.models import ZonesRequest


DEFAULT_BLOB_UPLOAD_BASE_URL = "https://blob.vercel-storage.com"
BLOB_UPLOAD_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
BLOB_UPLOAD_ATTEMPTS = 3
DEFAULT_TIPO_IMPORTACAO_ZONEAMENTO = 8


@dataclass
class PublicationResult:
    status: str
    tif_url: Optional[str]
    raster_id: Optional[int]
    add_raster_payload: Optional[Dict[str, Any]]
    external_response: Optional[Dict[str, Any]]
    warnings: List[str]


def _sanitize_key(value: Any, fallback: str = "item") -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode(
        "ascii", "ignore"
    ).decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "_", text).strip("_").lower()
    return text or fallback


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _coalesce_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is None:
            continue

        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError):
            continue

        if parsed > 0:
            return parsed

    return None


def _coalesce_text(*values: Any) -> Optional[str]:
    value = _coalesce(*values)
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _normalize_usuario_ids(value: Any) -> Optional[List[int]]:
    if value is None:
        return None

    source = value if isinstance(value, list) else [value]
    normalized: List[int] = []
    seen = set()

    for item in source:
        parsed = _coalesce_int(item)
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        normalized.append(parsed)

    return normalized or None


def _resolve_blob_token() -> str:
    token = (
        os.getenv("BLOB_READ_WRITE_TOKEN")
        or os.getenv("NEXT_PUBLIC_BLOB_READ_WRITE_TOKEN")
        or ""
    ).strip()
    if not token:
        raise RuntimeError(
            "Env BLOB_READ_WRITE_TOKEN/NEXT_PUBLIC_BLOB_READ_WRITE_TOKEN não configurada."
        )
    return token


def _build_blob_upload_url(blob_name: str) -> str:
    cleaned_name = str(blob_name or "").strip().lstrip("/")
    if not cleaned_name:
        raise ValueError("Nome do blob não informado.")

    base_url = (
        os.getenv("BLOB_UPLOAD_BASE_URL")
        or os.getenv("BLOB_STORAGE_UPLOAD_BASE_URL")
        or os.getenv("BLOB_STORAGE_BASE_URL")
        or DEFAULT_BLOB_UPLOAD_BASE_URL
    ).strip()
    if not base_url:
        raise RuntimeError("URL base do blob para upload não configurada.")

    encoded_name = "/".join(
        quote(segment, safe="") for segment in cleaned_name.split("/") if segment
    )
    return f"{base_url.rstrip('/')}/{encoded_name}"


def _should_retry_blob_upload_error(error: Exception) -> bool:
    if isinstance(error, UrlHTTPError):
        return error.code in BLOB_UPLOAD_RETRYABLE_STATUS

    return isinstance(error, (URLError, TimeoutError, OSError))


def _upload_blob_bytes(*, file_bytes: bytes, blob_name: str, content_type: str) -> str:
    token = _resolve_blob_token()
    upload_url = _build_blob_upload_url(blob_name)
    last_error: Exception | None = None

    for attempt in range(1, BLOB_UPLOAD_ATTEMPTS + 1):
        req = Request(
            upload_url,
            data=file_bytes,
            method="PUT",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": content_type,
                "x-add-random-suffix": "1",
                "x-content-type": content_type,
            },
        )

        try:
            with urlopen(req, timeout=60) as resp:
                raw_body = resp.read().decode("utf-8")
                body = json.loads(raw_body) if raw_body else {}
                blob_url = (
                    body.get("downloadUrl")
                    or body.get("url")
                    or resp.headers.get("x-vercel-blob-url")
                )
                if not blob_url:
                    raise RuntimeError("Blob respondeu sem URL do arquivo.")
                return str(resolve_blob_download_url(str(blob_url)))
        except Exception as error:
            last_error = error
            if attempt < BLOB_UPLOAD_ATTEMPTS and _should_retry_blob_upload_error(error):
                time.sleep(min(2 ** (attempt - 1), 5))
                continue
            raise RuntimeError(f"Falha ao enviar arquivo para Blob: {error}") from error

    raise RuntimeError(f"Falha ao enviar arquivo para Blob: {last_error}")


def _upload_file_to_blob(path: Path, blob_name: str, content_type: Optional[str] = None) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo para upload não encontrado: {file_path}")

    resolved_content_type = (
        content_type
        or mimetypes.guess_type(str(file_path))[0]
        or "application/octet-stream"
    )

    return _upload_blob_bytes(
        file_bytes=file_path.read_bytes(),
        blob_name=blob_name,
        content_type=resolved_content_type,
    )


def _build_blob_name(cfg: ZonesRequest, tif_path: Path) -> str:
    gleba_identifier = cfg.gleba_id or cfg.gleba
    tipo = _sanitize_key(cfg.tipo or ("gleba" if gleba_identifier else "talhao"), "tipo")
    identificador = _sanitize_key(
        cfg.id or cfg.talhao_id or cfg.talhao or gleba_identifier or "id",
        "id",
    )
    processo = _sanitize_key(cfg.processo or "Zoneamento", "zoneamento")
    campanha = _sanitize_key(cfg.safra_id or cfg.safra or cfg.job.area_name, "campanha")
    area_name = _sanitize_key(cfg.job.area_name, "area")
    return (
        f"rasters/zoneamento/{tipo}_{identificador}_{processo}_{campanha}_{area_name}"
        f"{tif_path.suffix or '.tif'}"
    )


def build_add_raster_payload(
    cfg: ZonesRequest,
    *,
    tif_url: Optional[str],
    tif_local_path: Path,
) -> Dict[str, Any]:
    tipo = _coalesce_text(cfg.tipo)
    fazenda_id = _coalesce_int(cfg.fazenda, cfg.fazenda_id)
    talhao_id = _coalesce_int(cfg.talhao, cfg.talhao_id)
    gleba_id = _coalesce_int(cfg.gleba, cfg.gleba_id)
    profundidade_id = _coalesce_int(cfg.profundidade_id, cfg.profundidade)
    safra_id = _coalesce_int(cfg.safra_id, cfg.safra)
    usuario_ids = _normalize_usuario_ids(cfg.usuario_id_cadastro)
    descricao = _coalesce_text(
        cfg.descricao,
        cfg.metadata.get("notes") if isinstance(cfg.metadata, dict) else None,
    )
    area_identifier = _coalesce_int(
        cfg.id,
        talhao_id if tipo != "gleba" else gleba_id,
        talhao_id,
        gleba_id,
    )
    safra_text = _coalesce_text(cfg.safra, str(safra_id) if safra_id is not None else None)
    data_ref = _coalesce_text(cfg.data, cfg.date)

    payload = {
        "raster_id": _coalesce_int(cfg.raster_id),
        "tipo": tipo or ("gleba" if gleba_id and not talhao_id else "talhao"),
        "id": area_identifier,
        "processo": _coalesce_text(cfg.processo, "Zoneamento"),
        "tipo_importacao": _coalesce_int(cfg.tipo_importacao) or DEFAULT_TIPO_IMPORTACAO_ZONEAMENTO,
        "atributo": _coalesce_text(cfg.atributo, "Zoneamento"),
        "campanha": safra_text,
        "descricao": descricao,
        "url": _coalesce_text(tif_url),
        "id_amostragem": _coalesce_int(cfg.id_amostragem),
        "safra": safra_text,
        "safra_id": safra_id,
        "cliente_id": _coalesce_int(cfg.cliente_id),
        "usuario_id_cadastro": usuario_ids,
        "data": data_ref,
        "fazenda": fazenda_id,
        "fazenda_id": fazenda_id,
        "talhao": talhao_id,
        "talhao_id": talhao_id,
        "talhao_nome": _coalesce_text(cfg.talhao_nome),
        "gleba": gleba_id,
        "gleba_id": gleba_id,
        "profundidade": profundidade_id,
        "profundidade_id": profundidade_id,
        "caminho_tif_local": str(tif_local_path),
    }

    return {
        key: value
        for key, value in payload.items()
        if value is not None and value != ""
    }


def _missing_add_raster_fields(payload: Dict[str, Any]) -> List[str]:
    required_fields = [
        "tipo",
        "id",
        "processo",
        "tipo_importacao",
        "atributo",
        "url",
        "id_amostragem",
        "safra_id",
        "cliente_id",
        "usuario_id_cadastro",
        "data",
        "fazenda",
        "talhao",
        "profundidade",
    ]

    missing = []
    for field in required_fields:
        value = payload.get(field)
        if value is None:
            missing.append(field)
            continue

        if isinstance(value, list) and not value:
            missing.append(field)
            continue

        if isinstance(value, str) and not value.strip():
            missing.append(field)

    return missing


def _post_add_raster(payload: Dict[str, Any]) -> Dict[str, Any]:
    base_url = (os.getenv("FACILITAGRO_FRONTEND_BASE_URL") or "").strip()
    if not base_url:
        raise RuntimeError("Env FACILITAGRO_FRONTEND_BASE_URL não configurada.")

    endpoint = f"{base_url.rstrip('/')}/api/v2/add_raster_interpolados"
    req = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(req, timeout=60) as resp:
            raw_body = resp.read().decode("utf-8")
            return json.loads(raw_body) if raw_body else {"status": "ok"}
    except Exception as error:
        raise RuntimeError(f"Falha ao enviar para add_raster_interpolados: {error}") from error


def _extract_raster_id(response: Any) -> Optional[int]:
    if not isinstance(response, dict):
        return None

    candidates = [
        response.get("data", {}).get("id") if isinstance(response.get("data"), dict) else None,
        response.get("id"),
    ]

    return _coalesce_int(*candidates)


def publish_zoneamento_raster(
    cfg: ZonesRequest,
    *,
    tif_path: Optional[str | Path],
    logger: Optional[logging.Logger] = None,
) -> PublicationResult:
    warnings: List[str] = []

    if tif_path is None:
        return PublicationResult(
            status="skipped",
            tif_url=None,
            raster_id=None,
            add_raster_payload=None,
            external_response=None,
            warnings=["Nenhum TIFF final foi informado para publicação do zoneamento."],
        )

    local_tif_path = Path(tif_path)
    if not local_tif_path.exists():
        return PublicationResult(
            status="warning",
            tif_url=None,
            raster_id=None,
            add_raster_payload=None,
            external_response=None,
            warnings=[f"TIFF final do zoneamento não encontrado: {local_tif_path}"],
        )

    tif_url: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    external_response: Optional[Dict[str, Any]] = None
    raster_id: Optional[int] = None
    status = "success"
    if logger:
        logger.info("Publish | starting | tif_path=%s", local_tif_path)

    try:
        blob_name = _build_blob_name(cfg, local_tif_path)
        if logger:
            logger.info("Publish | uploading tif to blob | blob_name=%s", blob_name)
        tif_url = _upload_file_to_blob(
            local_tif_path,
            blob_name,
            content_type="image/tiff",
        )
        if logger:
            logger.info("Publish | tif uploaded | tif_url=%s", tif_url)
    except Exception as error:
        if logger:
            logger.error("Publish | tif upload failed | error=%s", error)
        return PublicationResult(
            status="warning",
            tif_url=None,
            raster_id=None,
            add_raster_payload=None,
            external_response=None,
            warnings=[str(error)],
        )

    payload = build_add_raster_payload(
        cfg,
        tif_url=tif_url,
        tif_local_path=local_tif_path,
    )
    if logger:
        logger.info(
            "Publish | add_raster payload built | keys=%s",
            sorted(payload.keys()),
        )

    missing_fields = _missing_add_raster_fields(payload)
    if missing_fields:
        if logger:
            logger.warning(
                "Publish | add_raster payload missing fields | fields=%s",
                missing_fields,
            )
        warnings.append(
            "Payload para add_raster_interpolados com campos ausentes: "
            + ", ".join(missing_fields)
        )
        status = "warning"
        return PublicationResult(
            status=status,
            tif_url=tif_url,
            raster_id=None,
            add_raster_payload=payload,
            external_response=None,
            warnings=warnings,
        )

    try:
        if logger:
            logger.info("Publish | notifying add_raster_interpolados")
        external_response = _post_add_raster(payload)
        raster_id = _extract_raster_id(external_response)
        if logger:
            logger.info(
                "Publish | add_raster response received | raster_id=%s",
                raster_id,
            )
        if raster_id is None:
            warnings.append(
                "Resposta do add_raster_interpolados recebida sem id do raster."
            )
            status = "warning"
    except Exception as error:
        if logger:
            logger.error("Publish | add_raster notification failed | error=%s", error)
        warnings.append(str(error))
        status = "warning"

    return PublicationResult(
        status=status,
        tif_url=tif_url,
        raster_id=raster_id,
        add_raster_payload=payload,
        external_response=external_response,
        warnings=warnings,
    )
