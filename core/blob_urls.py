from __future__ import annotations

import json
import os
from typing import Optional
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


BLOB_STORAGE_HOST_FRAGMENT = "blob.vercel-storage.com"


def is_http_url(raw_url: Optional[str]) -> bool:
    if not isinstance(raw_url, str):
        return False

    cleaned = raw_url.strip()
    if not cleaned:
        return False

    try:
        parsed = urlparse(cleaned)
    except Exception:
        return False

    return parsed.scheme in {"http", "https"} and bool((parsed.netloc or "").strip())


def resolve_blob_token() -> Optional[str]:
    token = (
        os.getenv("BLOB_READ_WRITE_TOKEN")
        or os.getenv("NEXT_PUBLIC_BLOB_READ_WRITE_TOKEN")
        or ""
    ).strip()
    return token or None


def is_blob_storage_url(raw_url: Optional[str]) -> bool:
    if not is_http_url(raw_url):
        return False

    parsed = urlparse(str(raw_url).strip())
    hostname = (parsed.hostname or "").strip().lower()
    return BLOB_STORAGE_HOST_FRAGMENT in hostname


def build_blob_request_headers(raw_url: Optional[str]) -> dict[str, str]:
    if not is_blob_storage_url(raw_url):
        return {}

    token = resolve_blob_token()
    if not token:
        return {}

    return {"Authorization": f"Bearer {token}"}


def resolve_blob_download_url(
    raw_url: Optional[str],
    *,
    frontend_base_url: Optional[str] = None,
    timeout: int = 30,
) -> Optional[str]:
    if not isinstance(raw_url, str):
        return raw_url

    cleaned = raw_url.strip()
    if not cleaned or not is_blob_storage_url(cleaned):
        return cleaned or raw_url

    base_url = (frontend_base_url or os.getenv("FACILITAGRO_FRONTEND_BASE_URL") or "").strip()
    if not base_url:
        return cleaned

    endpoint = (
        f"{base_url.rstrip('/')}/api/v1/blobs/listBlob?"
        f"{urlencode({'url': cleaned})}"
    )
    request = Request(endpoint, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
    except Exception:
        return cleaned

    try:
        payload = json.loads(raw_body) if raw_body else {}
    except Exception:
        return cleaned

    if not isinstance(payload, dict):
        return cleaned

    resolved = payload.get("downloadUrl") or payload.get("url")
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()

    return cleaned
