"""Bitácora local de visitas de obra (JSON + fotos en disco)."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from src.data_loader import PROJECT_ROOT

INSPECTIONS_DIR: Path = PROJECT_ROOT / "data" / "inspections"
VISIT_FILENAME = "visit.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_local_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return (slug[:48] if slug else "visita")


def _visit_dir(visit_id: str) -> Path:
    return INSPECTIONS_DIR / visit_id


def _visit_path(visit_id: str) -> Path:
    return _visit_dir(visit_id) / VISIT_FILENAME


def _write_visit(visit: dict[str, Any]) -> None:
    path = _visit_path(str(visit["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "photos").mkdir(exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(visit, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_visit(visit_id: str) -> dict[str, Any]:
    path = _visit_path(visit_id)
    if not path.exists():
        raise FileNotFoundError(f"No existe la visita {visit_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def create_visit(place: str) -> dict[str, Any]:
    place_clean = " ".join(place.split()).strip()
    if not place_clean:
        raise ValueError("Indique el nombre del lugar.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visit_id = f"{stamp}_{_slug(place_clean)}"
    visit = {
        "schema": "crackexpert.ai/visita/v1",
        "id": visit_id,
        "place": place_clean,
        "created_at": _now_iso(),
        "created_at_local": _now_local_label(),
        "updated_at": _now_iso(),
        "records": [],
    }
    INSPECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    _write_visit(visit)
    return visit


def list_visits() -> list[dict[str, Any]]:
    if not INSPECTIONS_DIR.exists():
        return []
    visits: list[dict[str, Any]] = []
    for folder in INSPECTIONS_DIR.iterdir():
        path = folder / VISIT_FILENAME
        if not path.is_file():
            continue
        try:
            visits.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    visits.sort(key=lambda v: str(v.get("created_at", "")), reverse=True)
    return visits


def summarize_visit(visit: dict[str, Any]) -> dict[str, int]:
    counts = {"total": 0, "Crítica": 0, "Moderada": 0, "Leve": 0, "Sin fisura": 0}
    for rec in visit.get("records", []):
        counts["total"] += 1
        sev = str((rec.get("expert_verdict") or {}).get("severity", "Sin fisura"))
        if sev in counts:
            counts[sev] += 1
    return counts


def photo_path(visit_id: str, image_file: str) -> Path:
    return _visit_dir(visit_id) / image_file


def add_record(
    visit_id: str,
    image: Image.Image,
    *,
    elemento: str,
    ambiente: str,
    perception: dict[str, Any],
    expert_verdict: dict[str, Any],
) -> dict[str, Any]:
    visit = load_visit(visit_id)
    n = len(visit.get("records", [])) + 1
    rec_id = f"{n:03d}"
    rel = f"photos/{rec_id}.jpg"
    dest = photo_path(visit_id, rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    buf = BytesIO()
    rgb.save(buf, format="JPEG", quality=88)
    dest.write_bytes(buf.getvalue())

    record = {
        "id": rec_id,
        "captured_at": _now_iso(),
        "captured_at_local": _now_local_label(),
        "elemento": elemento,
        "ambiente": ambiente,
        "image_file": rel,
        "perception": perception,
        "expert_verdict": expert_verdict,
    }
    visit.setdefault("records", []).append(record)
    visit["updated_at"] = _now_iso()
    _write_visit(visit)
    return record


def update_record(visit_id: str, record_id: str, **fields: Any) -> dict[str, Any]:
    visit = load_visit(visit_id)
    for rec in visit.get("records", []):
        if rec.get("id") == record_id:
            rec.update(fields)
            visit["updated_at"] = _now_iso()
            _write_visit(visit)
            return rec
    raise KeyError(f"No está el registro {record_id} en la visita {visit_id}")
