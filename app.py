"""CrackExpert AI — visita de obra: fotos + 2 datos + bitácora local."""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crack_geometry import analyze_orientation  # noqa: E402
from src.data_loader import IMAGE_SIZE, PROJECT_ROOT  # noqa: E402
from src.expert_system import (  # noqa: E402
    ELEMENT_OPTIONS,
    EXPOSURE_OPTIONS,
    FIELD_QUESTIONS,
    PATRON_ORIENTACION_OPTIONS,
    cf_from_ml,
    diagnose,
    patron_from_geometry,
    patron_with_field_answers,
    verdict_to_dict,
)
from src.inspections import (  # noqa: E402
    add_record,
    create_visit,
    list_visits,
    load_visit,
    photo_path,
    summarize_visit,
    update_record,
)

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_NAME = "mobilenet_v2.keras"

SEVERITY_STYLE: dict[str, tuple[str, str]] = {
    "Crítica": ("#B71C1C", "#FFEBEE"),
    "Moderada": ("#E65100", "#FFF3E0"),
    "Leve": ("#2E7D32", "#E8F5E9"),
    "Sin fisura": ("#1565C0", "#E3F2FD"),
}

MEANING: dict[str, str] = {
    "Sin fisura": "En esta foto no se ve una fisura clara. Siga con la inspección habitual.",
    "Leve": "Hay una fisura, pero el riesgo inmediato parece bajo. Conviene vigilarla.",
    "Moderada": "La fisura puede afectar durabilidad (humedad o corrosión). Conviene actuar pronto.",
    "Crítica": "El patrón sugiere un mecanismo serio. Un ingeniero debe revisar el elemento.",
}


def _discover_keras_models() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.keras"))


def _pick_model(paths: list[Path]) -> Path:
    for path in paths:
        if path.name == DEFAULT_MODEL_NAME:
            return path
    return paths[0]


@st.cache_resource(show_spinner="Cargando modelo…")
def load_model(path_str: str) -> keras.Model:
    return keras.models.load_model(path_str)


def _bytes_to_image(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


def image_to_batch(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def _request_hostname() -> str:
    try:
        host = st.context.headers.get("Host", "")
        if host:
            return host.split(":")[0].strip().lower()
    except Exception:
        pass
    try:
        url = getattr(st.context, "url", None)
        if url:
            return (urlparse(str(url)).hostname or "").lower()
    except Exception:
        pass
    return ""


def _is_loopback_host(hostname: str) -> bool:
    return hostname in {"localhost", "127.0.0.1", "::1", ""}


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 820px; }
        .severity-badge {
            display: inline-block;
            padding: 0.4rem 0.95rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1.05rem;
        }
        .quiz-stage { position: relative; margin: 0.35rem 0 1.1rem; }
        .quiz-back {
            position: absolute; left: 14px; right: 14px; top: 22px;
            height: 78%;
            background: #dceef3;
            border-radius: 28px;
            box-shadow: 0 6px 18px rgba(20, 50, 80, 0.08);
            z-index: 0;
        }
        .quiz-back.deep {
            top: 36px; left: 26px; right: 26px;
            background: #cfe6ed;
            z-index: 0;
        }
        .quiz-front {
            position: relative; z-index: 1;
            background: #f3f7fa;
            border-radius: 28px;
            padding: 1.25rem 1.15rem 0.35rem;
            box-shadow: 0 12px 28px rgba(20, 50, 80, 0.14);
        }
        .quiz-front p.quiz-q, .quiz-q {
            font-size: 1.18rem; font-weight: 650; color: #1a1a1a;
            line-height: 1.35; margin: 0 0 0.35rem;
        }
        .quiz-progress { color: #5a6a72; font-size: 0.88rem; margin-bottom: 0.15rem; }
        div[data-testid="stVerticalBlock"]:has(.quiz-front) div[data-testid="stButton"] > button {
            border-radius: 999px !important;
            background: #a8dce8 !important;
            color: #1e4a55 !important;
            border: none !important;
            font-weight: 650 !important;
            min-height: 3rem !important;
            box-shadow: none !important;
        }
        div[data-testid="stVerticalBlock"]:has(.quiz-front) div[data-testid="stButton"] > button:hover {
            background: #8fd0df !important;
            color: #14363e !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _severity_badge(label: str) -> str:
    fg, bg = SEVERITY_STYLE.get(label, ("#262730", "#F0F2F6"))
    return (
        f'<span class="severity-badge" style="color:{fg};background:{bg};'
        f'border:1px solid {fg}33;">{label}</span>'
    )


def _perception_payload(data: dict) -> dict:
    geom = data["geom"]
    prob = data["prob"]
    return {
        "model_file": data["model"],
        "image_source": data["image_source"],
        "image_name": data["image_name"],
        "ml_probability": round(float(prob), 6),
        "ml_certainty_factor": round(float(cf_from_ml(prob)), 4),
        "geometry": {
            "label": geom.label,
            "display_name": geom.display_name,
            "confidence": geom.confidence,
            "angle_deg": geom.angle_deg,
            "notes": geom.notes,
        },
        "patron_orientacion": data["patron"],
        "field_answers": dict(data.get("field_answers") or {}),
    }


def _diagnose_from_data(data: dict, patron: str | None = None) -> object:
    answers = dict(data.get("field_answers") or {})
    chosen = patron_with_field_answers(patron or data["patron"], answers)
    return diagnose(
        data["prob"],
        elemento=data["elemento"],
        ambiente=data["ambiente"],
        patron_orientacion=chosen,
        ancho_mm=None,
        calidad_ancho=0.0,
        field_answers=answers,
    )


def _save_inspection(visit_id: str, image: Image.Image, data: dict) -> dict:
    verdict = data["verdict"]
    record = add_record(
        visit_id,
        image,
        elemento=data["elemento"],
        ambiente=data["ambiente"],
        perception=_perception_payload(data),
        expert_verdict=verdict_to_dict(verdict),
        field_answers=dict(data.get("field_answers") or {}),
    )
    data["record_id"] = record["id"]
    st.session_state["insp"] = data
    st.session_state["last_record_id"] = record["id"]
    st.session_state["saved_ok"] = True
    return data


def _stash_uploaded_file() -> None:
    uploaded = st.session_state.get("photo_upload")
    if uploaded is None:
        return
    try:
        payload = uploaded.getvalue()
        Image.open(BytesIO(payload)).convert("RGB")
    except Exception:
        st.session_state["draft_error"] = (
            "No se pudo leer esa imagen. En el celular elija JPG o PNG "
            "(algunos teléfonos usan HEIC y aquí no abre)."
        )
        return
    st.session_state.pop("draft_error", None)
    st.session_state["draft_image"] = payload
    st.session_state["draft_name"] = getattr(uploaded, "name", None) or "foto.jpg"
    st.session_state["draft_source"] = "file_uploader"


def _stash_camera_file() -> None:
    camera_file = st.session_state.get("photo_camera")
    if camera_file is None:
        return
    try:
        payload = camera_file.getvalue()
        Image.open(BytesIO(payload)).convert("RGB")
    except Exception:
        st.session_state["draft_error"] = "No se pudo leer la foto de la cámara. Intente de nuevo."
        return
    st.session_state.pop("draft_error", None)
    st.session_state["draft_image"] = payload
    st.session_state["draft_name"] = getattr(camera_file, "name", None) or "camera.jpg"
    st.session_state["draft_source"] = "camera_input"


def _clear_draft_photo() -> None:
    for key in ("draft_image", "draft_name", "draft_source", "draft_error"):
        st.session_state.pop(key, None)
    st.session_state["_reset_photo"] = True


def _clear_quiz() -> None:
    for key in ("quiz_active", "quiz_index", "quiz_answers", "quiz_pending"):
        st.session_state.pop(key, None)


def _render_quiz_cards() -> str | None:
    pending = st.session_state["quiz_pending"]
    index = int(st.session_state.get("quiz_index", 0))
    total = len(FIELD_QUESTIONS)
    question = FIELD_QUESTIONS[index]
    remaining = total - index - 1

    raw = pending.get("image_bytes")
    if raw:
        st.image(_bytes_to_image(raw), use_container_width=True)
    st.caption("Responda una carta. Si no está seguro, use **No lo sé** — no inventamos el dato.")

    backs = ""
    if remaining >= 2:
        backs = '<div class="quiz-back deep"></div><div class="quiz-back"></div>'
    elif remaining == 1:
        backs = '<div class="quiz-back"></div>'
    qtext = html.escape(str(question["text"]))
    card = (
        f'<div class="quiz-stage">{backs}'
        f'<div class="quiz-front">'
        f'<div class="quiz-progress">{index + 1} de {total}</div>'
        f'<p class="quiz-q">{qtext}</p>'
        f"</div></div>"
    )
    st.markdown(card, unsafe_allow_html=True)

    chosen: str | None = None
    for option in question["options"]:
        if st.button(str(option), key=f"quiz_{index}_{option}", use_container_width=True):
            chosen = str(option)
    if st.button("Cancelar preguntas", key="quiz_cancel"):
        _clear_quiz()
        st.rerun()
    return chosen


def _visit_picker() -> None:
    visits = list_visits()
    st.subheader("Visita")
    if st.session_state.get("visit_id"):
        try:
            visit = load_visit(st.session_state["visit_id"])
        except FileNotFoundError:
            st.session_state.pop("visit_id", None)
            visit = None
        if visit:
            counts = summarize_visit(visit)
            st.success(
                f"**{visit['place']}** · {visit.get('created_at_local', '')} · "
                f"{counts['total']} foto(s)"
            )
            if st.button("Cerrar visita"):
                st.session_state.pop("visit_id", None)
                st.session_state.pop("insp", None)
                st.session_state.pop("last_image", None)
                st.session_state.pop("last_record_id", None)
                _clear_draft_photo()
                _clear_quiz()
                st.rerun()
            return

    place = st.text_input("Nombre del lugar (casa, edificio, zona)")
    if st.button("Empezar visita", type="primary", disabled=not place.strip()):
        visit = create_visit(place)
        st.session_state["visit_id"] = visit["id"]
        st.session_state.pop("insp", None)
        st.rerun()

    if visits:
        labels = {
            f"{v.get('created_at_local', '')} · {v.get('place', '')} · "
            f"{len(v.get('records', []))} foto(s)": v["id"]
            for v in visits
        }
        chosen = st.selectbox("O reabrir una visita", list(labels.keys()))
        if st.button("Abrir visita"):
            st.session_state["visit_id"] = labels[chosen]
            st.session_state.pop("insp", None)
            st.rerun()
    else:
        st.caption("Aún no hay visitas guardadas en este equipo.")


def _render_visit_log(visit_id: str) -> None:
    visit = load_visit(visit_id)
    records = visit.get("records", [])
    if not records:
        st.caption("Todavía no hay fotos en esta visita.")
        return
    counts = summarize_visit(visit)
    st.write(
        f"**Resumen:** {counts['total']} · "
        f"Crítica {counts['Crítica']} · Moderada {counts['Moderada']} · "
        f"Leve {counts['Leve']} · Sin fisura {counts['Sin fisura']}"
    )
    for rec in reversed(records):
        sev = str((rec.get("expert_verdict") or {}).get("severity", ""))
        cols = st.columns([1, 2.2])
        img_file = rec.get("image_file")
        with cols[0]:
            if img_file:
                path = photo_path(visit_id, str(img_file))
                if path.exists():
                    st.image(str(path), use_container_width=True)
        with cols[1]:
            st.markdown(_severity_badge(sev), unsafe_allow_html=True)
            st.write(
                f"{rec.get('captured_at_local', '')} · "
                f"{rec.get('elemento', '')} · {rec.get('ambiente', '')}"
            )


def main() -> None:
    st.set_page_config(page_title="CrackExpert AI", page_icon="🧱", layout="centered")
    _inject_css()
    st.title("CrackExpert AI")
    st.caption("Una visita, varias fotos. Si hay fisura, salen 9 cartas de campo.")

    models = _discover_keras_models()
    if not models:
        st.error("Falta el modelo en la carpeta `models/`. Entrene primero o copie un archivo `.keras`.")
        st.stop()
    model_path = _pick_model(models)

    _visit_picker()
    visit_id = st.session_state.get("visit_id")
    if not visit_id:
        st.info("Empiece o abra una visita para ir guardando las fotos en este equipo.")
        return

    host = _request_hostname()
    lan = host and not _is_loopback_host(host)

    quiz_active = bool(st.session_state.get("quiz_active"))
    if quiz_active:
        pending = st.session_state.get("quiz_pending") or {}
        if not pending.get("image_bytes"):
            _clear_quiz()
            st.rerun()
        chosen = _render_quiz_cards()
        if chosen is not None:
            index = int(st.session_state.get("quiz_index", 0))
            qid = str(FIELD_QUESTIONS[index]["id"])
            answers = dict(st.session_state.get("quiz_answers") or {})
            answers[qid] = chosen
            st.session_state["quiz_answers"] = answers
            if index + 1 >= len(FIELD_QUESTIONS):
                data = dict(pending)
                data["field_answers"] = answers
                data["patron"] = patron_with_field_answers(data["patron"], answers)
                data["verdict"] = _diagnose_from_data(data)
                saved_image = _bytes_to_image(data["image_bytes"])
                _save_inspection(visit_id, saved_image, data)
                _clear_draft_photo()
                _clear_quiz()
                st.rerun()
            st.session_state["quiz_index"] = index + 1
            st.rerun()
        st.divider()
        st.subheader("Fotos de esta visita")
        _render_visit_log(visit_id)
        return

    st.subheader("Foto")
    if "use_desktop_camera" not in st.session_state:
        st.session_state["use_desktop_camera"] = not lan
    if st.session_state.pop("_reset_photo", False):
        st.session_state["photo_upload"] = None
        if "photo_camera" in st.session_state:
            st.session_state["photo_camera"] = None

    if lan or not st.session_state["use_desktop_camera"]:
        st.info("En el celular: Examinar → Cámara o Galería. Si no aparece, pruebe JPG o PNG.")
        st.file_uploader(
            "Tomar foto o elegir imagen",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="photo_upload",
            on_change=_stash_uploaded_file,
        )
    else:
        st.camera_input("Tomar foto", key="photo_camera", on_change=_stash_camera_file)
        st.file_uploader(
            "O subir una imagen",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="photo_upload",
            on_change=_stash_uploaded_file,
        )

    if st.session_state.get("photo_upload") is not None:
        _stash_uploaded_file()
    elif st.session_state.get("photo_camera") is not None:
        _stash_camera_file()

    image: Image.Image | None = None
    image_source = str(st.session_state.get("draft_source") or "")
    image_name = str(st.session_state.get("draft_name") or "")
    if st.session_state.get("draft_image"):
        try:
            image = _bytes_to_image(st.session_state["draft_image"])
        except Exception:
            st.session_state["draft_error"] = "La foto guardada se dañó. Cárguela otra vez."
            st.session_state.pop("draft_image", None)
            image = None

    if st.session_state.get("draft_error"):
        st.error(st.session_state["draft_error"])

    if image is not None:
        st.image(image, use_container_width=True)
        if st.button("Quitar foto"):
            _clear_draft_photo()
            st.rerun()
    else:
        st.write("Foto de cerca, de frente al elemento. Espere a verla aquí antes de generar.")

    c1, c2 = st.columns(2)
    with c1:
        elemento = st.selectbox("¿Qué elemento es?", ELEMENT_OPTIONS)
    with c2:
        ambiente = st.selectbox("¿Dónde está esta foto?", EXPOSURE_OPTIONS)

    run = st.button("Generar dictamen", type="primary", disabled=image is None, use_container_width=True)

    if run and image is not None:
        model = load_model(str(model_path))
        prob = float(model.predict(image_to_batch(image), verbose=0).reshape(-1)[0])
        has_crack = prob >= 0.50
        geom = analyze_orientation(np.asarray(image.convert("RGB")), assume_crack=has_crack)
        patron = patron_from_geometry(elemento, geom.label).value
        data = {
            "prob": prob,
            "geom": geom,
            "patron": patron,
            "elemento": elemento,
            "ambiente": ambiente,
            "model": model_path.name,
            "image_source": image_source,
            "image_name": image_name,
            "image_bytes": bytes(st.session_state["draft_image"]),
            "field_answers": {},
        }
        if not has_crack:
            data["verdict"] = _diagnose_from_data(data)
            _save_inspection(visit_id, image, data)
            _clear_draft_photo()
            st.rerun()
        st.session_state["quiz_pending"] = data
        st.session_state["quiz_active"] = True
        st.session_state["quiz_index"] = 0
        st.session_state["quiz_answers"] = {}
        st.session_state.pop("insp", None)
        st.rerun()

    data = st.session_state.get("insp")
    if data:
        verdict = data["verdict"]
        geom = data["geom"]
        prob = data["prob"]
        sev = verdict.severity.value
        if st.session_state.get("saved_ok"):
            st.success("Quedó guardado en esta visita.")
            st.session_state["saved_ok"] = False

        st.divider()
        st.subheader("Última foto")
        st.markdown(_severity_badge(sev), unsafe_allow_html=True)
        if verdict.has_crack:
            st.write("**Hay una fisura** en la foto.")
            st.write(f"**Cómo se ve:** {geom.display_name}")
        else:
            st.write("**No se ve una fisura clara** en esta foto.")
        st.write(verdict.headline or MEANING.get(sev, verdict.mechanism))
        if verdict.has_crack:
            st.write("**Qué hacer**")
            for item in verdict.action_plan[:5]:
                st.markdown(f"- {item}")

        with st.expander("Corregir orientación (si la foto engaña)"):
            override = st.selectbox(
                "Patrón que usará el experto",
                PATRON_ORIENTACION_OPTIONS,
                index=PATRON_ORIENTACION_OPTIONS.index(data["patron"])
                if data["patron"] in PATRON_ORIENTACION_OPTIONS
                else 0,
            )
            if st.button("Recalcular y actualizar lo guardado"):
                data["patron"] = override
                verdict2 = _diagnose_from_data(data, patron=override)
                data["verdict"] = verdict2
                rec_id = data.get("record_id") or st.session_state.get("last_record_id")
                if rec_id:
                    update_record(
                        visit_id,
                        rec_id,
                        perception=_perception_payload(data),
                        expert_verdict=verdict_to_dict(verdict2),
                    )
                st.session_state["insp"] = data
                st.rerun()

        with st.expander("Detalle técnico"):
            st.write(f"Modelo: `{data['model']}`")
            st.write(f"Probabilidad de fisura: {prob:.3f}")
            st.write(f"CF combinado: {verdict.cf_combined:.3f}")
            if geom.angle_deg is not None:
                st.write(f"Ángulo estimado: {geom.angle_deg}° · confianza geometría {geom.confidence:.2f}")
            st.write(geom.notes)
            st.write(verdict.mechanism)
            st.write(verdict.normative_basis)
            report = {
                "schema": "crackexpert.ai/informe/v1",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "visit_id": visit_id,
                "perception": _perception_payload(data),
                "site_metadata": {
                    "elemento": data["elemento"],
                    "ambiente": data["ambiente"],
                    "ancho_mm": None,
                    "patron_orientacion": data["patron"],
                    "field_answers": dict(data.get("field_answers") or {}),
                },
                "expert_verdict": verdict_to_dict(verdict),
            }
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Descargar esta foto en JSON",
                data=json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name=f"crackexpert_foto_{stamp}.json",
                mime="application/json",
                use_container_width=True,
            )

    st.divider()
    st.subheader("Fotos de esta visita")
    _render_visit_log(visit_id)


if __name__ == "__main__":
    main()
