"""CrackExpert AI — visita de obra: fotos + 2 datos + bitácora local."""

from __future__ import annotations

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
    PATRON_ORIENTACION_OPTIONS,
    cf_from_ml,
    diagnose,
    patron_from_geometry,
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
    }


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
    st.caption("Una visita, varias fotos. Cada foto tiene su ambiente.")

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

    st.subheader("Foto")
    if lan:
        st.info("En el celular: Examinar → Cámara o Tomar foto.")
        uploaded = st.file_uploader(
            "Tomar foto o subir imagen",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )
        camera_file = None
    else:
        camera_file = st.camera_input("Tomar foto")
        uploaded = st.file_uploader("O subir una imagen", type=["jpg", "jpeg", "png", "webp", "bmp"])

    image: Image.Image | None = None
    image_source = ""
    image_name = ""
    if camera_file is not None:
        image = _bytes_to_image(camera_file.getvalue())
        image_source = "camera_input"
        image_name = camera_file.name or "camera.jpg"
    elif uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        image_source = "file_uploader"
        image_name = uploaded.name

    if image is not None:
        st.image(image, use_container_width=True)
    else:
        st.write("Foto de cerca, de frente al elemento.")

    c1, c2 = st.columns(2)
    with c1:
        elemento = st.selectbox("¿Qué elemento es?", ELEMENT_OPTIONS)
    with c2:
        ambiente = st.selectbox("¿Dónde está esta foto?", EXPOSURE_OPTIONS)

    run = st.button("Guardar en la visita", type="primary", disabled=image is None, use_container_width=True)

    if run and image is not None:
        model = load_model(str(model_path))
        prob = float(model.predict(image_to_batch(image), verbose=0).reshape(-1)[0])
        has_crack = prob >= 0.50
        geom = analyze_orientation(np.asarray(image.convert("RGB")), assume_crack=has_crack)
        patron = patron_from_geometry(elemento, geom.label).value
        verdict = diagnose(
            prob,
            elemento=elemento,
            ambiente=ambiente,
            patron_orientacion=patron,
            ancho_mm=None,
            calidad_ancho=0.0,
        )
        data = {
            "prob": prob,
            "geom": geom,
            "patron": patron,
            "verdict": verdict,
            "elemento": elemento,
            "ambiente": ambiente,
            "model": model_path.name,
            "image_source": image_source,
            "image_name": image_name,
        }
        record = add_record(
            visit_id,
            image,
            elemento=elemento,
            ambiente=ambiente,
            perception=_perception_payload(data),
            expert_verdict=verdict_to_dict(verdict),
        )
        data["record_id"] = record["id"]
        st.session_state["insp"] = data
        st.session_state["last_record_id"] = record["id"]
        st.session_state["saved_ok"] = True

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
        st.write(MEANING.get(sev, verdict.mechanism))
        if verdict.has_crack:
            st.write("**Qué hacer**")
            for item in verdict.action_plan[:3]:
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
                verdict2 = diagnose(
                    prob,
                    elemento=data["elemento"],
                    ambiente=data["ambiente"],
                    patron_orientacion=override,
                    ancho_mm=None,
                    calidad_ancho=0.0,
                )
                data["patron"] = override
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
