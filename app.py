"""Interfaz Streamlit: CNN de fisuras + sistema experto patológico."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import IMAGE_SIZE, PROJECT_ROOT  # noqa: E402
from src.expert_system import (  # noqa: E402
    ELEMENT_OPTIONS,
    EXPOSURE_OPTIONS,
    PATRON_ORIENTACION_OPTIONS,
    diagnose,
    wmax_mm,
    Exposure,
)
from src.models import MODEL_SPECS  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"


def _available_models() -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = []
    for spec in MODEL_SPECS:
        path = MODELS_DIR / f"{spec.name}.keras"
        if path.exists():
            found.append((spec.name, spec.display_name, path))
    return found


@st.cache_resource(show_spinner="Cargando modelo…")
def load_model(path_str: str) -> keras.Model:
    return keras.models.load_model(path_str)


def image_to_batch(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def main() -> None:
    st.set_page_config(
        page_title="CrackExpert AI",
        page_icon="🧱",
        layout="wide",
    )
    st.title("CrackExpert AI")
    st.caption(
        "PUCE Sede Manabí · Sistemas Expertos — detección visual (CNN) + "
        "razonamiento normativo ACI 224R / ACI 318 / NEC-SE-HM"
    )

    models = _available_models()
    if not models:
        st.error("No hay pesos `.keras` en `models/`. Ejecute primero `python main.py`.")
        st.stop()

    with st.sidebar:
        st.header("Modelo de percepción")
        labels = {display: path for _, display, path in models}
        chosen = st.selectbox("Arquitectura", list(labels.keys()))
        st.markdown("---")
        st.header("Variables de campo")
        elemento = st.selectbox("Tipo de elemento estructural", ELEMENT_OPTIONS)
        ambiente = st.selectbox("Ambiente de exposición", EXPOSURE_OPTIONS)
        patron = st.selectbox(
            "Patrón u orientación visual",
            PATRON_ORIENTACION_OPTIONS,
            help=(
                "Observación de la geometría de la fisura. "
                "Determina el mecanismo físico probable (cortante, flexión, torsión, corrosión, etc.)."
            ),
        )
        medir = st.checkbox("Tengo medición de ancho (mm)", value=True)
        ancho = st.number_input(
            "Ancho estimado (mm)",
            min_value=0.0,
            max_value=20.0,
            value=0.30,
            step=0.05,
            format="%.2f",
            disabled=not medir,
        )
        calidad = st.slider(
            "Calidad de la medición de ancho (Qw)",
            min_value=0.0,
            max_value=1.0,
            value=1.0 if medir else 0.0,
            step=0.05,
            help="1.0 = fisurómetro; 0.5 = estimación visual.",
        )
        rust = st.checkbox("Manchas de óxido")
        spall = st.checkbox("Desprendimiento (spalling)")
        through = st.checkbox("Fisura pasante (losa/muro)")

        try:
            limite = wmax_mm(Exposure(ambiente))
            st.info(f"w_max ACI 224R para este ambiente: **{limite:.2f} mm**")
        except Exception:
            pass

    uploaded = st.file_uploader(
        "Imagen del elemento (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    col_img, col_out = st.columns([1, 1.15])
    image: Image.Image | None = None
    with col_img:
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption=uploaded.name, use_container_width=True)
        else:
            st.info("Cargue una fotografía de la superficie de hormigón.")

    run = st.button("Diagnosticar", type="primary", disabled=image is None)

    if run and image is not None:
        model = load_model(str(labels[chosen]))
        batch = image_to_batch(image)
        prob = float(model.predict(batch, verbose=0).reshape(-1)[0])
        verdict = diagnose(
            prob,
            elemento=elemento,
            ambiente=ambiente,
            patron_orientacion=patron,
            ancho_mm=float(ancho) if medir else None,
            calidad_ancho=float(calidad) if medir else 0.0,
            through_crack=through,
            rust_stains=rust,
            spalling=spall,
        )

        with col_out:
            st.subheader("1. Percepción (CNN)")
            st.metric("P(fisura)", f"{prob:.4f}")
            if prob >= 0.50:
                st.success(f"La red **{chosen}** clasifica la imagen como **fisura** (umbral 0.50).")
            else:
                st.warning(f"La red **{chosen}** no detecta fisura (P < 0.50).")

            st.subheader("2. Diagnóstico patológico (sistema experto)")
            if verdict.severity.value == "Crítica":
                st.error(f"Nivel de severidad: **{verdict.severity.value}**")
            elif verdict.severity.value == "Moderada":
                st.warning(f"Nivel de severidad: **{verdict.severity.value}**")
            elif verdict.severity.value == "Leve":
                st.success(f"Nivel de severidad: **{verdict.severity.value}**")
            else:
                st.info(f"Nivel de severidad: **{verdict.severity.value}**")

            st.markdown(f"**Factor de certeza combinado (MYCIN):** `{verdict.cf_combined:.3f}`  (rango [−1, 1])")
            st.markdown(f"**Mecanismo físico probable:** {verdict.mechanism}")
            st.markdown(f"**Fundamento normativo:** {verdict.normative_basis}")
            st.markdown(f"**w_max de exposición:** {verdict.max_allowed_width_mm:.2f} mm")

            st.markdown("**Plan de acción técnico**")
            for item in verdict.action_plan:
                st.markdown(f"- {item}")

            with st.expander("Reglas disparadas (trazabilidad)"):
                for rule in verdict.fired_rules:
                    st.markdown(
                        f"- `{rule.code}` · CF={rule.cf_contrib:.3f} · "
                        f"{rule.severity.value} — {rule.statement}"
                    )
            st.caption(verdict.notes)


if __name__ == "__main__":
    main()
