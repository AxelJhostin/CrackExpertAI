"""Evaluación comparativa de casos reales con los 4 modelos entrenados.

Uso:
    1. Copiar fotos (.jpeg, .jpg, .png, .webp, .bmp) a data/external_test/
    2. python test_external.py

Cada ejecución ANEXA una corrida con fecha a:
    reports/external_test_comparison.md
    reports/external_test_comparison.csv
No se borra el historial de pruebas anteriores.

Preprocesado alineado al entrenamiento: RGB 224×224, float32 en [0, 255].
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import IMAGE_SIZE, PROJECT_ROOT  # noqa: E402
from src.models import MODEL_SPECS  # noqa: E402

IMAGE_DIR: Path = PROJECT_ROOT / "data" / "external_test"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CSV_PATH: Path = REPORTS_DIR / "external_test_comparison.csv"
MD_PATH: Path = REPORTS_DIR / "external_test_comparison.md"
THRESHOLD: float = 0.50
EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            files.append(path)
    return sorted(files, key=lambda p: p.name.lower())


def load_image_batch(path: Path) -> np.ndarray:
    """Carga una imagen como batch (1, 224, 224, 3) en [0, 255], igual que el pipeline."""
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def diagnose(prob_crack: float) -> tuple[str, float]:
    if prob_crack >= THRESHOLD:
        return "FISURA", prob_crack * 100.0
    return "SANA", (1.0 - prob_crack) * 100.0


def load_trained_models() -> list[tuple[str, str, keras.Model]]:
    loaded: list[tuple[str, str, keras.Model]] = []
    missing: list[str] = []
    for spec in MODEL_SPECS:
        weights = MODELS_DIR / f"{spec.name}.keras"
        if not weights.exists():
            missing.append(spec.display_name)
            continue
        print(f"[load] {spec.display_name:<22} <- {weights.name}")
        model = keras.models.load_model(weights)
        loaded.append((spec.name, spec.display_name, model))
    if missing:
        print(f"[aviso] Sin pesos en disco: {', '.join(missing)}")
    if not loaded:
        raise FileNotFoundError(
            f"No hay modelos .keras en {MODELS_DIR}. Ejecute primero: python main.py"
        )
    return loaded


def cell(prob: float) -> str:
    tag, conf = diagnose(prob)
    return f"{tag} {conf:5.1f}% (P={prob:.3f})"


def print_table(
    rows: list[dict[str, object]],
    model_keys: list[str],
    display: dict[str, str],
) -> None:
    col_file = "Archivo"
    width_file = max(len(col_file), max(len(str(r["archivo"])) for r in rows))
    widths = {k: max(len(display[k]), 28) for k in model_keys}

    header = f"{col_file:<{width_file}} | " + " | ".join(
        f"{display[k]:<{widths[k]}}" for k in model_keys
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        line = f"{str(row['archivo']):<{width_file}} | "
        line += " | ".join(
            f"{cell(float(row[f'p_{k}'])):<{widths[k]}}" for k in model_keys
        )
        print(line)


def _csv_fieldnames(model_keys: list[str]) -> list[str]:
    fields = ["corrida", "archivo"]
    for name in model_keys:
        fields.extend([f"diagnostico_{name}", f"certeza_{name}", f"P_{name}"])
    fields.append("consenso")
    return fields


def _row_to_csv(
    row: dict[str, object],
    model_keys: list[str],
    stamp: str,
) -> dict[str, object]:
    out: dict[str, object] = {
        "corrida": stamp,
        "archivo": row["archivo"],
        "consenso": row["consenso"],
    }
    for name in model_keys:
        prob = float(row[f"p_{name}"])
        label, conf = diagnose(prob)
        out[f"diagnostico_{name}"] = label
        out[f"certeza_{name}"] = round(conf, 2)
        out[f"P_{name}"] = round(prob, 4)
    return out


def write_csv(
    rows: list[dict[str, object]],
    model_keys: list[str],
    stamp: str,
) -> None:
    """Reescribe el CSV conservando filas previas y agregando la corrida actual."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(model_keys)
    previous: list[dict[str, object]] = []
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        with CSV_PATH.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for old in reader:
                if not old.get("corrida"):
                    old["corrida"] = "(corrida previa)"
                previous.append(old)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for old in previous:
            writer.writerow(old)
        for row in rows:
            writer.writerow(_row_to_csv(row, model_keys, stamp))


def write_markdown(
    rows: list[dict[str, object]],
    model_keys: list[str],
    display: dict[str, str],
    stamp: str,
) -> None:
    """Anexa una sección `## Corrida <fecha>` al markdown; no borra corridas previas."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    headers = [display[k] for k in model_keys]
    section = [
        f"## Corrida {stamp}",
        "",
        f"- Fotos evaluadas: **{len(rows)}**",
        f"- Umbral de decisión: **{THRESHOLD:.2f}**",
        f"- `P` = probabilidad de fisura (Positive). Certeza = P si FISURA, 1−P si SANA.",
        "",
        "| Archivo | " + " | ".join(headers) + " | Consenso |",
        "| --- | " + " | ".join("---" for _ in headers) + " | --- |",
    ]
    for row in rows:
        cells = [str(row["archivo"])]
        for name in model_keys:
            cells.append(cell(float(row[f"p_{name}"])))
        cells.append(str(row["consenso"]))
        section.append("| " + " | ".join(cells) + " |")
    section += ["", "---", ""]
    block = "\n".join(section)

    header = (
        "# Comparación en casos reales (`data/external_test`)\n\n"
        "Bitácora acumulativa: cada `python test_external.py` agrega una corrida "
        "con fecha. El historial no se elimina.\n\n"
    )
    if not MD_PATH.exists() or MD_PATH.stat().st_size == 0:
        MD_PATH.write_text(header + block, encoding="utf-8")
        return

    previous = MD_PATH.read_text(encoding="utf-8")
    if not previous.endswith("\n"):
        previous += "\n"
    MD_PATH.write_text(previous + "\n" + block, encoding="utf-8")


def consensus(labels: list[str]) -> str:
    fisura = sum(1 for lab in labels if lab == "FISURA")
    n = len(labels)
    if fisura == n:
        return f"FISURA ({n}/{n})"
    if fisura == 0:
        return f"SANA ({n}/{n})"
    return f"DISCREPANCIA ({fisura}/{n} fisura)"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    images = list_images(IMAGE_DIR)

    print("=" * 72)
    print("  EVALUACION COMPARATIVA - casos reales (4 modelos)")
    print("=" * 72)
    print(f"Carpeta: {IMAGE_DIR}")

    if not images:
        print(f"\n[!] No hay imágenes en {IMAGE_DIR}/")
        print("    Coloque fotos .jpeg/.jpg/.png y vuelva a ejecutar: python test_external.py")
        return

    print(f"Fotos:   {len(images)}")
    models = load_trained_models()
    model_keys = [name for name, _, _ in models]
    display = {name: display_name for name, display_name, _ in models}

    rows: list[dict[str, object]] = []
    for path in images:
        batch = load_image_batch(path)
        row: dict[str, object] = {"archivo": path.name}
        labels: list[str] = []
        print(f"\n-> {path.name}")
        for name, display_name, model in models:
            prob = float(model.predict(batch, verbose=0).reshape(-1)[0])
            row[f"p_{name}"] = prob
            label, conf = diagnose(prob)
            labels.append(label)
            print(
                f"   {display_name:<22}  {label:<7}  certeza={conf:6.2f}%  P(fisura)={prob:.4f}"
            )
        row["consenso"] = consensus(labels)
        print(f"   Consenso: {row['consenso']}")
        rows.append(row)

    print("\n" + "=" * 72)
    print("  TABLA COMPARATIVA")
    print("=" * 72)
    print_table(rows, model_keys, display)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_csv(rows, model_keys, stamp)
    write_markdown(rows, model_keys, display, stamp)
    print(f"\nCorrida anexada: {stamp}")
    print(f"CSV: {CSV_PATH}")
    print(f"MD:  {MD_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
