"""Evaluación comparativa de casos reales con los 4 modelos entrenados.

Uso:
    Opción A (sin etiqueta): fotos sueltas en data/external_test/
    Opción B (OOD etiquetado, recomendado):
        data/external_test/Positive/   # hay fisura
        data/external_test/Negative/   # superficie sana / negativos difíciles
    python test_external.py

Cada ejecución ANEXA una corrida con fecha. No borra el historial.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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


def collect_cases() -> list[tuple[Path, int | None]]:
    """Positive=1, Negative=0; archivos en la raíz quedan sin etiqueta."""
    cases: list[tuple[Path, int | None]] = []
    pos_dir = IMAGE_DIR / "Positive"
    neg_dir = IMAGE_DIR / "Negative"
    if pos_dir.is_dir():
        cases.extend((p, 1) for p in list_images(pos_dir))
    if neg_dir.is_dir():
        cases.extend((p, 0) for p in list_images(neg_dir))
    for path in list_images(IMAGE_DIR):
        cases.append((path, None))
    return cases


def load_image_batch(path: Path) -> np.ndarray:
    """Carga una imagen como batch (1, 224, 224, 3) en [0, 255], igual que el pipeline."""
    if path.stat().st_size == 0:
        raise ValueError(f"archivo vacío (0 bytes): {path.name}")
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
    fields = ["corrida", "archivo", "etiqueta_real"]
    for name in model_keys:
        fields.extend([f"diagnostico_{name}", f"certeza_{name}", f"P_{name}", f"acierto_{name}"])
    fields.append("consenso")
    return fields


def _row_to_csv(
    row: dict[str, object],
    model_keys: list[str],
    stamp: str,
) -> dict[str, object]:
    y_true = row.get("y_true")
    etiqueta = "Positive" if y_true == 1 else "Negative" if y_true == 0 else ""
    out: dict[str, object] = {
        "corrida": stamp,
        "archivo": row["archivo"],
        "etiqueta_real": etiqueta,
        "consenso": row["consenso"],
    }
    for name in model_keys:
        prob = float(row[f"p_{name}"])
        label, conf = diagnose(prob)
        pred = 1 if label == "FISURA" else 0
        out[f"diagnostico_{name}"] = label
        out[f"certeza_{name}"] = round(conf, 2)
        out[f"P_{name}"] = round(prob, 4)
        if y_true in (0, 1):
            out[f"acierto_{name}"] = int(pred == int(y_true))
        else:
            out[f"acierto_{name}"] = ""
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
    metrics_md: str = "",
) -> None:
    """Anexa una sección `## Corrida <fecha>` al markdown; no borra corridas previas."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    headers = [display[k] for k in model_keys]
    n_lab = sum(1 for r in rows if r.get("y_true") in (0, 1))
    section = [
        f"## Corrida {stamp}",
        "",
        f"- Fotos evaluadas: **{len(rows)}** (etiquetadas OOD: {n_lab})",
        f"- Umbral de decisión: **{THRESHOLD:.2f}**",
        f"- `P` = probabilidad de fisura (Positive). Certeza = P si FISURA, 1−P si SANA.",
        "",
        "| Archivo | Etiqueta | " + " | ".join(headers) + " | Consenso |",
        "| --- | --- | " + " | ".join("---" for _ in headers) + " | --- |",
    ]
    for row in rows:
        y_true = row.get("y_true")
        etiqueta = "Positive" if y_true == 1 else "Negative" if y_true == 0 else "—"
        cells = [str(row["archivo"]), str(etiqueta)]
        for name in model_keys:
            cells.append(cell(float(row[f"p_{name}"])))
        cells.append(str(row["consenso"]))
        section.append("| " + " | ".join(cells) + " |")
    section += ["", metrics_md, "---", ""]
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


def ood_metrics(rows: list[dict[str, object]], model_keys: list[str], display: dict[str, str]) -> str:
    labeled = [r for r in rows if r.get("y_true") in (0, 1)]
    if not labeled:
        return ""
    y_true = np.array([int(r["y_true"]) for r in labeled], dtype=int)
    lines = [
        "### Métricas OOD (solo fotos con etiqueta Positive/Negative)",
        "",
        f"n = {len(labeled)} (Positive={int(y_true.sum())}, Negative={int((1 - y_true).sum())})",
        "",
        "| Modelo | Accuracy | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    print("\n--- Metricas OOD (fotos etiquetadas) ---")
    for name in model_keys:
        y_pred = np.array(
            [1 if float(r[f"p_{name}"]) >= THRESHOLD else 0 for r in labeled],
            dtype=int,
        )
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(
            f"  {display[name]:<22}  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}"
        )
        lines.append(
            f"| {display[name]} | {acc:.3f} | {prec:.3f} | {rec:.3f} | {f1:.3f} |"
        )
    lines += ["", ""]
    return "\n".join(lines)


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
    (IMAGE_DIR / "Positive").mkdir(exist_ok=True)
    (IMAGE_DIR / "Negative").mkdir(exist_ok=True)
    cases = collect_cases()

    print("=" * 72)
    print("  EVALUACION COMPARATIVA - casos reales (4 modelos)")
    print("=" * 72)
    print(f"Carpeta: {IMAGE_DIR}")

    if not cases:
        print(f"\n[!] No hay imágenes en {IMAGE_DIR}/ ni en Positive/ Negative/")
        print("    Positive = fisura, Negative = sana. Luego: python test_external.py")
        return

    n_lab = sum(1 for _, y in cases if y is not None)
    print(f"Fotos:   {len(cases)} (etiquetadas: {n_lab})")
    models = load_trained_models()
    model_keys = [name for name, _, _ in models]
    display = {name: display_name for name, display_name, _ in models}

    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for path, y_true in cases:
        try:
            batch = load_image_batch(path)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            skipped.append(path.name)
            print(f"\n-> {path.name}")
            print(f"   [omitida] no se puede leer ({exc})")
            continue
        row: dict[str, object] = {"archivo": path.name, "y_true": y_true}
        labels: list[str] = []
        print(f"\n-> {path.name}" + (f"  [y={'Positive' if y_true == 1 else 'Negative'}]" if y_true in (0, 1) else ""))
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

    if not rows:
        print("\n[!] Ninguna foto se pudo leer. Revise data/external_test/.")
        return

    print("\n" + "=" * 72)
    print("  TABLA COMPARATIVA")
    print("=" * 72)
    print_table(rows, model_keys, display)
    metrics_md = ood_metrics(rows, model_keys, display)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_csv(rows, model_keys, stamp)
    write_markdown(rows, model_keys, display, stamp, metrics_md)
    print(f"\nCorrida anexada: {stamp}")
    if skipped:
        print(f"Omitidas ({len(skipped)}): {', '.join(skipped)}")
        print("Si un .jpeg pesa 0 bytes, vuelva a copiar la foto original.")
    print(f"CSV: {CSV_PATH}")
    print(f"MD:  {MD_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
