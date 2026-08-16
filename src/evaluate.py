"""Evaluación formal: curvas, matrices, ROC, errores y bitácora experimental."""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from tensorflow import keras

from src.data_loader import CLASS_NAMES, PROJECT_ROOT, iter_test_filepaths
from src.models import ModelSpec
from src.train import TrainResult

FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"
COMPARISON_CSV: Path = PROJECT_ROOT / "reports" / "models_comparison.csv"
EXPERIMENTS_LOG: Path = PROJECT_ROOT / "reports" / "experiments_log.md"
CSV_COLUMNS: tuple[str, ...] = (
    "Modelo",
    "Parametros_Totales",
    "Tamano_MB",
    "Latencia_ms",
    "Val_Accuracy",
    "Test_Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC",
)


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def collect_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve y_true, y_prob y un lote de imágenes para inspección visual."""
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    images: list[np.ndarray] = []
    for batch_x, batch_y in dataset:
        probs = model.predict(batch_x, verbose=0).reshape(-1)
        y_true.append(batch_y.numpy().reshape(-1))
        y_prob.append(probs)
        images.append(batch_x.numpy())
    return (
        np.concatenate(y_true).astype(int),
        np.concatenate(y_prob).astype(float),
        np.concatenate(images, axis=0),
    )


def plot_learning_curves(
    result: TrainResult,
) -> Path:
    """Loss y accuracy vs épocas, con línea de corte entre fases."""
    _ensure_dirs()
    history = result.history
    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Val loss")
    axes[0].set_title(f"Pérdida — {result.spec.display_name}")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("binary_crossentropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["accuracy"], label="Train acc")
    axes[1].plot(epochs, history["val_accuracy"], label="Val acc")
    axes[1].set_title(f"Precisión — {result.spec.display_name}")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if result.phase1_epochs > 0:
        for ax in axes:
            ax.axvline(result.phase1_epochs, color="gray", linestyle="--", linewidth=1)
            ax.text(
                result.phase1_epochs,
                ax.get_ylim()[1] * 0.95,
                " Fase 2",
                color="gray",
                va="top",
            )

    fig.tight_layout()
    out = FIGURES_DIR / f"learning_curves_{result.spec.name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrix(
    spec: ModelSpec,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    _ensure_dirs()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([0, 1], labels=list(CLASS_NAMES))
    ax.set_yticks([0, 1], labels=list(CLASS_NAMES))
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Matriz de confusión — {spec.display_name}")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, int(value), ha="center", va="center", fontsize=14)
    fig.tight_layout()
    out = FIGURES_DIR / f"confusion_matrix_{spec.name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_roc_comparison(
    roc_data: list[tuple[str, np.ndarray, np.ndarray, float]],
) -> Path:
    """Superpone las 4 curvas ROC y reporta AUC."""
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, fpr, tpr, roc_auc in roc_data:
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Comparación ROC — conjunto de test (n=600)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "roc_curves_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_misclassified_examples(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    images: np.ndarray,
    filepaths: list[Path],
    max_fp: int = 4,
    max_fn: int = 4,
) -> Path:
    """Cuadrícula de falsos positivos y falsos negativos con probabilidad."""
    _ensure_dirs()
    y_pred = (y_prob >= 0.5).astype(int)
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    fp_idx = fp_idx[np.argsort(-y_prob[fp_idx])][:max_fp]
    fn_idx = fn_idx[np.argsort(y_prob[fn_idx])][:max_fn]

    selected: list[tuple[str, int]] = [("FP", i) for i in fp_idx] + [("FN", i) for i in fn_idx]
    if not selected:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.set_title("Sin errores de clasificación en el test retenido.")
        out = FIGURES_DIR / "misclassified_examples.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    cols = min(4, len(selected))
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for k, ax in enumerate(axes_flat):
        if k >= len(selected):
            ax.axis("off")
            continue
        kind, idx = selected[k]
        image = images[idx]
        vis = np.clip(image / 255.0, 0.0, 1.0)
        ax.imshow(vis)
        fname = filepaths[idx].name if idx < len(filepaths) else f"idx_{idx}"
        ax.set_title(f"{kind}  p={y_prob[idx]:.3f}\n{fname}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Errores en test: FP (sin fisura→fisura) y FN (fisura→sana)", fontsize=11)
    fig.tight_layout()
    out = FIGURES_DIR / "misclassified_examples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _latency_ms(model: keras.Model, dataset: tf.data.Dataset, warmup: int = 2) -> float:
    """Latencia media por imagen (ms) sobre el conjunto de test."""
    times: list[float] = []
    n_images = 0
    for i, (batch_x, _) in enumerate(dataset):
        start = time.perf_counter()
        _ = model.predict(batch_x, verbose=0)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            times.append(elapsed)
            n_images += int(batch_x.shape[0])
    if n_images == 0:
        return float("nan")
    return (sum(times) / n_images) * 1000.0


def _model_size_mb(path: Path | None) -> float:
    if path is None or not path.exists():
        return float("nan")
    return path.stat().st_size / (1024 * 1024)


def _best_val_accuracy(history: dict[str, list[float]]) -> float:
    values = history.get("val_accuracy") or history.get("val_binary_accuracy") or []
    return float(max(values)) if values else float("nan")


def evaluate_all(
    results: list[TrainResult],
    datasets: dict[str, tf.data.Dataset],
) -> Path:
    """Genera figuras, CSV de comparación y actualiza la bitácora."""
    _ensure_dirs()
    test_ds = datasets["test"]
    val_ds = datasets["val"]
    filepaths = list(iter_test_filepaths())

    rows: list[dict[str, object]] = []
    roc_data: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    best_for_errors: tuple[TrainResult, np.ndarray, np.ndarray, np.ndarray] | None = None
    best_f1 = -1.0

    for result in results:
        print(f"[eval] Evaluando {result.spec.display_name}...")
        plot_learning_curves(result)

        y_true, y_prob, images = collect_predictions(result.model, test_ds)
        y_pred = (y_prob >= 0.5).astype(int)
        plot_confusion_matrix(result.spec, y_true, y_pred)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = float(auc(fpr, tpr))
        roc_data.append((result.spec.display_name, fpr, tpr, roc_auc))

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        test_acc = float(accuracy_score(y_true, y_pred))
        val_metrics = result.model.evaluate(val_ds, verbose=0, return_dict=True)
        val_acc = float(val_metrics.get("accuracy", _best_val_accuracy(result.history)))

        row = {
            "Modelo": result.spec.display_name,
            "Parametros_Totales": int(result.model.count_params()),
            "Tamano_MB": round(_model_size_mb(result.weights_path), 3),
            "Latencia_ms": round(_latency_ms(result.model, test_ds), 3),
            "Val_Accuracy": round(val_acc, 4),
            "Test_Accuracy": round(test_acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(roc_auc, 4),
        }
        rows.append(row)
        print(
            f"[eval] {result.spec.display_name}: "
            f"test_acc={test_acc:.4f} F1={f1:.4f} AUC={roc_auc:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_for_errors = (result, y_true, y_prob, images)

    plot_roc_comparison(roc_data)
    if best_for_errors is not None:
        _, y_true, y_prob, images = best_for_errors
        plot_misclassified_examples(y_true, y_prob, images, filepaths)

    _write_csv(rows)
    _write_experiments_log(results, rows)
    print(f"[eval] Tabla: {COMPARISON_CSV}")
    print(f"[eval] Figuras: {FIGURES_DIR}")
    return COMPARISON_CSV


def _write_csv(rows: list[dict[str, object]]) -> None:
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _select_optimal(rows: list[dict[str, object]]) -> dict[str, object]:
    """Criterio: maximizar F1 en test; desempate por ROC-AUC y luego menor latencia."""
    return max(
        rows,
        key=lambda r: (float(r["F1_Score"]), float(r["ROC_AUC"]), -float(r["Latencia_ms"])),
    )


def _write_experiments_log(results: list[TrainResult], rows: list[dict[str, object]]) -> None:
    best = _select_optimal(rows) if rows else None
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"## Corrida {stamp}",
        "",
        "### Configuración",
        "",
        "- Dataset: `arunrk7/surface-crack-detection`",
        "- Subconjunto: 4.000 imágenes (2.000 Positive / 2.000 Negative)",
        "- Splits estratificados (`random_state=42`): train 2.800 / val 600 / test 600",
        "- Input: 224×224 RGB, rango [0, 255]",
        "- Augmentation (solo train): RandomFlip, RandomRotation(0.1), RandomZoom(0.1), RandomBrightness(0.1)",
        "- Optimizador: Adam | Loss: binary_crossentropy",
        "- Fase 1: LR=1e-3, backbone congelado (CNN: entrenamiento completo)",
        "- Fase 2: LR=1e-5, fine-tuning (MobileNetV2: 25 capas; ResNet50V2/EfficientNet-B0: 20 capas)",
        "- EarlyStopping: monitor=val_loss, patience=5, restore_best_weights=True",
        "",
        "### Métricas por modelo (test independiente n=600)",
        "",
        "| " + " | ".join(CSV_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in CSV_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in CSV_COLUMNS) + " |")
    lines += ["", "### Selección del modelo óptimo", ""]
    if best is None:
        lines.append("Sin métricas disponibles.")
    else:
        lines.append(
            f"Se selecciona **{best['Modelo']}** por mayor F1-Score en test "
            f"({best['F1_Score']}), con ROC-AUC={best['ROC_AUC']} y "
            f"latencia={best['Latencia_ms']} ms/imagen. El criterio prioriza el "
            f"equilibrio precisión/exhaustividad sobre el conjunto retenido, "
            f"que no participó en el ajuste de hiperparámetros."
        )
    lines += ["", "---", ""]

    EXPERIMENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    previous = EXPERIMENTS_LOG.read_text(encoding="utf-8") if EXPERIMENTS_LOG.exists() else (
        "# Bitácora experimental — CrackExpert AI\n\n"
        "Registro cronológico de corridas de entrenamiento y evaluación.\n\n"
    )
    EXPERIMENTS_LOG.write_text(previous + "\n".join(lines), encoding="utf-8")
