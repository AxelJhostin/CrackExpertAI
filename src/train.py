"""Entrenamiento en dos fases, EarlyStopping, checkpoints y persistencia .keras."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src.data_loader import PROJECT_ROOT
from src.models import MODEL_SPECS, ModelSpec, build_model, unfreeze_top_layers

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
EPOCHS_PHASE1: int = 20
EPOCHS_PHASE2: int = 20
LR_PHASE1: float = 1e-3
LR_PHASE2: float = 1e-5


@dataclass
class TrainResult:
    """Resultado de una corrida de entrenamiento para un modelo."""

    spec: ModelSpec
    model: keras.Model
    history: dict[str, list[float]] = field(default_factory=dict)
    phase1_epochs: int = 0
    weights_path: Path | None = None


def _merge_history(
    first: keras.callbacks.History | None,
    second: keras.callbacks.History | None,
) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for hist in (first, second):
        if hist is None:
            continue
        for key, values in hist.history.items():
            merged.setdefault(key, []).extend([float(v) for v in values])
    return merged


def _callbacks(checkpoint_path: Path) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]


def _compile(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def train_one(
    spec: ModelSpec,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    epochs_phase1: int = EPOCHS_PHASE1,
    epochs_phase2: int = EPOCHS_PHASE2,
) -> TrainResult:
    """Entrena un modelo en dos fases y guarda `models/<nombre>.keras`."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS_DIR / f"{spec.name}.keras"
    print(f"\n{'=' * 72}")
    print(f"[train] {spec.display_name} ({spec.name})")
    print(f"{'=' * 72}")

    model = build_model(spec)

    print(f"[train] Fase 1 — cabeza densa / backbone congelado | LR={LR_PHASE1:g}")
    _compile(model, LR_PHASE1)
    phase1_snapshot = MODELS_DIR / f"{spec.name}_phase1.keras"
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        callbacks=_callbacks(phase1_snapshot),
        verbose=1,
    )
    model.save(phase1_snapshot)

    print(
        f"[train] Fase 2 — fine-tuning "
        f"(capas={spec.fine_tune_layers if spec.is_transfer else 'todas'}) | LR={LR_PHASE2:g}"
    )
    unfreeze_top_layers(model, spec.fine_tune_layers or 0)
    _compile(model, LR_PHASE2)
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase2,
        callbacks=_callbacks(weights_path),
        verbose=1,
    )

    best_p1 = min(history_p1.history.get("val_loss", [float("inf")]))
    best_p2 = min(history_p2.history.get("val_loss", [float("inf")]))
    if best_p1 < best_p2 and phase1_snapshot.exists():
        print("[train] Fase 1 superó a Fase 2 en val_loss; se restauran pesos de Fase 1.")
        model = keras.models.load_model(phase1_snapshot)

    model.save(weights_path)
    if phase1_snapshot.exists():
        phase1_snapshot.unlink()
    history = _merge_history(history_p1, history_p2)
    history_path = REPORTS_DIR / f"history_{spec.name}.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    result = TrainResult(
        spec=spec,
        model=model,
        history=history,
        phase1_epochs=len(history_p1.history.get("loss", [])),
        weights_path=weights_path,
    )
    print(f"[train] Guardado: {weights_path}")
    return result


def train_all(
    datasets: dict[str, tf.data.Dataset],
    *,
    specs: tuple[ModelSpec, ...] = MODEL_SPECS,
) -> list[TrainResult]:
    """Entrena las cuatro arquitecturas de forma secuencial."""
    results: list[TrainResult] = []
    for spec in specs:
        results.append(train_one(spec, datasets["train"], datasets["val"]))
    return results

