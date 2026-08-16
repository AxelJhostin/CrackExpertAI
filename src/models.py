"""Definición de las cuatro arquitecturas experimentales.

Las imágenes de entrada se esperan en RGB 224x224 con rango [0, 255].
Cada modelo de transfer learning aplica su propio `preprocess_input`.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.data_loader import IMAGE_SIZE

INPUT_SHAPE: tuple[int, int, int] = (*IMAGE_SIZE, 3)


@dataclass(frozen=True)
class ModelSpec:
    """Metadatos de entrenamiento asociados a una arquitectura."""

    name: str
    display_name: str
    fine_tune_layers: int | None
    is_transfer: bool


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("cnn_custom", "CNN personalizada", fine_tune_layers=None, is_transfer=False),
    ModelSpec("mobilenet_v2", "MobileNetV2", fine_tune_layers=25, is_transfer=True),
    ModelSpec("resnet50_v2", "ResNet50V2", fine_tune_layers=20, is_transfer=True),
    ModelSpec("efficientnet_b0", "EfficientNet-B0", fine_tune_layers=20, is_transfer=True),
)


def _classification_head(x: tf.Tensor, dropout_rate: float) -> tf.Tensor:
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    return layers.Dense(1, activation="sigmoid", name="crack_prob")(x)


def build_cnn_custom() -> keras.Model:
    """Baseline: CNN desde cero con 3 bloques Conv2D + BN + MaxPool + Dropout."""
    inputs = keras.Input(shape=INPUT_SHAPE, name="image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale_01")(inputs)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv_{filters}")(x)
        x = layers.BatchNormalization(name=f"bn_{filters}")(x)
        x = layers.Activation("relu", name=f"relu_{filters}")(x)
        x = layers.MaxPooling2D(2, name=f"pool_{filters}")(x)
        x = layers.Dropout(0.25, name=f"drop_{filters}")(x)

    x = layers.Dense(128, activation="relu", name="dense_128")(
        layers.GlobalAveragePooling2D(name="gap")(x)
    )
    x = layers.Dropout(0.50, name="dense_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="crack_prob")(x)
    return keras.Model(inputs, outputs, name="cnn_custom")


def _imagenet_minus_one(name: str) -> layers.Layer:
    """Normalización ImageNet de [0, 255] a [-1, 1] (MobileNetV2 / ResNet50V2).

    Se usa Rescaling en lugar de Lambda para serializar el .keras sin problemas.
    """
    return layers.Rescaling(1.0 / 127.5, offset=-1.0, name=name)


def _build_transfer_model(
    name: str,
    base_model: keras.Model,
    preprocess_layer: layers.Layer | None,
) -> keras.Model:
    """Ensambla backbone ImageNet congelable + cabeza binaria."""
    base_model.trainable = False
    inputs = keras.Input(shape=INPUT_SHAPE, name="image")
    x = preprocess_layer(inputs) if preprocess_layer is not None else inputs
    x = base_model(x, training=False)
    outputs = _classification_head(x, dropout_rate=0.30)
    return keras.Model(inputs, outputs, name=name)


def build_mobilenet_v2() -> keras.Model:
    """MobileNetV2 preentrenado en ImageNet (Fase 1: backbone congelado)."""
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
    )
    return _build_transfer_model(
        "mobilenet_v2",
        base,
        _imagenet_minus_one("preprocess_mobilenet_v2"),
    )


def build_resnet50_v2() -> keras.Model:
    """ResNet50V2 preentrenado en ImageNet (Fase 1: backbone congelado)."""
    base = keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
    )
    return _build_transfer_model(
        "resnet50_v2",
        base,
        _imagenet_minus_one("preprocess_resnet50_v2"),
    )


def build_efficientnet_b0() -> keras.Model:
    """EfficientNet-B0 preentrenado en ImageNet (Fase 1: backbone congelado)."""
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
    )
    # EfficientNet.preprocess_input es identidad sobre [0, 255].
    return _build_transfer_model("efficientnet_b0", base, None)


def get_backbone(model: keras.Model) -> keras.Model | None:
    """Devuelve el backbone de transfer learning, si existe."""
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name not in {model.name}:
            return layer
    return None


def unfreeze_top_layers(model: keras.Model, n_layers: int) -> None:
    """Fine-tuning: descongela las últimas n capas y mantiene BatchNorm en inferencia.

    Best practice: no entrenar estadísticas de BN con batches pequeños del dominio
    destino; se dejan no entrenables para estabilidad.
    """
    backbone = get_backbone(model)
    if backbone is None:
        # CNN desde cero: se refina la red completa a LR bajo.
        model.trainable = True
        return

    backbone.trainable = True
    freeze_until = max(len(backbone.layers) - n_layers, 0)
    for i, layer in enumerate(backbone.layers):
        if i < freeze_until:
            layer.trainable = False
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def build_model(spec: ModelSpec) -> keras.Model:
    """Fábrica de arquitecturas a partir de ModelSpec."""
    builders = {
        "cnn_custom": build_cnn_custom,
        "mobilenet_v2": build_mobilenet_v2,
        "resnet50_v2": build_resnet50_v2,
        "efficientnet_b0": build_efficientnet_b0,
    }
    return builders[spec.name]()
