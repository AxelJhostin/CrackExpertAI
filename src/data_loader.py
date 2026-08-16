"""Descarga, submuestreo balanceado, partición estratificada y tf.data.

Dataset: arunrk7/surface-crack-detection (Positive / Negative).
Subconjunto: 4.000 imágenes (2.000 por clase).
Splits fijos (random_state=42): 70% train / 15% val / 15% test.
Aumento de datos: exclusivamente en train.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_STATE: int = 42
IMAGE_SIZE: tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
N_PER_CLASS: int = 2000
CLASS_NAMES: tuple[str, str] = ("Negative", "Positive")
KAGGLE_DATASET: str = "arunrk7/surface-crack-detection"

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

AUTOTUNE = tf.data.AUTOTUNE


def _list_images(folder: Path) -> list[Path]:
    """Lista imágenes de un directorio de forma determinista (orden lexicográfico)."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in folder.rglob("*") if p.suffix.lower() in extensions]
    return sorted(files)


def _find_class_dir(root: Path, class_name: str) -> Path:
    """Localiza la carpeta Positive/Negative dentro del cache de kagglehub."""
    direct = root / class_name
    if direct.is_dir() and _list_images(direct):
        return direct
    matches = [p for p in root.rglob(class_name) if p.is_dir()]
    for candidate in sorted(matches, key=lambda p: len(p.parts)):
        if _list_images(candidate):
            return candidate
    raise FileNotFoundError(
        f"No se encontró la carpeta '{class_name}' bajo {root}."
    )


def download_raw_dataset() -> Path:
    """Descarga el dataset con kagglehub y apunta data/raw al cache local."""
    import kagglehub

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    marker = RAW_DIR / "SOURCE.txt"
    marker.write_text(
        f"kagglehub:{KAGGLE_DATASET}\ncache={cache_path}\n",
        encoding="utf-8",
    )
    print(f"[data] Dataset descargado/cacheado en: {cache_path}")
    return cache_path


def _sample_balanced(positive: list[Path], negative: list[Path]) -> tuple[list[Path], list[int]]:
    """Selecciona exactamente 2.000 Positive y 2.000 Negative con semilla fija."""
    rng = np.random.default_rng(RANDOM_STATE)
    if len(positive) < N_PER_CLASS or len(negative) < N_PER_CLASS:
        raise ValueError(
            f"Se requieren al menos {N_PER_CLASS} imágenes por clase. "
            f"Positive={len(positive)}, Negative={len(negative)}."
        )
    pos_idx = rng.choice(len(positive), size=N_PER_CLASS, replace=False)
    neg_idx = rng.choice(len(negative), size=N_PER_CLASS, replace=False)
    pos_sel = [positive[i] for i in np.sort(pos_idx)]
    neg_sel = [negative[i] for i in np.sort(neg_idx)]
    paths = neg_sel + pos_sel
    labels = [0] * N_PER_CLASS + [1] * N_PER_CLASS
    return paths, labels


def _stratified_splits(
    paths: list[Path],
    labels: list[int],
) -> dict[str, tuple[list[Path], list[int]]]:
    """70% train (2.800) / 15% val (600) / 15% test (600), estratificado."""
    paths_arr = np.array(paths, dtype=object)
    labels_arr = np.array(labels)
    x_tv, x_test, y_tv, y_test = train_test_split(
        paths_arr,
        labels_arr,
        test_size=0.15,
        stratify=labels_arr,
        random_state=RANDOM_STATE,
    )
    # 15% del total equivale a 15/85 del resto tras retirar el test.
    x_train, x_val, y_train, y_val = train_test_split(
        x_tv,
        y_tv,
        test_size=0.15 / 0.85,
        stratify=y_tv,
        random_state=RANDOM_STATE,
    )
    splits = {
        "train": (list(x_train), y_train.tolist()),
        "val": (list(x_val), y_val.tolist()),
        "test": (list(x_test), y_test.tolist()),
    }
    for name, (split_paths, split_labels) in splits.items():
        n_pos = int(np.sum(np.array(split_labels) == 1))
        n_neg = int(np.sum(np.array(split_labels) == 0))
        print(f"[data] {name:>5}: {len(split_paths)} imgs (Positive={n_pos}, Negative={n_neg})")
    return splits


def _copy_splits(splits: dict[str, tuple[list[Path], list[int]]]) -> Path:
    """Copia el subconjunto a data/processed/{split}/{class}/ de forma idempotente."""
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    for split_name, (paths, labels) in splits.items():
        for path, label in zip(paths, labels, strict=True):
            class_name = CLASS_NAMES[int(label)]
            dest_dir = PROCESSED_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / path.name
            if dest.exists():
                dest = dest_dir / f"{path.stem}_{path.parent.name}{path.suffix}"
            shutil.copy2(path, dest)
    print(f"[data] Subconjunto copiado en: {PROCESSED_DIR}")
    return PROCESSED_DIR


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Lee una imagen, la convierte a RGB float32 [0, 255] y la redimensiona a 224x224."""
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def build_augmentation() -> tf.keras.Sequential:
    """Aumento exclusivo de train: flip, rotación, zoom y brillo."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1, value_range=(0.0, 255.0)),
        ],
        name="train_augmentation",
    )


def _paths_and_labels(split_dir: Path) -> tuple[list[str], list[int]]:
    paths: list[str] = []
    labels: list[int] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        for image_path in _list_images(class_dir):
            paths.append(str(image_path))
            labels.append(class_index)
    return paths, labels


def _dataset_from_split(split_name: str, augment: bool) -> tf.data.Dataset:
    """Construye un tf.data.Dataset para un split ya materializado en disco."""
    split_dir = PROCESSED_DIR / split_name
    file_paths, labels = _paths_and_labels(split_dir)
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE)
    if split_name == "train":
        ds = ds.shuffle(buffer_size=len(file_paths), seed=RANDOM_STATE, reshuffle_each_iteration=True)
    if augment:
        augmenter = build_augmentation()

        def _apply_aug(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            return augmenter(image, training=True), label

        ds = ds.map(_apply_aug, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def prepare_data(*, force_rebuild: bool = False) -> dict[str, tf.data.Dataset]:
    """Orquesta descarga, muestreo, partición y datasets listos para entrenar.

    Returns:
        Diccionario con claves train / val / test. Train incluye augmentation;
        val y test solo redimensionan a 224x224 RGB (rango [0, 255]).
    """
    expected_counts = {"train": 2800, "val": 600, "test": 600}
    already_built = PROCESSED_DIR.exists() and all(
        sum(1 for _ in (PROCESSED_DIR / split / class_name).glob("*") if _.is_file())
        == (expected_counts[split] // 2)
        for split in expected_counts
        for class_name in CLASS_NAMES
    )

    if force_rebuild or not already_built:
        cache_root = download_raw_dataset()
        positive = _list_images(_find_class_dir(cache_root, "Positive"))
        negative = _list_images(_find_class_dir(cache_root, "Negative"))
        print(f"[data] Disponibles: Positive={len(positive)}, Negative={len(negative)}")
        paths, labels = _sample_balanced(positive, negative)
        splits = _stratified_splits(paths, labels)
        _copy_splits(splits)
    else:
        print(f"[data] Reutilizando partición existente en {PROCESSED_DIR}")

    datasets = {
        "train": _dataset_from_split("train", augment=True),
        "val": _dataset_from_split("val", augment=False),
        "test": _dataset_from_split("test", augment=False),
    }
    return datasets


def iter_test_filepaths() -> Iterable[Path]:
    """Rutas de test en el mismo orden que el dataset (Negative luego Positive)."""
    for class_name in CLASS_NAMES:
        yield from _list_images(PROCESSED_DIR / "test" / class_name)
