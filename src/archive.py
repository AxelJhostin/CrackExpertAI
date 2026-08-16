"""Archiva figuras y métricas de la corrida vigente antes de sobrescribirlas."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from src.data_loader import PROJECT_ROOT

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
ARCHIVE_DIR: Path = REPORTS_DIR / "archive"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Artefactos que main.py / evaluate.py pisan en cada corrida.
SNAPSHOT_FILES: tuple[str, ...] = (
    "models_comparison.csv",
    "metrics.json",
)
SNAPSHOT_GLOBS: tuple[str, ...] = (
    "history_*.json",
)


def _has_anything_to_archive() -> bool:
    if FIGURES_DIR.is_dir() and any(FIGURES_DIR.glob("*.png")):
        return True
    for name in SNAPSHOT_FILES:
        if (REPORTS_DIR / name).is_file() and (REPORTS_DIR / name).stat().st_size > 0:
            return True
    for pattern in SNAPSHOT_GLOBS:
        if any(REPORTS_DIR.glob(pattern)):
            return True
    return False


def archive_previous_run() -> Path | None:
    """Copia la corrida actual a reports/archive/<fecha>/ y deja el frente libre.

    No mueve `experiments_log.md`, `TRAINING_RESULTS.md` ni
    `external_test_comparison.*` (esos ya son acumulativos).
    Los pesos `.keras` no se copian (son pesados); sí se archivan curvas y CSV.
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if not _has_anything_to_archive():
        print("[archive] No hay corrida previa que guardar.")
        return None

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest = ARCHIVE_DIR / stamp
    dest.mkdir(parents=True, exist_ok=False)
    copied: list[str] = []

    if FIGURES_DIR.is_dir():
        fig_dest = dest / "figures"
        fig_dest.mkdir(parents=True, exist_ok=True)
        for png in sorted(FIGURES_DIR.glob("*.png")):
            shutil.copy2(png, fig_dest / png.name)
            copied.append(f"figures/{png.name}")

    for name in SNAPSHOT_FILES:
        src = REPORTS_DIR / name
        if src.is_file() and src.stat().st_size > 0:
            shutil.copy2(src, dest / name)
            copied.append(name)

    for pattern in SNAPSHOT_GLOBS:
        for src in sorted(REPORTS_DIR.glob(pattern)):
            shutil.copy2(src, dest / src.name)
            copied.append(src.name)

    (dest / "README.txt").write_text(
        "Snapshot de la corrida previa, tomado automáticamente antes de "
        f"reescribir reports/ el {stamp.replace('_', ' ')}.\n\n"
        "Archivos:\n" + "\n".join(f"- {item}" for item in copied) + "\n",
        encoding="utf-8",
    )
    print(f"[archive] Corrida previa guardada en: {dest}")
    print(f"[archive] {len(copied)} archivo(s) copiados (figuras, CSV, historiales).")
    return dest
