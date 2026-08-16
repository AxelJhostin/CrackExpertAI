"""Orquestador del pipeline experimental de CrackExpert AI.

Uso (ejecutar manualmente):
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite `python main.py` desde la raíz del repositorio.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import prepare_data  # noqa: E402
from src.evaluate import evaluate_all  # noqa: E402
from src.train import train_all  # noqa: E402


def main() -> None:
    print("=" * 72)
    print("CrackExpert AI — pipeline de entrenamiento y evaluación")
    print("=" * 72)

    print("\n[1/3] Descarga, submuestreo balanceado y partición estratificada")
    datasets = prepare_data(force_rebuild=False)

    print("\n[2/3] Entrenamiento de 4 arquitecturas (2 fases, EarlyStopping)")
    results = train_all(datasets)

    print("\n[3/3] Evaluación en test retenido, figuras y bitácora")
    csv_path = evaluate_all(results, datasets)

    print("\n" + "=" * 72)
    print("Pipeline finalizado.")
    print(f"Comparación: {csv_path}")
    print("Figuras: reports/figures/")
    print("Bitácora: reports/experiments_log.md")
    print("=" * 72)


if __name__ == "__main__":
    main()
