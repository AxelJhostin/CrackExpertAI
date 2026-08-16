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

from src.archive import archive_previous_run  # noqa: E402
from src.data_loader import prepare_data  # noqa: E402
from src.evaluate import evaluate_all  # noqa: E402
from src.train import train_all  # noqa: E402


def main() -> None:
    print("=" * 72)
    print("CrackExpert AI — pipeline de entrenamiento y evaluación")
    print("=" * 72)

    print("\n[0/4] Archivar figuras y métricas de la corrida anterior (si existen)")
    archived = archive_previous_run()

    print("\n[1/4] Descarga, submuestreo balanceado (8.000 imgs) y partición estratificada")
    datasets = prepare_data(force_rebuild=False)

    print("\n[2/4] Entrenamiento de 4 arquitecturas (2 fases, EarlyStopping)")
    results = train_all(datasets)

    print("\n[3/4] Evaluación en test retenido, figuras y bitácora")
    csv_path = evaluate_all(results, datasets)

    print("\n[4/4] Reevaluación de casos reales en data/external_test/")
    from test_external import main as run_external_test

    run_external_test()

    print("\n" + "=" * 72)
    print("Pipeline finalizado.")
    print(f"Comparación vigente: {csv_path}")
    print("Figuras vigentes: reports/figures/")
    if archived is not None:
        print(f"Histórico de la corrida anterior: {archived}")
    print("Bitácora acumulativa: reports/experiments_log.md")
    print("Casos reales (acumulativo): reports/external_test_comparison.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
