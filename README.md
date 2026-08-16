# CrackExpert AI

Sistema híbrido de inteligencia artificial para la detección visual y evaluación patológica de fisuras en elementos de hormigón armado.

## Estructura

```
crackexpert-ai/
├── data/
│   ├── raw/                 # Dataset base (Positive / Negative)
│   └── processed/           # 70% train / 15% val / 15% test
├── models/                  # Modelo guardado (.keras o .pt y metadatos)
├── reports/
│   ├── figures/             # Curvas de pérdida, precisión, matriz de confusión, ROC
│   └── metrics.json         # Métricas numéricas formales para el informe
├── src/
│   ├── data_loader.py       # Descarga, partición estratificada y preprocesamiento
│   ├── train.py             # Entrenamiento, curvas y evaluación
│   ├── expert_system.py     # Motor de reglas de patología estructural (ACI/NEC)
│   └── predict.py           # Pipeline unificado (ML + Sistema Experto)
├── app.py                   # App web interactiva (Streamlit)
└── requirements.txt
```
