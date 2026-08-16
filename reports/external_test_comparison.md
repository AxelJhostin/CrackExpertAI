# Comparación en casos reales (`data/external_test`)

Umbral de decisión: **0.50**. `P` = probabilidad de fisura (clase Positive). Certeza = P si el diagnóstico es FISURA, o 1−P si es SANA.

| Archivo | CNN personalizada | MobileNetV2 | ResNet50V2 | EfficientNet-B0 | Consenso |
| --- | --- | --- | --- | --- | --- |
| 1.jpeg | FISURA 100.0% (P=1.000) | FISURA  99.2% (P=0.992) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| 2.jpeg | FISURA  93.1% (P=0.931) | FISURA  62.6% (P=0.626) | SANA  93.0% (P=0.070) | FISURA  99.9% (P=0.999) | DISCREPANCIA (3/4 fisura) |
| 3.jpeg | FISURA  99.8% (P=0.998) | FISURA  98.8% (P=0.988) | FISURA  58.6% (P=0.586) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| 4.jpeg | SANA  95.5% (P=0.045) | SANA  92.0% (P=0.080) | FISURA  61.2% (P=0.612) | FISURA  99.9% (P=0.999) | DISCREPANCIA (2/4 fisura) |
| 5.jpeg | FISURA  81.8% (P=0.818) | SANA  94.2% (P=0.058) | SANA  88.5% (P=0.115) | SANA  91.2% (P=0.088) | DISCREPANCIA (1/4 fisura) |
| 6.jpeg | SANA  97.7% (P=0.023) | SANA  98.7% (P=0.013) | SANA  99.9% (P=0.001) | SANA  97.9% (P=0.021) | SANA (4/4) |
| 7.jpeg | FISURA  94.7% (P=0.947) | FISURA  96.8% (P=0.968) | SANA  98.7% (P=0.013) | FISURA  76.7% (P=0.767) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.43.25 AM.jpeg | FISURA  87.7% (P=0.877) | FISURA  89.8% (P=0.898) | SANA  97.9% (P=0.021) | FISURA  92.8% (P=0.928) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.43.28 AM.jpeg | FISURA  93.6% (P=0.936) | SANA  65.3% (P=0.347) | SANA  96.9% (P=0.031) | SANA  97.5% (P=0.025) | DISCREPANCIA (1/4 fisura) |

Tabla CSV: `reports/external_test_comparison.csv`

## Corrida 2026-08-16 08:05:38

- Fotos evaluadas: **9**
- Umbral de decisión: **0.50**
- `P` = probabilidad de fisura (Positive). Certeza = P si FISURA, 1−P si SANA.

| Archivo | CNN personalizada | MobileNetV2 | ResNet50V2 | EfficientNet-B0 | Consenso |
| --- | --- | --- | --- | --- | --- |
| 1.jpeg | FISURA 100.0% (P=1.000) | FISURA  99.2% (P=0.992) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| 2.jpeg | FISURA  93.1% (P=0.931) | FISURA  62.6% (P=0.626) | SANA  93.0% (P=0.070) | FISURA  99.9% (P=0.999) | DISCREPANCIA (3/4 fisura) |
| 3.jpeg | FISURA  99.8% (P=0.998) | FISURA  98.8% (P=0.988) | FISURA  58.6% (P=0.586) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| 4.jpeg | SANA  95.5% (P=0.045) | SANA  92.0% (P=0.080) | FISURA  61.2% (P=0.612) | FISURA  99.9% (P=0.999) | DISCREPANCIA (2/4 fisura) |
| 5.jpeg | FISURA  81.8% (P=0.818) | SANA  94.2% (P=0.058) | SANA  88.5% (P=0.115) | SANA  91.2% (P=0.088) | DISCREPANCIA (1/4 fisura) |
| 6.jpeg | SANA  97.7% (P=0.023) | SANA  98.7% (P=0.013) | SANA  99.9% (P=0.001) | SANA  97.9% (P=0.021) | SANA (4/4) |
| 7.jpeg | FISURA  94.7% (P=0.947) | FISURA  96.8% (P=0.968) | SANA  98.7% (P=0.013) | FISURA  76.7% (P=0.767) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.43.25 AM.jpeg | FISURA  87.7% (P=0.877) | FISURA  89.8% (P=0.898) | SANA  97.9% (P=0.021) | FISURA  92.8% (P=0.928) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.43.28 AM.jpeg | FISURA  93.6% (P=0.936) | SANA  65.3% (P=0.347) | SANA  96.9% (P=0.031) | SANA  97.5% (P=0.025) | DISCREPANCIA (1/4 fisura) |

---
