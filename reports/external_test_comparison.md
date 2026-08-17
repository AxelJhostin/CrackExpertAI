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

## Corrida 2026-08-16 17:49:21

- Fotos evaluadas: **24**
- Umbral de decisión: **0.50**
- `P` = probabilidad de fisura (Positive). Certeza = P si FISURA, 1−P si SANA.

| Archivo | CNN personalizada | MobileNetV2 | ResNet50V2 | EfficientNet-B0 | Consenso |
| --- | --- | --- | --- | --- | --- |
| 1.jpeg | FISURA 100.0% (P=1.000) | FISURA  99.2% (P=0.992) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| 10.jpeg | FISURA  97.7% (P=0.977) | SANA  92.2% (P=0.078) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | DISCREPANCIA (3/4 fisura) |
| 11.jpeg | SANA  94.5% (P=0.055) | SANA  95.0% (P=0.050) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | DISCREPANCIA (2/4 fisura) |
| 12.jpeg | SANA  53.9% (P=0.461) | SANA  87.4% (P=0.126) | SANA  95.8% (P=0.042) | SANA  74.4% (P=0.256) | SANA (4/4) |
| 15.jpeg | SANA  92.9% (P=0.071) | SANA  99.7% (P=0.003) | SANA 100.0% (P=0.000) | FISURA  99.9% (P=0.999) | DISCREPANCIA (1/4 fisura) |
| 2.jpeg | FISURA  96.3% (P=0.963) | FISURA  62.6% (P=0.626) | SANA  93.0% (P=0.070) | FISURA  99.9% (P=0.999) | DISCREPANCIA (3/4 fisura) |
| 3.jpeg | FISURA  99.9% (P=0.999) | FISURA  98.8% (P=0.988) | FISURA  58.6% (P=0.586) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| 4.jpeg | SANA  70.9% (P=0.291) | SANA  92.0% (P=0.080) | FISURA  61.2% (P=0.612) | FISURA  99.9% (P=0.999) | DISCREPANCIA (2/4 fisura) |
| 5.jpeg | FISURA  78.0% (P=0.780) | SANA  94.2% (P=0.058) | SANA  88.5% (P=0.115) | SANA  91.2% (P=0.088) | DISCREPANCIA (1/4 fisura) |
| 6.jpeg | SANA  99.8% (P=0.002) | SANA  98.7% (P=0.013) | SANA  99.9% (P=0.001) | SANA  97.9% (P=0.021) | SANA (4/4) |
| 7.jpeg | FISURA  96.6% (P=0.966) | FISURA  96.8% (P=0.968) | SANA  98.7% (P=0.013) | FISURA  76.7% (P=0.767) | DISCREPANCIA (3/4 fisura) |
| 8.jpeg | FISURA  99.2% (P=0.992) | FISURA  96.3% (P=0.963) | FISURA  99.8% (P=0.998) | FISURA  99.4% (P=0.994) | FISURA (4/4) |
| buena1.jpeg | FISURA  99.9% (P=0.999) | SANA  77.3% (P=0.227) | SANA  99.9% (P=0.001) | SANA  67.6% (P=0.324) | DISCREPANCIA (1/4 fisura) |
| buena2.jpeg | SANA  93.8% (P=0.062) | SANA  99.5% (P=0.005) | SANA 100.0% (P=0.000) | SANA  91.3% (P=0.087) | SANA (4/4) |
| buena3.jpeg | SANA  61.9% (P=0.381) | SANA  99.6% (P=0.004) | SANA  99.9% (P=0.001) | SANA  99.9% (P=0.001) | SANA (4/4) |
| buena4.jpeg | SANA  99.6% (P=0.004) | SANA  99.8% (P=0.002) | SANA 100.0% (P=0.000) | SANA  99.9% (P=0.001) | SANA (4/4) |
| columna.jpeg | FISURA 100.0% (P=1.000) | FISURA  98.7% (P=0.987) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| columna2.jpeg | FISURA 100.0% (P=1.000) | FISURA  94.8% (P=0.948) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| fisuraColumnaPared.jpeg | FISURA  77.4% (P=0.774) | SANA  76.3% (P=0.237) | SANA  99.0% (P=0.010) | SANA  91.9% (P=0.081) | DISCREPANCIA (1/4 fisura) |
| fisuraColumnaPared2.jpeg | FISURA  99.9% (P=0.999) | FISURA  87.6% (P=0.876) | SANA  78.9% (P=0.211) | SANA  71.0% (P=0.290) | DISCREPANCIA (2/4 fisura) |
| fisuraColumnaPared3.jpeg | FISURA 100.0% (P=1.000) | FISURA  94.8% (P=0.948) | FISURA  80.1% (P=0.801) | FISURA  99.3% (P=0.993) | FISURA (4/4) |
| fisuraColumnaPared34.jpeg | FISURA  98.1% (P=0.981) | FISURA  92.5% (P=0.925) | FISURA  99.6% (P=0.996) | FISURA  98.1% (P=0.981) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.43.25 AM.jpeg | FISURA  70.9% (P=0.709) | FISURA  89.8% (P=0.898) | SANA  97.9% (P=0.021) | FISURA  92.8% (P=0.928) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.43.28 AM.jpeg | FISURA  89.1% (P=0.891) | SANA  65.3% (P=0.347) | SANA  96.9% (P=0.031) | SANA  97.5% (P=0.025) | DISCREPANCIA (1/4 fisura) |

---

## Corrida 2026-08-16 19:08:31

- Fotos evaluadas: **71** (etiquetadas OOD: 71)
- Umbral de decisión: **0.50**
- `P` = probabilidad de fisura (Positive). Certeza = P si FISURA, 1−P si SANA.

| Archivo | Etiqueta | CNN personalizada | MobileNetV2 | ResNet50V2 | EfficientNet-B0 | Consenso |
| --- | --- | --- | --- | --- | --- | --- |
| 6.jpeg | Positive | SANA  99.8% (P=0.002) | SANA  98.7% (P=0.013) | SANA  99.9% (P=0.001) | SANA  97.9% (P=0.021) | SANA (4/4) |
| buena1.jpeg | Positive | FISURA  99.9% (P=0.999) | SANA  77.3% (P=0.227) | SANA  99.9% (P=0.001) | SANA  67.6% (P=0.324) | DISCREPANCIA (1/4 fisura) |
| buena2.jpeg | Positive | SANA  93.8% (P=0.062) | SANA  99.5% (P=0.005) | SANA 100.0% (P=0.000) | SANA  91.3% (P=0.087) | SANA (4/4) |
| buena3.jpeg | Positive | SANA  61.9% (P=0.381) | SANA  99.6% (P=0.004) | SANA  99.9% (P=0.001) | SANA  99.9% (P=0.001) | SANA (4/4) |
| buena4.jpeg | Positive | SANA  99.6% (P=0.004) | SANA  99.8% (P=0.002) | SANA 100.0% (P=0.000) | SANA  99.9% (P=0.001) | SANA (4/4) |
| WhatsApp Image 2026-08-16 at 7.04.22 PM.jpeg | Positive | FISURA 100.0% (P=1.000) | FISURA  84.4% (P=0.844) | FISURA  85.7% (P=0.857) | FISURA  97.5% (P=0.975) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.04.23 PM (1).jpeg | Positive | FISURA  59.6% (P=0.596) | SANA  60.7% (P=0.393) | SANA 100.0% (P=0.000) | SANA  99.7% (P=0.003) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.23 PM.jpeg | Positive | FISURA 100.0% (P=1.000) | SANA  76.6% (P=0.234) | SANA  99.7% (P=0.003) | SANA  93.5% (P=0.065) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.24 PM (1).jpeg | Positive | FISURA  95.1% (P=0.951) | FISURA  84.9% (P=0.849) | SANA  50.4% (P=0.496) | SANA  68.9% (P=0.311) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.24 PM (2).jpeg | Positive | FISURA  97.2% (P=0.972) | SANA  71.0% (P=0.290) | FISURA  59.7% (P=0.597) | FISURA  65.6% (P=0.656) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.24 PM (3).jpeg | Positive | SANA  95.6% (P=0.044) | FISURA  59.4% (P=0.594) | SANA  99.8% (P=0.002) | SANA  60.5% (P=0.395) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.24 PM (4).jpeg | Positive | FISURA  56.4% (P=0.564) | FISURA  65.0% (P=0.650) | FISURA  58.9% (P=0.589) | SANA  95.8% (P=0.042) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.24 PM.jpeg | Positive | FISURA 100.0% (P=1.000) | FISURA  91.3% (P=0.913) | SANA  99.5% (P=0.005) | FISURA  75.3% (P=0.753) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.25 PM (1).jpeg | Positive | FISURA  62.9% (P=0.629) | SANA  92.8% (P=0.072) | SANA  94.1% (P=0.059) | SANA  99.3% (P=0.007) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.25 PM (2).jpeg | Positive | FISURA  99.5% (P=0.995) | SANA  75.8% (P=0.242) | SANA 100.0% (P=0.000) | SANA  99.5% (P=0.005) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.25 PM (3).jpeg | Positive | FISURA  98.8% (P=0.988) | SANA  61.5% (P=0.385) | SANA  97.7% (P=0.023) | SANA  93.6% (P=0.064) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.25 PM (4).jpeg | Positive | SANA  97.4% (P=0.026) | SANA  93.1% (P=0.069) | SANA 100.0% (P=0.000) | SANA  98.6% (P=0.014) | SANA (4/4) |
| WhatsApp Image 2026-08-16 at 7.04.25 PM.jpeg | Positive | FISURA  99.9% (P=0.999) | FISURA  85.8% (P=0.858) | SANA 100.0% (P=0.000) | SANA  97.6% (P=0.024) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.26 PM (1).jpeg | Positive | FISURA 100.0% (P=1.000) | FISURA  65.9% (P=0.659) | FISURA  90.0% (P=0.900) | FISURA  95.1% (P=0.951) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.04.26 PM.jpeg | Positive | FISURA  93.4% (P=0.934) | FISURA  79.5% (P=0.795) | SANA  99.4% (P=0.006) | SANA  62.4% (P=0.376) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.27 PM (1).jpeg | Positive | FISURA  52.4% (P=0.524) | FISURA  57.9% (P=0.579) | SANA  99.7% (P=0.003) | SANA  53.3% (P=0.467) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.27 PM (2).jpeg | Positive | FISURA 100.0% (P=1.000) | SANA  80.4% (P=0.196) | SANA  99.7% (P=0.003) | SANA  90.3% (P=0.097) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.27 PM.jpeg | Positive | FISURA  86.9% (P=0.869) | FISURA  72.5% (P=0.725) | SANA  99.0% (P=0.010) | SANA  98.9% (P=0.011) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.28 PM.jpeg | Positive | SANA  97.7% (P=0.023) | SANA  92.0% (P=0.080) | SANA 100.0% (P=0.000) | SANA  99.0% (P=0.010) | SANA (4/4) |
| 1.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  99.2% (P=0.992) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| 10.jpeg | Negative | FISURA  97.7% (P=0.977) | SANA  92.2% (P=0.078) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | DISCREPANCIA (3/4 fisura) |
| 11.jpeg | Negative | SANA  94.5% (P=0.055) | SANA  95.0% (P=0.050) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | DISCREPANCIA (2/4 fisura) |
| 12.jpeg | Negative | SANA  53.9% (P=0.461) | SANA  87.4% (P=0.126) | SANA  95.8% (P=0.042) | SANA  74.4% (P=0.256) | SANA (4/4) |
| 15.jpeg | Negative | SANA  92.9% (P=0.071) | SANA  99.7% (P=0.003) | SANA 100.0% (P=0.000) | FISURA  99.9% (P=0.999) | DISCREPANCIA (1/4 fisura) |
| 2.jpeg | Negative | FISURA  96.3% (P=0.963) | FISURA  62.6% (P=0.626) | SANA  93.0% (P=0.070) | FISURA  99.9% (P=0.999) | DISCREPANCIA (3/4 fisura) |
| 3.jpeg | Negative | FISURA  99.9% (P=0.999) | FISURA  98.8% (P=0.988) | FISURA  58.6% (P=0.586) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| 4.jpeg | Negative | SANA  70.9% (P=0.291) | SANA  92.0% (P=0.080) | FISURA  61.2% (P=0.612) | FISURA  99.9% (P=0.999) | DISCREPANCIA (2/4 fisura) |
| 5.jpeg | Negative | FISURA  78.0% (P=0.780) | SANA  94.2% (P=0.058) | SANA  88.5% (P=0.115) | SANA  91.2% (P=0.088) | DISCREPANCIA (1/4 fisura) |
| 7.jpeg | Negative | FISURA  96.6% (P=0.966) | FISURA  96.8% (P=0.968) | SANA  98.7% (P=0.013) | FISURA  76.7% (P=0.767) | DISCREPANCIA (3/4 fisura) |
| 8.jpeg | Negative | FISURA  99.2% (P=0.992) | FISURA  96.3% (P=0.963) | FISURA  99.8% (P=0.998) | FISURA  99.4% (P=0.994) | FISURA (4/4) |
| columna.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  98.7% (P=0.987) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| columna2.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  94.8% (P=0.948) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| fisuraColumnaPared.jpeg | Negative | FISURA  77.4% (P=0.774) | SANA  76.3% (P=0.237) | SANA  99.0% (P=0.010) | SANA  91.9% (P=0.081) | DISCREPANCIA (1/4 fisura) |
| fisuraColumnaPared2.jpeg | Negative | FISURA  99.9% (P=0.999) | FISURA  87.6% (P=0.876) | SANA  78.9% (P=0.211) | SANA  71.0% (P=0.290) | DISCREPANCIA (2/4 fisura) |
| fisuraColumnaPared3.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  94.8% (P=0.948) | FISURA  80.1% (P=0.801) | FISURA  99.3% (P=0.993) | FISURA (4/4) |
| fisuraColumnaPared34.jpeg | Negative | FISURA  98.1% (P=0.981) | FISURA  92.5% (P=0.925) | FISURA  99.6% (P=0.996) | FISURA  98.1% (P=0.981) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.13 PM (1).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  93.0% (P=0.930) | FISURA  99.9% (P=0.999) | FISURA  99.9% (P=0.999) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.13 PM.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  99.5% (P=0.995) | FISURA  99.8% (P=0.998) | FISURA  99.5% (P=0.995) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.14 PM (1).jpeg | Negative | FISURA  99.8% (P=0.998) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA  99.7% (P=0.997) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.14 PM.jpeg | Negative | FISURA  95.1% (P=0.951) | FISURA  81.3% (P=0.813) | FISURA  94.8% (P=0.948) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.15 PM (1).jpeg | Negative | FISURA  98.4% (P=0.984) | FISURA  99.6% (P=0.996) | FISURA  92.6% (P=0.926) | FISURA  99.9% (P=0.999) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.15 PM.jpeg | Negative | FISURA  99.9% (P=0.999) | FISURA  99.9% (P=0.999) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.16 PM.jpeg | Negative | FISURA  98.0% (P=0.980) | FISURA  73.9% (P=0.739) | SANA  98.9% (P=0.011) | SANA  75.4% (P=0.246) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.17 PM.jpeg | Negative | FISURA  91.1% (P=0.911) | FISURA  71.4% (P=0.714) | FISURA  63.1% (P=0.631) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.35 PM (1).jpeg | Negative | SANA  60.2% (P=0.398) | SANA  56.7% (P=0.433) | SANA  99.8% (P=0.002) | SANA  93.8% (P=0.062) | SANA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.35 PM (2).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  81.7% (P=0.817) | FISURA  76.5% (P=0.765) | SANA  63.9% (P=0.361) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.35 PM (3).jpeg | Negative | FISURA  99.7% (P=0.997) | FISURA  94.5% (P=0.945) | SANA  90.6% (P=0.094) | FISURA 100.0% (P=1.000) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.35 PM.jpeg | Negative | FISURA  98.0% (P=0.980) | FISURA  91.2% (P=0.912) | FISURA 100.0% (P=1.000) | FISURA  99.3% (P=0.993) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.36 PM (1).jpeg | Negative | FISURA 100.0% (P=1.000) | SANA  62.5% (P=0.375) | SANA  99.0% (P=0.010) | FISURA  64.8% (P=0.648) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.36 PM (2).jpeg | Negative | FISURA  99.8% (P=0.998) | FISURA  82.1% (P=0.821) | FISURA  94.9% (P=0.949) | FISURA  94.3% (P=0.943) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.36 PM (3).jpeg | Negative | FISURA  99.9% (P=0.999) | FISURA  74.7% (P=0.747) | SANA  99.4% (P=0.006) | FISURA  99.7% (P=0.997) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.36 PM.jpeg | Negative | FISURA  61.8% (P=0.618) | SANA  95.8% (P=0.042) | SANA  99.9% (P=0.001) | SANA  62.2% (P=0.378) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.37 PM (1).jpeg | Negative | FISURA  99.2% (P=0.992) | FISURA  70.5% (P=0.705) | FISURA  92.7% (P=0.927) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.37 PM (2).jpeg | Negative | FISURA  99.9% (P=0.999) | FISURA  93.7% (P=0.937) | SANA  99.7% (P=0.003) | FISURA  99.3% (P=0.993) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.37 PM (3).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.37 PM (4).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.37 PM.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  94.4% (P=0.944) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.38 PM (1).jpeg | Negative | FISURA  99.9% (P=0.999) | FISURA  98.4% (P=0.984) | FISURA  90.1% (P=0.901) | FISURA  83.0% (P=0.830) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.38 PM (2).jpeg | Negative | FISURA  99.8% (P=0.998) | FISURA  99.5% (P=0.995) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.38 PM (3).jpeg | Negative | SANA  98.8% (P=0.012) | SANA  99.1% (P=0.009) | SANA  99.6% (P=0.004) | FISURA 100.0% (P=1.000) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.38 PM.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  91.7% (P=0.917) | FISURA  99.8% (P=0.998) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.39 PM (1).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  94.8% (P=0.948) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.39 PM (2).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  98.7% (P=0.987) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.39 PM.jpeg | Negative | SANA  92.9% (P=0.071) | SANA  99.7% (P=0.003) | SANA 100.0% (P=0.000) | FISURA  99.9% (P=0.999) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.43.25 AM.jpeg | Negative | FISURA  70.9% (P=0.709) | FISURA  89.8% (P=0.898) | SANA  97.9% (P=0.021) | FISURA  92.8% (P=0.928) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.43.28 AM.jpeg | Negative | FISURA  89.1% (P=0.891) | SANA  65.3% (P=0.347) | SANA  96.9% (P=0.031) | SANA  97.5% (P=0.025) | DISCREPANCIA (1/4 fisura) |

### Métricas OOD (solo fotos con etiqueta Positive/Negative)

n = 71 (Positive=24, Negative=47)

| Modelo | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| CNN personalizada | 0.338 | 0.298 | 0.708 | 0.420 |
| MobileNetV2 | 0.324 | 0.227 | 0.417 | 0.294 |
| ResNet50V2 | 0.310 | 0.121 | 0.167 | 0.140 |
| EfficientNet-B0 | 0.183 | 0.095 | 0.167 | 0.121 |


---

## Corrida 2026-08-17 02:22:55

- Fotos evaluadas: **77** (etiquetadas OOD: 77)
- Umbral de decisión: **0.50**
- `P` = probabilidad de fisura (Positive). Certeza = P si FISURA, 1−P si SANA.

| Archivo | Etiqueta | CNN personalizada | MobileNetV2 | ResNet50V2 | EfficientNet-B0 | Consenso |
| --- | --- | --- | --- | --- | --- | --- |
| 6.jpeg | Positive | SANA  97.7% (P=0.023) | SANA  99.0% (P=0.010) | SANA  96.9% (P=0.031) | SANA  96.2% (P=0.038) | SANA (4/4) |
| buena1.jpeg | Positive | FISURA 100.0% (P=1.000) | SANA  92.4% (P=0.076) | SANA 100.0% (P=0.000) | SANA  55.7% (P=0.443) | DISCREPANCIA (1/4 fisura) |
| buena2.jpeg | Positive | SANA  97.5% (P=0.025) | SANA 100.0% (P=0.000) | SANA 100.0% (P=0.000) | SANA  95.7% (P=0.043) | SANA (4/4) |
| buena3.jpeg | Positive | SANA  60.9% (P=0.391) | SANA 100.0% (P=0.000) | SANA  96.1% (P=0.039) | SANA 100.0% (P=0.000) | SANA (4/4) |
| buena4.jpeg | Positive | SANA  98.8% (P=0.012) | SANA 100.0% (P=0.000) | SANA 100.0% (P=0.000) | SANA  99.9% (P=0.001) | SANA (4/4) |
| WhatsApp Image 2026-08-16 at 7.04.22 PM.jpeg | Positive | FISURA 100.0% (P=1.000) | FISURA  90.7% (P=0.907) | SANA  73.0% (P=0.270) | FISURA  94.5% (P=0.945) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.23 PM (1).jpeg | Positive | FISURA  77.3% (P=0.773) | SANA  99.6% (P=0.004) | SANA 100.0% (P=0.000) | SANA  99.7% (P=0.003) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.23 PM.jpeg | Positive | FISURA 100.0% (P=1.000) | SANA  90.8% (P=0.092) | SANA 100.0% (P=0.000) | SANA  92.0% (P=0.080) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.24 PM (1).jpeg | Positive | FISURA  97.0% (P=0.970) | SANA  94.7% (P=0.053) | SANA  91.9% (P=0.081) | SANA  77.1% (P=0.229) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.24 PM (2).jpeg | Positive | FISURA  97.2% (P=0.972) | SANA  92.4% (P=0.076) | FISURA  76.8% (P=0.768) | FISURA  67.4% (P=0.674) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.24 PM (3).jpeg | Positive | SANA  88.5% (P=0.115) | SANA  85.6% (P=0.144) | SANA 100.0% (P=0.000) | FISURA  73.5% (P=0.735) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.24 PM (4).jpeg | Positive | SANA  54.4% (P=0.456) | SANA  81.2% (P=0.188) | SANA  68.1% (P=0.319) | SANA  98.7% (P=0.013) | SANA (4/4) |
| WhatsApp Image 2026-08-16 at 7.04.24 PM.jpeg | Positive | FISURA 100.0% (P=1.000) | SANA  50.6% (P=0.494) | SANA  99.9% (P=0.001) | FISURA  90.1% (P=0.901) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.25 PM (1).jpeg | Positive | FISURA  89.7% (P=0.897) | SANA  99.9% (P=0.001) | SANA  99.3% (P=0.007) | SANA  99.6% (P=0.004) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.25 PM (2).jpeg | Positive | FISURA  99.5% (P=0.995) | SANA  99.0% (P=0.010) | SANA 100.0% (P=0.000) | SANA  99.8% (P=0.002) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.25 PM (3).jpeg | Positive | FISURA  99.1% (P=0.991) | SANA  97.2% (P=0.028) | SANA  99.9% (P=0.001) | SANA  96.2% (P=0.038) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.25 PM (4).jpeg | Positive | SANA  99.5% (P=0.005) | SANA  99.9% (P=0.001) | SANA 100.0% (P=0.000) | SANA  94.8% (P=0.052) | SANA (4/4) |
| WhatsApp Image 2026-08-16 at 7.04.25 PM.jpeg | Positive | FISURA  99.9% (P=0.999) | FISURA  63.8% (P=0.638) | SANA 100.0% (P=0.000) | SANA  95.5% (P=0.045) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.26 PM (1).jpeg | Positive | FISURA 100.0% (P=1.000) | FISURA  98.0% (P=0.980) | FISURA  87.5% (P=0.875) | FISURA  94.5% (P=0.945) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.04.26 PM.jpeg | Positive | FISURA  93.6% (P=0.936) | FISURA  53.7% (P=0.537) | SANA 100.0% (P=0.000) | FISURA  66.4% (P=0.664) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.27 PM (1).jpeg | Positive | FISURA  84.2% (P=0.842) | FISURA  62.0% (P=0.620) | SANA  99.9% (P=0.001) | SANA  72.1% (P=0.279) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.27 PM (2).jpeg | Positive | FISURA 100.0% (P=1.000) | SANA  96.6% (P=0.034) | SANA  99.9% (P=0.001) | SANA  93.8% (P=0.062) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.27 PM.jpeg | Positive | FISURA  88.4% (P=0.884) | SANA  93.8% (P=0.062) | SANA  99.1% (P=0.009) | SANA  98.9% (P=0.011) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.04.28 PM.jpeg | Positive | SANA  99.5% (P=0.005) | SANA  99.7% (P=0.003) | SANA 100.0% (P=0.000) | SANA  95.9% (P=0.041) | SANA (4/4) |
| WhatsApp Image 2026-08-16 at 7.11.37 PM.jpeg | Positive | FISURA 100.0% (P=1.000) | FISURA  51.9% (P=0.519) | SANA  98.1% (P=0.019) | SANA  52.7% (P=0.473) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.11.48 PM.jpeg | Positive | FISURA 100.0% (P=1.000) | FISURA  70.9% (P=0.709) | SANA  92.7% (P=0.073) | FISURA  92.9% (P=0.929) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.12.05 PM.jpeg | Positive | FISURA 100.0% (P=1.000) | SANA  92.2% (P=0.078) | SANA  99.2% (P=0.008) | FISURA  65.0% (P=0.650) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.12.17 PM.jpeg | Positive | FISURA 100.0% (P=1.000) | FISURA  97.6% (P=0.976) | SANA  86.7% (P=0.133) | SANA  50.6% (P=0.494) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.13.05 PM.jpeg | Positive | FISURA 100.0% (P=1.000) | FISURA  99.4% (P=0.994) | SANA  99.5% (P=0.005) | FISURA  73.8% (P=0.738) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.13.27 PM.jpeg | Positive | SANA  53.5% (P=0.465) | FISURA  99.8% (P=0.998) | SANA  99.9% (P=0.001) | FISURA  92.5% (P=0.925) | DISCREPANCIA (2/4 fisura) |
| 1.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA  99.9% (P=0.999) | FISURA  99.9% (P=0.999) | FISURA (4/4) |
| 10.jpeg | Negative | FISURA  98.6% (P=0.986) | SANA  85.5% (P=0.145) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | DISCREPANCIA (3/4 fisura) |
| 11.jpeg | Negative | SANA  93.6% (P=0.064) | SANA  96.7% (P=0.033) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | DISCREPANCIA (2/4 fisura) |
| 12.jpeg | Negative | SANA  69.6% (P=0.304) | SANA  77.1% (P=0.229) | FISURA  88.3% (P=0.883) | SANA  80.4% (P=0.196) | DISCREPANCIA (1/4 fisura) |
| 15.jpeg | Negative | SANA  77.5% (P=0.225) | SANA  99.8% (P=0.002) | SANA  96.6% (P=0.034) | FISURA  99.9% (P=0.999) | DISCREPANCIA (1/4 fisura) |
| 2.jpeg | Negative | FISURA  97.2% (P=0.972) | FISURA  59.8% (P=0.598) | SANA  99.3% (P=0.007) | FISURA  99.7% (P=0.997) | DISCREPANCIA (3/4 fisura) |
| 3.jpeg | Negative | FISURA  99.9% (P=0.999) | FISURA  99.8% (P=0.998) | FISURA  92.1% (P=0.921) | FISURA  99.9% (P=0.999) | FISURA (4/4) |
| 4.jpeg | Negative | SANA  81.0% (P=0.190) | SANA  99.6% (P=0.004) | FISURA  97.4% (P=0.974) | FISURA 100.0% (P=1.000) | DISCREPANCIA (2/4 fisura) |
| 5.jpeg | Negative | FISURA  91.6% (P=0.916) | SANA  99.0% (P=0.010) | SANA  77.0% (P=0.230) | SANA  91.2% (P=0.088) | DISCREPANCIA (1/4 fisura) |
| 7.jpeg | Negative | FISURA  93.4% (P=0.934) | FISURA  93.6% (P=0.936) | SANA  94.4% (P=0.056) | FISURA  71.5% (P=0.715) | DISCREPANCIA (3/4 fisura) |
| 8.jpeg | Negative | FISURA  99.3% (P=0.993) | FISURA  99.5% (P=0.995) | FISURA 100.0% (P=1.000) | FISURA  98.6% (P=0.986) | FISURA (4/4) |
| columna.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| columna2.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  99.9% (P=0.999) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| fisuraColumnaPared.jpeg | Negative | SANA  60.4% (P=0.396) | SANA  97.0% (P=0.030) | SANA  98.5% (P=0.015) | SANA  90.4% (P=0.096) | SANA (4/4) |
| fisuraColumnaPared2.jpeg | Negative | FISURA 100.0% (P=1.000) | SANA  73.9% (P=0.261) | FISURA  74.8% (P=0.748) | SANA  80.0% (P=0.200) | DISCREPANCIA (2/4 fisura) |
| fisuraColumnaPared3.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  99.3% (P=0.993) | FISURA  55.0% (P=0.550) | FISURA  96.3% (P=0.963) | FISURA (4/4) |
| fisuraColumnaPared34.jpeg | Negative | FISURA  99.6% (P=0.996) | FISURA  98.4% (P=0.984) | FISURA 100.0% (P=1.000) | FISURA  94.2% (P=0.942) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.13 PM (1).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  98.4% (P=0.984) | FISURA  99.4% (P=0.994) | FISURA  99.9% (P=0.999) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.13 PM.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA  98.5% (P=0.985) | FISURA  99.1% (P=0.991) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.14 PM (1).jpeg | Negative | FISURA  99.8% (P=0.998) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA  98.8% (P=0.988) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.14 PM.jpeg | Negative | FISURA  93.6% (P=0.936) | FISURA  73.1% (P=0.731) | SANA  57.4% (P=0.426) | FISURA 100.0% (P=1.000) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.15 PM (1).jpeg | Negative | FISURA  97.6% (P=0.976) | FISURA  99.6% (P=0.996) | FISURA  90.7% (P=0.907) | FISURA  99.9% (P=0.999) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.15 PM.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.16 PM.jpeg | Negative | FISURA  94.6% (P=0.946) | SANA  58.9% (P=0.411) | SANA  99.2% (P=0.008) | SANA  77.4% (P=0.226) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.17 PM.jpeg | Negative | FISURA  92.3% (P=0.923) | FISURA  88.4% (P=0.884) | FISURA  90.3% (P=0.903) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.35 PM (1).jpeg | Negative | SANA  69.1% (P=0.309) | SANA  66.2% (P=0.338) | SANA  98.7% (P=0.013) | SANA  89.9% (P=0.101) | SANA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.35 PM (2).jpeg | Negative | FISURA 100.0% (P=1.000) | SANA  87.2% (P=0.128) | FISURA  55.1% (P=0.551) | SANA  80.3% (P=0.197) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.35 PM (3).jpeg | Negative | FISURA  99.7% (P=0.997) | FISURA  99.0% (P=0.990) | SANA  98.0% (P=0.020) | FISURA 100.0% (P=1.000) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.35 PM.jpeg | Negative | FISURA  97.5% (P=0.975) | FISURA  97.8% (P=0.978) | FISURA  99.9% (P=0.999) | FISURA  99.6% (P=0.996) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.36 PM (1).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  53.3% (P=0.533) | SANA  95.8% (P=0.042) | FISURA  87.2% (P=0.872) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.36 PM (2).jpeg | Negative | FISURA  99.9% (P=0.999) | FISURA  96.2% (P=0.962) | FISURA  95.3% (P=0.953) | FISURA  97.7% (P=0.977) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.36 PM (3).jpeg | Negative | FISURA  99.9% (P=0.999) | FISURA  90.0% (P=0.900) | SANA  99.8% (P=0.002) | FISURA  99.9% (P=0.999) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.36 PM.jpeg | Negative | FISURA  63.3% (P=0.633) | SANA  97.6% (P=0.024) | SANA  99.9% (P=0.001) | FISURA  78.3% (P=0.783) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.37 PM (1).jpeg | Negative | FISURA  99.7% (P=0.997) | FISURA  63.8% (P=0.638) | FISURA  97.2% (P=0.972) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.37 PM (2).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  98.4% (P=0.984) | SANA  99.8% (P=0.002) | FISURA  99.4% (P=0.994) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.37 PM (3).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.37 PM (4).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.37 PM.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.38 PM (1).jpeg | Negative | FISURA  99.9% (P=0.999) | FISURA  72.9% (P=0.729) | FISURA  94.1% (P=0.941) | FISURA  55.5% (P=0.555) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.38 PM (2).jpeg | Negative | FISURA  99.7% (P=0.997) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.38 PM (3).jpeg | Negative | SANA  90.1% (P=0.099) | SANA  98.6% (P=0.014) | FISURA  72.6% (P=0.726) | FISURA 100.0% (P=1.000) | DISCREPANCIA (2/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.01.38 PM.jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  99.5% (P=0.995) | FISURA  98.0% (P=0.980) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.39 PM (1).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA  99.9% (P=0.999) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.39 PM (2).jpeg | Negative | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA 100.0% (P=1.000) | FISURA (4/4) |
| WhatsApp Image 2026-08-16 at 7.01.39 PM.jpeg | Negative | SANA  77.5% (P=0.225) | SANA  99.8% (P=0.002) | SANA  96.6% (P=0.034) | FISURA  99.9% (P=0.999) | DISCREPANCIA (1/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.43.25 AM.jpeg | Negative | FISURA  88.6% (P=0.886) | FISURA  97.9% (P=0.979) | SANA  81.2% (P=0.188) | FISURA  86.5% (P=0.865) | DISCREPANCIA (3/4 fisura) |
| WhatsApp Image 2026-08-16 at 7.43.28 AM.jpeg | Negative | FISURA  97.6% (P=0.976) | SANA  97.1% (P=0.029) | SANA  97.9% (P=0.021) | SANA  96.3% (P=0.037) | DISCREPANCIA (1/4 fisura) |

### Métricas OOD (solo fotos con etiqueta Positive/Negative)

n = 77 (Positive=30, Negative=47)

| Modelo | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| CNN personalizada | 0.377 | 0.350 | 0.700 | 0.467 |
| MobileNetV2 | 0.325 | 0.238 | 0.333 | 0.278 |
| ResNet50V2 | 0.234 | 0.061 | 0.067 | 0.063 |
| EfficientNet-B0 | 0.234 | 0.204 | 0.333 | 0.253 |


---
