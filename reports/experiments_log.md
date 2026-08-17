# Bitácora experimental — CrackExpert AI

Registro cronológico de corridas de entrenamiento y evaluación.

Las entradas se anexan automáticamente al finalizar `python main.py`.
Cada corrida documenta configuración, métricas por split de test y la
justificación del modelo óptimo (máximo F1 en test, desempate por ROC-AUC).
## Corrida 2026-08-16 02:06:32

### Configuración

- Dataset: `arunrk7/surface-crack-detection`
- Subconjunto: 4.000 imágenes (2.000 Positive / 2.000 Negative)
- Splits estratificados (`random_state=42`): train 2.800 / val 600 / test 600
- Input: 224×224 RGB, rango [0, 255]
- Augmentation (solo train): RandomFlip, RandomRotation(0.1), RandomZoom(0.1), RandomBrightness(0.1)
- Optimizador: Adam | Loss: binary_crossentropy
- Fase 1: LR=1e-3, backbone congelado (CNN: entrenamiento completo)
- Fase 2: LR=1e-5, fine-tuning (MobileNetV2: 25 capas; ResNet50V2/EfficientNet-B0: 20 capas)
- EarlyStopping: monitor=val_loss, patience=5, restore_best_weights=True

### Métricas por modelo (test independiente n=600)

| Modelo | Parametros_Totales | Tamano_MB | Latencia_ms | Val_Accuracy | Test_Accuracy | Precision | Recall | F1_Score | ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN personalizada | 110561 | 1.343 | 6.444 | 0.9933 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| MobileNetV2 | 2259265 | 19.491 | 11.129 | 0.9967 | 0.9967 | 0.9934 | 1.0 | 0.9967 | 1.0 |
| ResNet50V2 | 23566849 | 150.61 | 22.302 | 0.9967 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| EfficientNet-B0 | 4050852 | 26.527 | 13.999 | 0.9967 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### Selección del modelo óptimo

Se selecciona **CNN personalizada** por mayor F1-Score en test (1.0), con ROC-AUC=1.0 y latencia=6.444 ms/imagen. El criterio prioriza el equilibrio precisión/exhaustividad sobre el conjunto retenido, que no participó en el ajuste de hiperparámetros.

---
## Corrida 2026-08-17 02:22:23

### Configuración

- Dataset: `arunrk7/surface-crack-detection`
- Subconjunto: 8.000 imágenes (4.000 Positive / 4.000 Negative)
- Splits estratificados (`random_state=42`): train 5.600 / val 1.200 / test 1.200
- Input: 224×224 RGB, rango [0, 255]
- Augmentation (solo train): RandomFlip, RandomRotation(0.1), RandomZoom(0.1), RandomBrightness(0.1)
- Optimizador: Adam | Loss: binary_crossentropy
- Fase 1: LR=1e-3, backbone congelado (CNN: entrenamiento completo)
- Fase 2: LR=1e-5, fine-tuning (MobileNetV2: 25 capas; ResNet50V2/EfficientNet-B0: 20 capas)
- EarlyStopping: monitor=val_loss, patience=5, restore_best_weights=True

### Métricas por modelo (test independiente n=1200)

| Modelo | Parametros_Totales | Tamano_MB | Latencia_ms | Val_Accuracy | Test_Accuracy | Precision | Recall | F1_Score | ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN personalizada | 110561 | 1.343 | 6.499 | 0.9983 | 0.9983 | 1.0 | 0.9967 | 0.9983 | 1.0 |
| MobileNetV2 | 2259265 | 19.491 | 11.31 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| ResNet50V2 | 23566849 | 150.61 | 22.666 | 1.0 | 0.9983 | 0.9967 | 1.0 | 0.9983 | 1.0 |
| EfficientNet-B0 | 4050852 | 26.527 | 14.071 | 0.9992 | 0.9992 | 0.9983 | 1.0 | 0.9992 | 1.0 |

### Selección del modelo óptimo

Se selecciona **MobileNetV2** por mayor F1-Score en test (1.0), con ROC-AUC=1.0 y latencia=11.31 ms/imagen. El criterio prioriza el equilibrio precisión/exhaustividad sobre el conjunto retenido, que no participó en el ajuste de hiperparámetros.

---
