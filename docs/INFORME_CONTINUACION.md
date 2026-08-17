# Continuación del informe — qué hacer mañana

**Para quién:** Axel / equipo CrackExpert AI, o un chat nuevo de Cursor.  
**Cuándo:** después de que termine `python main.py` (corrida de **8.000** imágenes).  
**Word:** `C:\Users\axela\Documents\Universidad\Sistema_Experto\Proyecto_SistemaExperto_HernandezAxel.docx`  
**Norte del proyecto:** [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) · SE: [`EXPERT_SYSTEM_SPEC.md`](EXPERT_SYSTEM_SPEC.md)

Pegar en un chat nuevo:

> Lee `docs/PROJECT_CONTEXT.md` y `docs/INFORME_CONTINUACION.md`. No reinventes el dominio. Redacta §4, §5, §6, §7 y §9 con cifras literales de `reports/` de la corrida 8k. No ejecutes `python main.py`. No inventes métricas.

---

## 0. Decisiones ya tomadas (no reabrir)

- Seguimos con **CrackExpert AI** (híbrido CNN + SE). No cambiar de dominio.
- ML = núcleo experimental (4 arquitecturas, protocolo 8k). SE = integración del resultado (ACI/NEC + CF MYCIN), no el 50 % del informe.
- Historia científica: en Kaggle F1 puede ir ~1,0 (**dominio saturado**); en fotos de casa (n=71) el F1 OOD de la piloto 4k fue bajo (CNN 0,420). Eso es **domain shift**, no “el modelo está roto”.
- No vender F1=1,0 como problema resuelto. Accuracy OOD es engañosa (24 Positive / 47 Negative): usar **F1 y recall**.
- Criterio de modelo final: **Kaggle + OOD + latencia**, no solo CNN custom por ~6 ms.
- Si el 8k sigue ~1,0 en test Kaggle: concluir **saturación de dominio**, no fracaso ni “hace falta U-Net / 10 arquitecturas”.
- Mejoras viables en §7: más OOD, augmentation tipo móvil. **No** 10 arquitecturas ni U-Net en esta entrega.
- El SE no es peritaje; no calcula φVn. “Crítica” = alerta de protocolo.
- En campo: foto + elemento + ambiente; **ancho no se pide**; orientación la infiere OpenCV.

---

## 1. Qué ya está en el Word (no reescribir de cero)

| Sección | Estado (16 ago 2026, noche) |
| --- | --- |
| Portada, índice | Hechos (revisar mayúsculas / nombre docente si falta) |
| §1.3–1.4 | Objetivos con **8.000**; alcance alineado (sin ancho en campo) |
| §2 | Protocolo 8k, tabla 2.6 (5.600 / 1.200 / 1.200) |
| §3 | Flujo CNN + OpenCV + SE; tablas 3.4.1–3.4.2; reglas; MYCIN; JSON |
| §8 | Redactada + **capturas** (pies Figura 8.1–8.6). Default de app: `mobilenet_v2.keras` hasta decidir modelo 8k |
| §6–7, §4–5 números, §9 | **Pendientes de la corrida 8k** |
| §10, anexos | Se pueden pegar sin 8k (borrador al final de este archivo) |

Índice del Word (subtítulos a respetar):

- **4.** Implementación y calidad del entrenamiento [rúbrica 1 y 2]  
  4.1 Stack · 4.2 Arquitectura CNN · 4.3 Hiperparámetros · 4.4 Regularización / augmentation · 4.5 Curvas · 4.6 EarlyStopping / sobreajuste  
- **5.** Evaluación [rúbrica 3]  
  5.1 Acc, Prec, Rec, F1 (val y test) · 5.2 Matrices · 5.3 ROC-AUC · 5.4 Latencia  
- **6.** Pruebas reales [rúbrica 4]  
  6.1 Protocolo celular OOD · 6.2 Tabla 71 fotos · 6.3 Dictamen SE  
- **7.** Análisis crítico [rúbrica 5]  
- **9.** Conclusiones  

Tablas HTML de apoyo (2.6 y 3.4.2): `docs/tablas_informe.html`.

---

## 2. Checklist al cerrar el entrenamiento (antes de redactar)

No inventar números. Si `models_comparison.csv` sigue diciendo 4.000 / test n=600 / fecha 02:06:32, **la corrida 8k no terminó**.

1. Abrir `reports/experiments_log.md` → debe haber una **Corrida** nueva (después de `2026-08-16 02:06:32`) con subconjunto **8.000** y test **n=1.200**.
2. Abrir `reports/models_comparison.csv` → copiar **tal cual** (decimales del CSV).
3. Abrir `reports/archive/` → debe existir snapshot de la piloto 4k (figuras/CSV viejos).
4. Figuras vigentes en `reports/figures/`:
   - `learning_curves_cnn_custom.png`
   - `learning_curves_mobilenet_v2.png`
   - `learning_curves_resnet50_v2.png`
   - `learning_curves_efficientnet_b0.png`
   - `confusion_matrix_*.png` (cuatro)
   - `roc_curves_comparison.png`
   - `misclassified_examples.png` (puede estar vacía de errores si F1=1,0)
5. Historiales: `reports/history_*.json` (épocas, si EarlyStopping cortó).
6. OOD: `reports/external_test_comparison.md` → **última** corrida (la de más abajo). Debe ser n=71 etiquetadas. Si `main.py` no la regeneró: `python test_external.py`.
7. `reports/TRAINING_RESULTS.md` → anexar lectura de la corrida 8k si el script no lo hizo.

### 2.1. Si Kaggle 8k ≈ 1,0 otra vez

Frase canónica para §5 y §9:

> Duplicar el subconjunto de 4.000 a 8.000 no rompe el techo del dominio Surface Crack Detection. El F1 próximo a 1,0 indica saturación visual del corpus de campus, no que la patología en obra esté resuelta. La discriminación entre arquitecturas se lee en OOD y en latencia.

### 2.2. Modelo que carga la app

`app.py` → `DEFAULT_MODEL_NAME = "mobilenet_v2.keras"`.  
Después de ver F1 OOD + latencia 8k, cambiar a una línea el `.keras` ganador y una frase en §8.1. No hace falta reentrenar.

---

## 3. De dónde sale cada cifra (informe ← reports/)

| Dato | Archivo |
| --- | --- |
| Acc, Prec, Rec, F1, AUC, MB, params, latencia (Kaggle) | `reports/models_comparison.csv` |
| Protocolo, semilla, 2 fases | `reports/experiments_log.md` (entrada 8k) |
| Épocas, val_loss, corte EarlyStopping | `reports/history_*.json` + curvas PNG |
| Matrices / ROC / curvas | `reports/figures/*.png` |
| OOD n=71, P por foto, consenso, F1 | **Última** sección de `reports/external_test_comparison.md` y `.csv` |
| Relato piloto 4k (delta) | `reports/TRAINING_RESULTS.md` Corrida 01 · archive |

**Piloto 4k (solo para comparar, no como tabla oficial de titulación):**  
Test n=600. CNN F1=1,0, latencia 6,444 ms. OOD 19:08:31 — CNN F1=0,420 / MN=0,294 / RN=0,140 / EN=0,121. Recall CNN=0,708. Acc OOD no destacar.

**OOD vigente hasta que corra 8k:** misma tabla 71 fotos; las **P** cambiarán con pesos nuevos.

---

## 4. Qué redactar mañana (orden recomendado)

### §4 — rúbrica 1 y 2

- 4.1–4.3: se puede reutilizar el protocolo ya escrito en chats (Adam, BCE, LR 1e-3 / 1e-5, batch 32, ES patience 5, augmentation solo train). Sustituir N: train 5.600 / val 1.200 / test 1.200.
- 4.5–4.6: pegar las **cuatro** curvas; decir si EarlyStopping cortó (en 4k solo MobileNet, 27 épocas). Interpretar: proceso controlado, no divergencia al abrir Fase 2.
- Una nota de 4–6 líneas: piloto 4k vs 8k (si F1 sigue 1,0 → saturación).

### §5 — rúbrica 3

- Tabla literal del CSV. Interpretar Precision/Recall (FP vs FN).
- Matrices: TN/FP/FN/TP leídos de las PNG (o de sklearn en evaluate).
- ROC. Interpretación: techo Kaggle ≠ obra. Puente a §6.

### §6 — rúbrica 4

- 6.1: protocolo celular ya borrado en el chat (71 fotos, 24/47, WhatsApp, umbral 0,50, `test_external.py`). Pegar y ajustar fecha de corrida 8k.
- 6.2: **no fingir alta precisión** si F1 OOD sigue ~0,4. Evidencia = carpeta `data/external_test/` + markdown. Tabla resumen 4 modelos + anexo con 71 filas. 4–6 fotos TP/FP/FN/TN.
- 6.3: SE con Figuras 8.3–8.4; R0 / R6b / R-P1.

### §7 — rúbrica 5

- Causas: juntas, textura, luz, compresión WhatsApp, encuentro columna-pared, dataset campus vs casa.
- Mejoras **viables**: más OOD etiquetado, augmentation (JPEG, brillo, perspectiva), umbral o coste de FN/FP. No U-Net ahora.
- Si CNN sigue siendo la de mayor F1 OOD y menor latencia, decirlo; si no, el ganador es el compromiso Kaggle+OOD+ms.

### §9

- Híbrido justificado. Percepción saturada in-domain. OOD = límite real. SE = normas + CF, no peritaje. Próximo paso: datos de campo, no más backbones de moda.

### §8 (retoque de una frase)

- Actualizar modelo `.keras` y latencia 8k si se midió. Las capturas ya están.

---

## 5. Código: no hace falta para redactar

Hecho: `main.py`, 4 modelos, archive, `test_external.py`, SE, geometría, `app.py`, `run_app.py`, visitas.

Opcional **después** de elegir modelo: `DEFAULT_MODEL_NAME` en `app.py`.

No ejecutar otro `python main.py` “por si acaso” si la corrida 8k ya acabó: pisa figuras (el archive debería haber guardado la anterior).

---

## 6. Borrador §10 — Referencias (pegar cuando se quiera)

Orden aproximado: normas, dataset, ML, SE, software.

1. ACI Committee 224. *ACI 224R-01: Control of Cracking in Concrete Structures*. American Concrete Institute, 2001 (reaprobado según edición usada).
2. ACI Committee 318. *Building Code Requirements for Structural Concrete (ACI 318) and Commentary*. American Concrete Institute.
3. MIDUVI. *NEC-SE-HM: Hormigón Armado*. Norma Ecuatoriana de la Construcción. Quito.
4. Özgenel, Ç. F.; Sorguç, A. G. Surface Crack Detection Dataset. Middle East Technical University. Difusión: Mendeley Data / Kaggle (`arunrk7/surface-crack-detection`).
5. Shortliffe, E. H.; Buchanan, B. G. A model of inexact reasoning in medicine. *Mathematical Biosciences*, 1975. (Factores de certeza MYCIN.)
6. He, K.; Zhang, X.; Ren, S.; Sun, J. Identity mappings in deep residual networks. *ECCV*, 2016. (ResNetV2.)
7. Sandler, M.; Howard, A.; Zhu, M.; Zhmoginov, A.; Chen, L.-C. MobileNetV2: inverted residuals and linear bottlenecks. *CVPR*, 2018.
8. Tan, M.; Le, Q. EfficientNet: rethinking model scaling for convolutional neural networks. *ICML*, 2019.
9. Chollet, F.; et al. Keras / TensorFlow. https://keras.io
10. Goodfellow, I.; Bengio, Y.; Courville, A. *Deep Learning*. MIT Press, 2016. (opcional, fundamentos.)
11. Giarratano, J.; Riley, G. *Expert Systems: Principles and Programming*. (o texto equivalente de SE usado en la cátedra.)

Completar con las ediciones exactas que tenga el equipo (NEC año, ACI 318 año). No citar métricas aquí.

---

## 7. Anexos sugeridos (sin números 8k)

- A. Hiperparámetros (`src/train.py`: épocas 20+20, LR, ES).
- B. CNN custom (`build_cnn_custom` en `src/models.py`).
- C. Contrato JSON (ejemplo didáctico de §3.5, no como resultado de test).
- D. Tabla completa OOD 71 filas (**mañana**, desde el `.md` último).
- E. Comandos: `python main.py` · `python test_external.py` · `python run_app.py`.

---

*Documento creado 16 de agosto de 2026 (noche), para retomar al día siguiente de la corrida 8k.*
