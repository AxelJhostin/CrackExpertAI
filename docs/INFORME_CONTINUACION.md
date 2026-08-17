# Continuación del informe — qué hacer mañana

**Para quién:** Axel / equipo CrackExpert AI, o un chat nuevo de Cursor.  
**Cuándo:** la corrida de **8.000** ya terminó (`2026-08-17 02:22:23`). Este archivo es el briefing para **redactar** §4–7 y §9.  
**Word (copia en el repo):** [`docs/Proyecto_SistemaExperto_HernandezAxel.docx`](Proyecto_SistemaExperto_HernandezAxel.docx)  
(También puede haber copia en Documentos; para el chat nuevo usar la de `docs/`.)  
**Norte del proyecto:** [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) · SE: [`EXPERT_SYSTEM_SPEC.md`](EXPERT_SYSTEM_SPEC.md)

Pegar en un chat nuevo:

> Lee `docs/PROJECT_CONTEXT.md` y `docs/INFORME_CONTINUACION.md`. No reinventes el dominio. Redacta §4, §5, §6, §7 y §9 con cifras literales de `reports/` de la corrida 8k. No ejecutes `python main.py`. No inventes métricas.

---

## 0. Decisiones ya tomadas (no reabrir)

- Seguimos con **CrackExpert AI** (híbrido CNN + SE). No cambiar de dominio.
- ML = núcleo experimental (4 arquitecturas, protocolo 8k). SE = integración del resultado (ACI/NEC + CF MYCIN), no el 50 % del informe.
- Historia científica: en Kaggle F1 ≥ 0,998 (**dominio saturado**); en fotos de casa n=77 (modelos 8k) el F1 OOD de la CNN es **0,467** (recall 0,70). MobileNet gana Kaggle (F1 1,0) y pierde en campo (F1 0,278). Eso es **domain shift**.
- No vender F1=1,0 como problema resuelto. Accuracy OOD es engañosa (30 Positive / 47 Negative): usar **F1 y recall**.
- Criterio de modelo: **Kaggle + OOD + latencia**. CSV → MobileNet; campo → CNN (más recall, 6,5 ms).
- Duplicar 4k→8k **no** rompió el techo Kaggle: saturación de dominio, no hace falta U-Net / 10 arquitecturas.
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
| §6–7, §4–5 números, §9 | **Listos para redactar** (8k + OOD 77 en `reports/`) |
| §10, anexos | Se pueden pegar sin 8k (borrador al final de este archivo) |

Índice del Word (subtítulos a respetar):

- **4.** Implementación y calidad del entrenamiento [rúbrica 1 y 2]  
  4.1 Stack · 4.2 Arquitectura CNN · 4.3 Hiperparámetros · 4.4 Regularización / augmentation · 4.5 Curvas · 4.6 EarlyStopping / sobreajuste  
- **5.** Evaluación [rúbrica 3]  
  5.1 Acc, Prec, Rec, F1 (val y test) · 5.2 Matrices · 5.3 ROC-AUC · 5.4 Latencia  
- **6.** Pruebas reales [rúbrica 4]  
  6.1 Protocolo celular OOD · 6.2 Tabla **77** fotos (30/47) · 6.3 Dictamen SE  
- **7.** Análisis crítico [rúbrica 5]  
- **9.** Conclusiones  

Tablas HTML de apoyo (2.6 y 3.4.2): `docs/tablas_informe.html`.

---

## 2. Checklist al cerrar el entrenamiento (antes de redactar)

No inventar números. La corrida 8k **ya está**: `experiments_log.md` entrada `2026-08-17 02:22:23`, test n=1.200.

Checklist de archivos (todo debería existir):

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
6. OOD: `reports/external_test_comparison.md` → **última** corrida `2026-08-17 02:22:55`, **n=77** (30/47).
7. `reports/TRAINING_RESULTS.md` → Corrida 02 anexada.

### 2.1. Kaggle 8k ≈ 1,0 (ya ocurrió)

Frase canónica para §5 y §9:

> Duplicar el subconjunto de 4.000 a 8.000 no rompe el techo del dominio Surface Crack Detection. El F1 próximo a 1,0 indica saturación visual del corpus de campus, no que la patología en obra esté resuelta. La discriminación entre arquitecturas se lee en OOD y en latencia.

### 2.2. Modelo que carga la app

`app.py` → `DEFAULT_MODEL_NAME = "mobilenet_v2.keras"` (coincide con el F1 Kaggle 8k).  
En §8: una frase honesta — en campus gana MobileNet; en fotos de celular la CNN recupera más fisuras (recall 0,70 vs 0,33). No hace falta reentrenar para el informe.

---

## 3. De dónde sale cada cifra (informe ← reports/)

| Dato | Archivo |
| --- | --- |
| Acc, Prec, Rec, F1, AUC, MB, params, latencia (Kaggle) | `reports/models_comparison.csv` |
| Protocolo, semilla, 2 fases | `reports/experiments_log.md` (entrada 8k) |
| Épocas, val_loss, corte EarlyStopping | `reports/history_*.json` + curvas PNG |
| Matrices / ROC / curvas | `reports/figures/*.png` |
| OOD n=77, P por foto, consenso, F1 | **Última** sección de `reports/external_test_comparison.md` y `.csv` |
| Relato piloto 4k (delta) | `reports/TRAINING_RESULTS.md` Corrida 01 · archive |

**Piloto 4k (solo para comparar):** Test n=600. CNN F1=1,0, latencia 6,444 ms. OOD 19:08:31 n=71 — CNN F1=0,420 / MN=0,294 / RN=0,140 / EN=0,121.

**OOD vigente (pesos 8k):** n=77 (30/47), `2026-08-17 02:22:55`. CNN F1=0,467 rec=0,700 · MN 0,278 · EN 0,253 · RN 0,063.

**Kaggle vigente:** MN F1=1,0 · EN 0,9992 · CNN 0,9983 (2 FN) · RN 0,9983 (2 FP). AUC=1,0.

---

## 4. Qué redactar mañana (orden recomendado)

### §4 — rúbrica 1 y 2

- 4.1–4.3: se puede reutilizar el protocolo ya escrito en chats (Adam, BCE, LR 1e-3 / 1e-5, batch 32, ES patience 5, augmentation solo train). Sustituir N: train 5.600 / val 1.200 / test 1.200.
- 4.5–4.6: pegar las **cuatro** curvas 8k. EarlyStopping cortó en las cuatro (CNN 36, MN 33, RN 26, EN 29 épocas). Interpretar: proceso controlado.
- Una nota de 4–6 líneas: piloto 4k vs 8k (si F1 sigue 1,0 → saturación).

### §5 — rúbrica 3

- Tabla literal del CSV. Interpretar Precision/Recall (FP vs FN).
- Matrices: TN/FP/FN/TP leídos de las PNG (o de sklearn en evaluate).
- ROC. Interpretación: techo Kaggle ≠ obra. Puente a §6.

### §6 — rúbrica 4

- 6.1: protocolo celular, umbral 0,50, `test_external.py`. **n=77** (30 Positive / 47 Negative), corrida `2026-08-17 02:22:55`. El piloto 4k era n=71 (24/47): no mezclar.
- 6.2: **no fingir alta precisión**. F1 OOD CNN 0,467. Evidencia = `data/external_test/` + markdown. Tabla 4 modelos + anexo. 4–6 fotos TP/FP/FN/TN.
- 6.3: SE con Figuras 8.3–8.4; cartas de campo; R0 / R6b / R-P1.
- Si preguntan “¿por qué no 2.000 fotos de obra?”: el OOD **no entrena** el modelo; **falsifica** el F1 de Kaggle. En hormigón no hay un cajón de muestras como en botellas. Ver recuadro más abajo.

### §7 — rúbrica 5

- Causas: juntas, textura, luz, JPEG/WhatsApp, encuentro columna-pared, dataset de campus vs casa.
- Mejoras **de esta entrega** (viables, no hechas): más OOD balanceado (150–300), augmentation tipo móvil. **No** U-Net ni 10 arquitecturas.
- CNN: mayor F1 OOD (0,467) y menor latencia (6,5 ms). MobileNet: F1 Kaggle 1,0 y F1 OOD 0,278.

**Recriminación del proyecto anterior (botellas) vs este (hormigón) — no mezclar rúbricas**

- Botellas: el objeto cabe en la mesa; se pueden juntar decenas o cientos de muestras reales **para entrenar**. Ahí “pocas fotos reales” es un hueco del *dataset de entrenamiento*.
- Fisuras en elementos: no hay un inventario en el laboratorio. Las 8.000 son el corpus público etiquetado (presencia/ausencia). Las 77 de celular son **prueba de desplazamiento**, no el train.
- Decirlo en §7 sin victimizarse: el límite es el **acceso a obra etiquetada**, no que se haya “olvidado” el campo.

### §9

- Entrega **cerrada**: híbrido CNN + SE, protocolo 8k, OOD, prototipo. No es un detector de patología de obra.
- Percepción saturada in-domain; OOD = límite real; SE = normas + CF, no peritaje.
- **Trabajo futuro (tesis / civil), no excusa de esta nota:** un sistema no binario (*Structural Damage AI*) que separe fisura, desprendimiento, corrosión, acero expuesto, con más datos de campo, dimensiones y prioridad de inspección. Eso exige tiempo de obra y etiquetado que esta asignatura no cubre.
- Próximo paso técnico: datos de campo y taxonomía de daño, no más backbones de moda.

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
