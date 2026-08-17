# CrackExpert AI — contexto canónico del proyecto

**Léeme primero** si abres un chat nuevo, si el hilo se alarga o si otra persona/agente retoma el repo.

- **Asignatura:** Sistemas Expertos  
- **Institución:** PUCE Sede Manabí  
- **Naturaleza:** proyecto académico de evaluación/titulación con rigor científico, reproducibilidad y aplicabilidad en ingeniería civil  
- **Normas:** ACI 224R-01, ACI 318, NEC-SE-HM  
- **Especificación del SE (detalle):** [`EXPERT_SYSTEM_SPEC.md`](EXPERT_SYSTEM_SPEC.md)  
- **Informe de entrenamientos:** [`../reports/TRAINING_RESULTS.md`](../reports/TRAINING_RESULTS.md)

---

## 0. Cómo retomar el trabajo (para un chat nuevo)

Pegar o adjuntar este archivo y decir:

> Lee `docs/PROJECT_CONTEXT.md`. Es el norte de CrackExpert AI (híbrido CNN + SE, rúbrica de la ingeniera, estado del repo). No reinventes el alcance. El siguiente entregable es redactar §4–7 y §9 (`docs/INFORME_CONTINUACION.md`) con cifras de la corrida 8k. No ejecutes `python main.py`.

---

## 1. Por qué existe este documento

El alcance se formuló con detalle y, en otra conversación larga, **se perdió el hilo**. Este `.md` es la memoria estable del proyecto: propósito, diseño, rúbrica, estructura del informe, estado del código y siguiente paso.

También es un **compromiso de continuidad con el grupo**. Hubo un fallo en otro proyecto que dejó gente a medias. CrackExpert AI se está construyendo para **no repetir eso**: entregable completo, trazable y defendible ante la rúbrica, no un prototipo a medias que nadie puede retomar.

El sistema **no sustituye** a un ingeniero civil. Es apoyo a la inspección y a la evaluación académica.

---

## 2. Qué estamos haciendo y por qué (idea central)

### 2.1. Problema real

La inspección visual manual de fisuras en hormigón armado (obra y post-sismo) es lenta, costosa, subjetiva y no escala.

### 2.2. Por qué no basta el ML solo

Una CNN es una caja negra **perceptiva**. Entrega \(P_{\mathrm{ML}} \in [0,1]\) (“¿hay una grieta en la foto?”). No sabe si el elemento es una columna crítica o un muro divisorio, ni si el ambiente es marino o interior seco, ni qué dice ACI/NEC sobre el ancho.

### 2.3. Por qué no basta el sistema experto solo

Un SE de reglas codifica normas, pero **no procesa** una matriz de píxeles tomada con el celular.

### 2.4. Solución: pipeline híbrido desacoplado

| Capa | Rol | Qué no hace |
| --- | --- | --- |
| **1a. Percepción (CNN)** | \(P_{\mathrm{ML}}\): ¿hay fisura en la foto? | No interpreta patología ni normas |
| **1b. Geometría (OpenCV)** | Orientación del trazo: vertical / horizontal / inclinada / malla | No distingue helicoidal; no mide mm |
| **2. Razonamiento (SE normativo)** | Reglas ACI/NEC + CF estilo MYCIN | No “ve” la imagen |
| **3. Bitácora local** | Visita de obra (lugar + hora) con N fotos | No es un sistema de gestión en la nube |

**Contrato entre capas:** la CNN aporta \(P_{\mathrm{ML}}\). OpenCV aporta la **orientación** y se mapea a `patron_orientacion` según el tipo de elemento. El inspector solo declara lo que la foto de cerca **no puede saber**: elemento y **ambiente por foto** (una visita puede mezclar baño húmedo y fachada seca). El ancho mm queda fuera del flujo principal (evidencia incompleta en el SE). El experto emite severidad, mecanismo, fundamento, plan y CF.

Cambiar una regla ACI **no** implica reentrenar. Cambiar el backbone **no** cambia \(w_{\max}\). La geometría **no** exige un dataset etiquetado de orientaciones.

**UI (criterio de producto, ago-2026):** lenguaje llano; sin selector de modelo a la vista; detalle técnico (CF, reglas, JSON) en expander. El usuario no rellena el patrón a mano salvo corrección.

---

## 3. Sistema experto (resumen operativo)

Detalle normativo y reglas SI–ENTONCES: `docs/EXPERT_SYSTEM_SPEC.md`. Código: `src/expert_system.py`. UI: `app.py`.

### 3.1. Entradas

1. \(P_{\mathrm{ML}}\) (CNN, umbral operativo 0,50).  
2. **Elemento:** Viga, Columna, Losa, Muro.  
3. **Ambiente y \(w_{\mathrm{adm}}\)** (brief de titulación):

| Ambiente | \(w_{\mathrm{adm}}\) (mm) |
| --- | ---: |
| Interior seco | 0,41 |
| Exterior húmedo | 0,30 |
| Marino / costero | 0,18 |
| Retención de agua | 0,10 |

*Nota de implementación:* el código puede usar 0,15 mm (ACI 224R “seawater”) para marino y no exponer aún “retención de agua” en la UI. Alinear con esta tabla si la ingeniera pide el brief al pie de la letra.

4. **Ancho** \(w\) (mm), opcional; si falta, evidencia incompleta. En la app de campo **no se pide**; el motor usa `ancho_mm=None`.  
5. **`patron_orientacion`:** en campo lo **infiere** `src/crack_geometry.py` + `patron_from_geometry(elemento, clase)`. El inspector puede corregirlo en el expander. Mapeo:

- Inclinada → Diagonal (~45° en apoyos)  
- Vertical + columna → Vertical paralela al eje  
- Vertical + viga/losa/muro → Vertical / perpendicular (vano)  
- Horizontal + columna → Horizontal transversal  
- Horizontal + otros → Longitudinal paralela al refuerzo  
- Malla → Malla / piel de cocodrilo  
- Helicoidal: no se infiere en 2D  

Tabla de mecanismos (motor, sin cambio de reglas):

| Patrón (interfaz) | Mecanismo | Severidad típica |
| --- | --- | --- |
| Diagonal (~45° en apoyos) | Cortante / tensión diagonal | Crítica |
| Vertical paralela al eje (columna) | Compresión / aplastamiento | Moderada–crítica |
| Vertical / perpendicular (centro de vano, viga) | Flexión pura | Según \(w\) vs \(w_{\mathrm{adm}}\) |
| Horizontal transversal (columna) | Flexo-tracción sísmica / viento | Crítica |
| Helicoidal / espiral (45°) | Torsión | Crítica |
| Malla / piel de cocodrilo | Retracción / curado deficiente | Leve–moderada |
| Longitudinal paralela al refuerzo | Corrosión / despasivación | Moderada–crítica |

### 3.2. Salidas

- Severidad: Leve (estético), Moderada (durabilidad/corrosión), Crítica (riesgo estructural).  
- Mecanismo físico + cita ACI/NEC.  
- Plan de acción (sellado, inyección, apuntalamiento, etc.).  
- \(\mathrm{CF}\) combinado (MYCIN en código; el brief original también menciona \(\mathrm{CF}_{\mathrm{regla}} \times P_{\mathrm{ML}}\)).

---

## 4. Machine learning (protocolo)

- **Dataset:** Surface Crack Detection (Özgenel & Sorguç / Kaggle `arunrk7/surface-crack-detection`).  
- **Subconjunto vigente:** **8.000** imágenes (4.000 Positive / 4.000 Negative), 224×224 RGB.  
- **Partición** `random_state=42`: train 5.600 (2.800/2.800) · val 1.200 (600/600) · test 1.200 (600/600).  
- **Anti-leakage:** augmentation **solo train** (Flip, Rotation 0,1, Zoom 0,1, Brightness 0,1). Val/test sin aumento.  
- **Cuatro modelos:** CNN custom (baseline); MobileNetV2 (edge); ResNet50V2 (residual); EfficientNet-B0 (compound scaling).  
- **Entrenamiento:** 2 fases (backbone congelado LR \(10^{-3}\); fine-tuning LR \(10^{-5}\)), Adam, `binary_crossentropy`, EarlyStopping (`val_loss`, patience 5, `restore_best_weights`).  
- **OOD:** fotos reales en `data/external_test/` (`test_external.py`); tablas acumulativas en `reports/external_test_comparison.*`.  
- **Histórico de corridas ML:** `main.py` archiva figuras/CSV previos en `reports/archive/` antes de pisar `reports/figures/`.  
- **Corrida 4k (completada):** test \(n=600\); CNN por F1 empatado y latencia. Archive: `reports/archive/2026-08-16_083631/`.  
- **Corrida 8k (completada, 17 ago 2026 02:22:23):** test \(n=1.200\). MobileNetV2 F1=1,0 (criterio CSV). CNN F1=0,9983 (2 FN, 6,5 ms). EfficientNet 0,9992 (1 FP). ResNet 0,9983 (2 FP). AUC=1,0 los cuatro. Relato: `reports/TRAINING_RESULTS.md` Corrida 02.

---

## 5. Stack y estructura del repo

Python 3.10+, TensorFlow/Keras, NumPy, OpenCV (headless), scikit-learn, Matplotlib, Pillow, Pandas, Streamlit, kagglehub.

```text
crackexpert-ai/
├── data/raw, data/processed, data/external_test
├── data/inspections/       # visitas locales (JSON + JPG; no se versionan)
├── models/                 # *.keras (no se versionan pesos grandes)
├── reports/
│   ├── figures/            # corrida vigente
│   ├── archive/            # snapshots de corridas anteriores
│   ├── models_comparison.csv
│   ├── experiments_log.md
│   ├── external_test_comparison.csv / .md
│   └── TRAINING_RESULTS.md
├── src/   data_loader, models, train, evaluate, expert_system,
│          crack_geometry, inspections, archive
├── docs/  EXPERT_SYSTEM_SPEC.md, PROJECT_CONTEXT.md (este archivo)
├── .streamlit/config.toml  # LAN: 0.0.0.0:8501, CORS/XSRF off
├── app.py, run_app.py, main.py, test_external.py, README.md
```

Comandos:

```bash
python main.py              # datos 8k → 4 modelos → métricas → OOD
python test_external.py     # solo fotos reales (anexa corrida)
python run_app.py           # Streamlit en LAN (IP + puerto; libera 8501 si quedó colgado)
streamlit run app.py        # mismo prototipo, solo localhost
```

---

## 6. Rúbrica de la ingeniera (puntos máximos)

Escala por criterio: Excelente / Bueno / En desarrollo / Insuficiente. Objetivo: **Excelente en los 6**.

| # | Criterio | Pts | Excelente (qué hay que evidenciar) | Dónde se cubre |
| --- | --- | ---: | --- | --- |
| 1 | Implementación técnica del modelo | 4 | Modelo correcto; arquitectura, hiperparámetros y configuración de entrenamiento **documentados y reproducibles** | Informe §4.1–4.3; `src/models.py`, `src/train.py`, `main.py` |
| 2 | Calidad del entrenamiento | 3 | Entrenamiento ejecutado; **curvas de aprendizaje coherentes**; proceso controlado (EarlyStopping, 2 fases) | Informe §4.4–4.6; `reports/figures/learning_curves_*.png` |
| 3 | Evaluación mediante métricas | 4 | Accuracy, Precision, Recall, **F1**, **matriz de confusión** e **interpretación** (no solo pegar números) | Informe §5; `models_comparison.csv`; `confusion_matrix_*.png`; ROC |
| 4 | Desempeño en pruebas reales | 3 | Clasificar **imágenes nuevas** con alta precisión y **evidencia** (tabla, fotos, consenso de modelos) | Informe §6; `data/external_test/`; `external_test_comparison.*`; `app.py` |
| 5 | Análisis crítico y mejora | 2 | Errores, **causas** (sombras, juntas, textura), mejoras **viables** | Informe §7 |
| 6 | Integración del modelo con el sistema | 2 | Explicar **cómo** la IA se integra al prototipo y demostrar **comunicación funcional** (CNN → \(P_{\mathrm{ML}}\) → SE → dictamen) | Informe §8; `app.py` + capturas móvil |

**Total rúbrica: 18 pts.**  
El informe de titulación (secciones 1–10) es el vehículo; la rúbrica es el criterio de nota de implementación/evaluación.

---

## 7. Estructura formal del informe técnico

1. Introducción y justificación (1.1 contexto, 1.2 híbrido, 1.3 objetivos, 1.4 alcance).  
2. Auditoría, EDA y procesamiento del dataset (origen, no leakage, cuantitativo, EDA, ruido, pipeline, 70/15/15).  
3. Arquitectura y diseño (flujo, capas, comparativa de modelos, SE ACI/NEC, contrato JSON).  
4. Implementación y calidad del entrenamiento **[rúbrica 1 y 2]**.  
5. Evaluación por métricas **[rúbrica 3]**.  
6. Pruebas reales y casos de estudio **[rúbrica 4]**.  
7. Análisis crítico de errores y mejora **[rúbrica 5]**.  
8. Integración y prototipo web **[rúbrica 6]**.  
9. Conclusiones y recomendaciones.  
10. Referencias (ACI/NEC, ML, SE).  
Anexos (código, arquitectura).

**Estado del informe (17 ago 2026):** §1–3 y §8 en el Word del repo: [`docs/Proyecto_SistemaExperto_HernandezAxel.docx`](Proyecto_SistemaExperto_HernandezAxel.docx). **§4–7 y §9 se pueden redactar:** corrida 8k y OOD 77 fotos ya están en `reports/`. Checklist: [`INFORME_CONTINUACION.md`](INFORME_CONTINUACION.md).  
**No vender F1 Kaggle ≈ 1,0 como problema resuelto.** Discriminación de modelos: OOD + latencia.

---

## 8. Estado del software y siguiente paso

| Ítem | Estado |
| --- | --- |
| Esqueleto, 4 arquitecturas, `main.py`, reportes | Hecho |
| Corrida 4k + informe en `TRAINING_RESULTS.md` | Hecho |
| Ampliación a 8k en `data_loader` | Hecho (código) |
| Archivo automático de figuras previas | Hecho (`src/archive.py`) |
| Entrenamiento 8k | **Hecho** `2026-08-17 02:22:23`. F1 Kaggle: MN 1,0 · EN 0,9992 · CNN 0,9983 · RN 0,9983 |
| OOD modelos 8k | **Hecho** `2026-08-17 02:22:55`, n=77 (30 Pos / 47 Neg). F1: CNN **0,467** · MN 0,278 · EN 0,253 · RN 0,063. Recall CNN 0,700 |
| OOD piloto 4k (delta) | 16-ago 19:08, n=71 (24/47). CNN 0,420 · MN 0,294 · RN 0,140 · EN 0,121 |
| Informe §4–7 y §9 | **Siguiente chat:** redacción con cifras 8k. No reentrenar |
| `test_external.py` comparativo + anexar corridas | Hecho |
| Motor experto + `patron_orientacion` + MYCIN | Hecho |
| Orientación OpenCV (`src/crack_geometry.py`) + mapeo al SE | Hecho |
| App flujo simple (foto + elemento + ambiente + 9 cartas de campo) | Hecho; pesos **`cnn_custom.keras`** (OOD + latencia) |
| Acceso LAN / celular (`run_app.py`, config Streamlit) | Hecho (firewall Windows a veces pide admin; WiFi de campus puede aislar clientes) |
| Bitácora de visitas `data/inspections/` (`src/inspections.py`) | Hecho |
| Capturas móvil + evidencias para informe §8 | Hecho (en el Word, 16 ago noche) |

**Prototipo de campo (cómo se usa)**

1. Abrir o crear **visita** (nombre del lugar; la hora la pone el sistema).  
2. Cada foto: imagen + elemento + ambiente de **esa** foto.  
3. Generar: CNN + geometría; si hay fisura, **9 cartas** (Sí/No/No lo sé). Luego el SE; queda JPG + `visit.json`.  
4. Resumen de la visita (conteo por severidad). Reabrir visitas anteriores en el mismo PC.

**Números OOD vigentes (modelos 8k; no inventar otros):**

| Modelo | Acc | Prec | Rec | F1 OOD |
| --- | ---: | ---: | ---: | ---: |
| CNN personalizada | 0.377 | 0.350 | 0.700 | **0.467** |
| MobileNetV2 | 0.325 | 0.238 | 0.333 | 0.278 |
| EfficientNet-B0 | 0.234 | 0.204 | 0.333 | 0.253 |
| ResNet50V2 | 0.234 | 0.061 | 0.067 | 0.063 |

Fuente: `reports/external_test_comparison.md` corrida `2026-08-17 02:22:55`, n=77 (30 Pos / 47 Neg). Accuracy OOD es engañosa; defender **F1 y recall**.  
Delta 4k (n=71, 24/47, `2026-08-16 19:08:31`): CNN 0.420 · MN 0.294 · RN 0.140 · EN 0.121. El set OOD creció 6 Positive; no atribuir todo el cambio al 8k.

**Para redactar el informe (siguiente chat)**

1. Cifras Kaggle 8k: `reports/models_comparison.csv` y `TRAINING_RESULTS.md` Corrida 02.  
2. OOD: última sección de `external_test_comparison.md` (n=77).  
3. Redactar §4, 5, 6, 7, 9 con valores **exactos**.  
4. §8 ya tiene capturas; una frase: CSV elige MobileNet; en campo la CNN tiene más recall.  
5. Criterio 5: no vender F1=1,0 de Kaggle; el cuento es *domain shift*.

---

## 9. Principios de trabajo (para no perder calidad)

- No ejecutar entrenamientos largos si el usuario pide **solo editar código**.  
- Cada cambio relevante: commit claro (si el usuario lo pide o si es la regla del repo).  
- No pisar historia de métricas: archive + bitácoras acumulativas.  
- Números del informe = archivos en `reports/`, no memoria del modelo.  
- El dictamen experto es **alerta de protocolo**, no certificado de estabilidad.

---

*Última consolidación: 17 de agosto de 2026 (mañana). Corrida 8k cerrada. Siguiente paso: informe §4–7 y §9.*
