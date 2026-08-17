# CrackExpert AI

![PUCE](https://img.shields.io/badge/PUCE-Sede%20Manabí-0B3D91?style=for-the-badge)
![Sistemas Expertos](https://img.shields.io/badge/Asignatura-Sistemas%20Expertos-2E7D32?style=for-the-badge)
![Enfoque](https://img.shields.io/badge/Enfoque-Híbrido%20CNN%20+%20SE-6A1B9A?style=for-the-badge)
![Normativa](https://img.shields.io/badge/Normas-ACI%20224R%20%7C%20ACI%20318%20%7C%20NEC--SE--HM-C62828?style=for-the-badge)

**Sistema híbrido de visión por computador y sistema experto para la detección visual y la evaluación patológica de fisuras en elementos de hormigón armado.**

Pontificia Universidad Católica del Ecuador — Sede Manabí · Asignatura de Sistemas Expertos.

Memoria del proyecto (decisiones, rúbrica, estado): [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md)  
Base de conocimiento del experto: [`docs/EXPERT_SYSTEM_SPEC.md`](docs/EXPERT_SYSTEM_SPEC.md)

---

## Estado actual (agosto 2026)

| Frente | Qué hay |
| --- | --- |
| **ML — corrida 4.000 imgs** | Completa. Test retenido \(n=600\): F1 ≈ 1,0 en tres modelos; CNN custom elegida por empate en F1 y menor latencia (~6,4 ms, 1,3 MB). Figuras y CSV en `reports/` y snapshot en `reports/archive/2026-08-16_083631/`. |
| **ML — 8.000 imgs** | Código listo (4.000/4.000, split 5.600 / 1.200 / 1.200, `seed=42`). Pendiente de noche: `python main.py` (archiva la corrida 4k y no reanuda épocas si se corta). |
| **OOD (fotos de casa / celular)** | 71 fotos etiquetadas (24 Positive / 47 Negative). En Kaggle saturan; en campo **no**. Mejor F1 OOD: CNN custom **0,42** (recall 0,71). ResNet/EfficientNet caen fuerte. Tabla: `reports/external_test_comparison.md`. |
| **Sistema experto** | ACI 224R / ACI 318 / NEC-SE-HM + CF MYCIN. Patrón (OpenCV) + **9 preguntas de campo**; «No lo sé» no inventa evidencia. Dictamen llano (titular + qué hacer) según esas respuestas. |
| **Prototipo** | `app.py` / `python run_app.py`: visita → foto + elemento + ambiente → CNN; si hay fisura, **cartas apiladas** (Sí / No / No lo sé) → dictamen. Bitácora en `data/inspections/`. |
| **Informe** | Caps. 1–3 y 8 (capturas) en el Word. Caps. 4–7 y 9: tras la corrida 8k; guía `docs/INFORME_CONTINUACION.md`. Si el 8k no cierra, se puede redactar con 4k + OOD. |

**Lectura para la defensa:** un F1 de 1,0 en Kaggle no cierra el problema. El aporte experimental es el *domain shift* (benchmark vs celular) y la integración CNN → sistema experto.

---

## 1. Descripción general

Las fisuras en hormigón armado indican pérdida de **servicio**, **durabilidad** o, en casos graves, **integridad**. La inspección visual manual es lenta, subjetiva y no escala.

Una CNN solo responde *¿hay una grieta en la foto?* (\(P_{\mathrm{ML}}\)). Un sistema experto de normas no procesa píxeles. **CrackExpert AI** desacopla ambas capas:

| Capa | Responsabilidad |
| --- | --- |
| Percepción (4 CNN) | \(P(\text{fisura} \mid \text{imagen})\) |
| Geometría (OpenCV) | Orientación del trazo (vertical / horizontal / inclinada / malla) |
| Razonamiento (SE) | Elemento, ambiente, patrón, 9 observaciones de campo → severidad, titular, plan, CF MYCIN |

El dictamen **no sustituye** un peritaje estructural.

```text
Foto → 224×224 RGB → CNN → P(fisura)
                         +
         elemento, ambiente, patrón (OpenCV)
                         +
         9 cartas de campo (si P ≥ 0,50)
                         ↓
              Motor experto ACI/NEC + MYCIN
                         ↓
     Severidad | Titular llano | Qué hacer | CF
```

---

## 2. Estructura del repositorio

```text
crackexpert-ai/
├── data/
│   ├── raw/                 # Cache Kaggle (no versionar imágenes)
│   ├── processed/           # Splits train/val/test
│   ├── external_test/
│   │   ├── Positive/        # OOD: hay fisura
│   │   └── Negative/        # OOD: sana / negativos difíciles
│   └── inspections/         # Visitas locales (JSON + JPG; no versionado)
├── models/                  # *.keras (pesos grandes fuera de git)
├── reports/
│   ├── figures/             # Corrida vigente
│   ├── archive/             # Snapshots de corridas anteriores
│   ├── models_comparison.csv
│   ├── experiments_log.md
│   ├── external_test_comparison.csv / .md
│   └── TRAINING_RESULTS.md
├── src/
│   ├── data_loader.py       # kagglehub, 8k, splits, augmentation solo train
│   ├── models.py            # CNN custom, MobileNetV2, ResNet50V2, EfficientNet-B0
│   ├── train.py             # 2 fases, EarlyStopping, checkpoints
│   ├── evaluate.py          # Curvas, matrices, ROC, CSV, bitácora
│   ├── archive.py           # Copia figures/CSV antes de pisarlos
│   ├── expert_system.py     # Reglas + MYCIN + 9 preguntas de campo
│   ├── crack_geometry.py    # Orientación OpenCV
│   └── inspections.py       # Bitácora de visitas
├── docs/
│   ├── PROJECT_CONTEXT.md
│   ├── EXPERT_SYSTEM_SPEC.md
│   └── INFORME_CONTINUACION.md
├── main.py                  # Pipeline experimental end-to-end
├── test_external.py         # Evaluación OOD (anexa corridas)
├── app.py                   # Streamlit
├── run_app.py               # Streamlit en LAN (celular)
└── requirements.txt
```

---

## 3. Dataset y protocolo ML

**Fuente:** [Surface Crack Detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) (`arunrk7/surface-crack-detection`).

| Protocolo | Total | Train (70 %) | Val (15 %) | Test (15 %) | Estado |
| --- | ---: | ---: | ---: | ---: | --- |
| Corrida 1 | 4.000 (2k/2k) | 2.800 | 600 | 600 | **Hecha** |
| Corrida 2 | 8.000 (4k/4k) | 5.600 | 1.200 | 1.200 | Lanzar `python main.py`; no reanuda si se interrumpe |

Semilla fija `random_state=42`. Augmentation **solo train** (Flip, Rotation 0,1, Zoom 0,1, Brightness 0,1). Val y test: resize 224×224 RGB, sin aumento.

**Cuatro arquitecturas** (Adam, `binary_crossentropy`, EarlyStopping `val_loss` patience 5):

| Modelo | Estrategia | Fine-tuning fase 2 |
| --- | --- | --- |
| CNN personalizada | Desde cero, 3 bloques Conv+BN+Pool+Dropout | Red completa, LR \(10^{-5}\) |
| MobileNetV2 | ImageNet, fase 1 congelada LR \(10^{-3}\) | 25 capas |
| ResNet50V2 | Idem | 20 capas |
| EfficientNet-B0 | Idem | 20 capas |

**Selección de modelo:** F1 y AUC en test Kaggle **más** F1/recall OOD, latencia y tamaño. No basarse solo en el benchmark saturado.

---

## 4. Resultados vigentes

Detalle y figuras: [`reports/TRAINING_RESULTS.md`](reports/TRAINING_RESULTS.md).

### 4.1. Test Kaggle (corrida 4k, \(n=600\))

| Modelo | Test Acc | F1 | ROC-AUC | Latencia (ms) | Tamaño (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN personalizada | 1,000 | 1,000 | 1,000 | 6,4 | 1,3 |
| MobileNetV2 | 0,997 | 0,997 | 1,000 | 11,1 | 19,5 |
| ResNet50V2 | 1,000 | 1,000 | 1,000 | 22,3 | 150,6 |
| EfficientNet-B0 | 1,000 | 1,000 | 1,000 | 14,0 | 26,5 |

### 4.2. Fotos reales OOD (16 ago 2026, \(n=71\))

| Modelo | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| CNN personalizada | 0,338 | 0,298 | 0,708 | **0,420** |
| MobileNetV2 | 0,324 | 0,227 | 0,417 | 0,294 |
| ResNet50V2 | 0,310 | 0,121 | 0,167 | 0,140 |
| EfficientNet-B0 | 0,183 | 0,095 | 0,167 | 0,121 |

Accuracy OOD es engañosa (24 fisura / 47 sanas). En el informe se defienden **F1 y recall** y el contraste con Kaggle.

---

## 5. Instalación y comandos

Python 3.10+. Kagglehub requiere token (`~/.kaggle/kaggle.json`).

```powershell
cd CrackExpertAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

| Comando | Efecto |
| --- | --- |
| `python main.py` | Archiva corrida previa → datos 8k → entrena 4 modelos → métricas → OOD |
| `python test_external.py` | Evalúa `data/external_test/Positive` y `Negative`; **anexa** CSV/MD |
| `python run_app.py` | Streamlit en LAN (`http://<IP>:8501`) para el celular |
| `streamlit run app.py` | Mismo prototipo, solo localhost |

Fotos OOD: JPEG en `Positive/` (fisura) y `Negative/` (sana o trampa: junta, mancha, textura).

---

## 6. Sistema experto y prototipo de campo

| Entrada | Origen |
| --- | --- |
| \(P_{\mathrm{ML}}\) | CNN |
| Patrón / orientación | OpenCV; el inspector puede corregirlo; «varias en red» fuerza malla |
| Elemento | Viga, columna, losa, muro |
| Ambiente | Interior seco (\(w_{\max}=0{,}41\) mm), exterior húmedo (0,30), marino (0,15 en código) |
| 9 cartas | Ubicación, una/malla, humedad, óxido/desprendimiento, antigüedad, temblor, carga arriba, si crece, si es pasante |

«No lo sé» **no dispara** la regla. Si la CNN no ve fisura, no se preguntan las 9.

**Flujo en la app:** visita (lugar) → foto (queda en sesión al elegirla; JPG/PNG en celular) → elemento y ambiente → Generar dictamen → cartas apiladas si hay fisura → titular + qué hacer + bitácora.

Salidas: Leve / Moderada / Crítica, texto llano según las respuestas, plan de acción, mecanismo técnico, cita normativa, \(\mathrm{CF}_{\mathrm{comb}}\).

---

## 7. Alcance y limitaciones

- Kaggle etiqueta **presencia/ausencia**, no mecanismo ni ancho en mm.  
- F1 ≈ 1,0 en ese dominio no implica generalización a celular.  
- OpenCV estima orientación 2D; no detecta helicoidal.  
- El SE es **alerta de protocolo**, no certificado de estabilidad.  
- Las visitas viven solo en este PC (`data/inspections/`).  
- Fotos HEIC de iPhone pueden no abrirse; usar JPG o PNG.

## 8. Uso académico

Proyecto formativo, Sistemas Expertos, PUCE Sede Manabí. ACI y NEC se citan como referencia; su aplicación profesional exige el texto oficial y un ingeniero civil responsable.
