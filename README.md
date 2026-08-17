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
| **ML — corrida 4.000 imgs** | Completa. Test \(n=600\). Snapshot: `reports/archive/2026-08-16_083631/` (y copia previa al 8k: `..._215127/`). |
| **ML — corrida 8.000 imgs** | **Hecha** (`2026-08-17 02:22:23`). Test \(n=1.200\). Kaggle saturado (F1 ≥ 0,998). CSV elige **MobileNetV2** (F1 1,0). CNN: 2 FN, 6,5 ms. Figuras vigentes: `reports/figures/`. Relato: `reports/TRAINING_RESULTS.md` Corrida 02. |
| **OOD (fotos de casa / celular)** | Modelos 8k, corrida `2026-08-17 02:22:55`: **n=77** (30 Pos / 47 Neg). Gana **CNN** F1 **0,467** (recall 0,70). MobileNet F1 0,278. ResNet 0,063. La corrida 4k era n=71 (24/47); no mezclar F1 como si el set fuera el mismo. |
| **Sistema experto** | ACI 224R / ACI 318 / NEC-SE-HM + CF MYCIN. Patrón (OpenCV) + **9 preguntas de campo**; «No lo sé» no inventa evidencia. |
| **Prototipo** | `app.py` / `python run_app.py`. Pesos por defecto: **`cnn_custom.keras`** (mejor F1/recall OOD y menor latencia). El CSV de Kaggle sigue eligiendo MobileNetV2; no se usa ese criterio en la app. |
| **Informe** | Caps. 1–3 y 8 en el Word. Caps. 4–7 y 9: redactar ahora con 8k + OOD. Guía: `docs/INFORME_CONTINUACION.md`. |

**Lectura para la defensa:** un F1 de 1,0 en Kaggle no cierra el problema. El aporte es el *domain shift* (MobileNet gana en campus y pierde en celular) y la integración CNN → sistema experto.

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
├── models/                  # cnn_custom, MobileNet, EfficientNet en git; ResNet no (150 MB)
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
| Corrida 2 | 8.000 (4k/4k) | 5.600 | 1.200 | 1.200 | **Hecha** (`python main.py`, 17 ago 2026) |

Semilla fija `random_state=42`. Augmentation **solo train** (Flip, Rotation 0,1, Zoom 0,1, Brightness 0,1). Val y test: resize 224×224 RGB, sin aumento.

**Cuatro arquitecturas** (Adam, `binary_crossentropy`, EarlyStopping `val_loss` patience 5):

| Modelo | Estrategia | Fine-tuning fase 2 |
| --- | --- | --- |
| CNN personalizada | Desde cero, 3 bloques Conv+BN+Pool+Dropout | Red completa, LR \(10^{-5}\) |
| MobileNetV2 | ImageNet, fase 1 congelada LR \(10^{-3}\) | 25 capas |
| ResNet50V2 | Idem | 20 capas |
| EfficientNet-B0 | Idem | 20 capas |

**Selección de modelo:** F1 y AUC en test Kaggle **más** F1/recall OOD, latencia y tamaño. No basarse solo en el benchmark saturado. **La app carga `models/cnn_custom.keras`.** El selector de `evaluate.py` (solo F1 Kaggle) queda como bitácora experimental.

---

## 4. Resultados vigentes

Detalle, curvas y lectura: [`reports/TRAINING_RESULTS.md`](reports/TRAINING_RESULTS.md) (Corrida 01 = 4k; Corrida 02 = 8k).

### 4.1. Test Kaggle — corrida 8k (\(n=1.200\)), 17 ago 2026

Fuente: `reports/models_comparison.csv`.

| Modelo | Test Acc | Precision | Recall | F1 | ROC-AUC | Latencia (ms) | Tamaño (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN personalizada | 0,9983 | 1,0000 | 0,9967 | 0,9983 | 1,0000 | **6,5** | **1,3** |
| MobileNetV2 | **1,0000** | 1,0000 | **1,0000** | **1,0000** | 1,0000 | 11,3 | 19,5 |
| ResNet50V2 | 0,9983 | 0,9967 | 1,0000 | 0,9983 | 1,0000 | 22,7 | 150,6 |
| EfficientNet-B0 | 0,9992 | 0,9983 | 1,0000 | 0,9992 | 1,0000 | 14,1 | 26,5 |

El CSV elige MobileNetV2 (único F1=1,0). CNN: 2 FN. ResNet: 2 FP. EfficientNet: 1 FP. AUC=1,0 en los cuatro = **saturación del dominio Kaggle**, no “problema resuelto”.

Piloto 4k (\(n=600\)): CNN/ResNet/EN F1=1,0; MobileNet 0,9967 (2 FP). Ver Corrida 01.

### 4.2. Fotos reales OOD — modelos 8k (17 ago 2026, \(n=77\))

30 fisura / 47 sanas. Fuente: `reports/external_test_comparison.md` corrida `2026-08-17 02:22:55`.

| Modelo | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| CNN personalizada | 0,377 | 0,350 | **0,700** | **0,467** |
| MobileNetV2 | 0,325 | 0,238 | 0,333 | 0,278 |
| EfficientNet-B0 | 0,234 | 0,204 | 0,333 | 0,253 |
| ResNet50V2 | 0,234 | 0,061 | 0,067 | 0,063 |

En campo gana la **CNN** (más recall). MobileNet, primero en Kaggle, se deja dos tercios de las fisuras reales. Por eso el prototipo usa **`cnn_custom.keras`**. Accuracy OOD es engañosa por el desbalance.

OOD con pesos 4k (\(n=71\), 24/47): CNN F1 0,420 · MN 0,294 · RN 0,140 · EN 0,121. El set creció en 6 Positive; no atribuir todo el delta al 8k.

---

## 5. Instalación y comandos

Python 3.10–3.12. Kagglehub requiere token (`~/.kaggle/kaggle.json`) solo para entrenar.

Los pesos livianos van en el repo (`cnn_custom.keras` para la app; MobileNet y EfficientNet para comparar). `resnet50_v2.keras` (~150 MB) no se versiona; cópialo a mano si lo necesitas.

```powershell
cd CrackExpertAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En macOS:

```bash
cd CrackExpertAI
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python run_app.py
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
