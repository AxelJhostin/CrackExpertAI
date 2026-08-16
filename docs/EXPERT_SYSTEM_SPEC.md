# Especificación del sistema experto — CrackExpert AI

**Documento técnico de la base de conocimiento y del motor de inferencia**  
Pontificia Universidad Católica del Ecuador — Sede Manabí · Sistemas Expertos  
Normas de referencia: **ACI 224R-01**, **ACI 318**, **NEC-SE-HM** (Norma Ecuatoriana de la Construcción — Hormigón Armado).

Este documento define el conocimiento **declarativo** (hechos, umbrales, reglas de producción) y el conocimiento **procedimental** (encadenamiento hacia adelante y combinación de factores de certeza estilo MYCIN). Complementa el README del repositorio y rige el diseño de `src/expert_system.py`.

---

## 1. Fundamentación normativa

### 1.1. Rol de cada fuente

| Fuente | Aporte al sistema experto |
| --- | --- |
| **ACI 224R-01** *Control of Cracking in Concrete Structures* | Tabla 4.1: anchos máximos de fisura de **flexión** según condición de exposición, orientados a **durabilidad** y apariencia, no a capacidad última. |
| **ACI 318** *Building Code Requirements for Structural Concrete* | Distinción de mecanismos: fisuración por **flexión** (dúctil, zona traccionada) frente a **cortante** (potencialmente frágil). El código evolucionó desde límites explícitos de ancho hacia control por recubrimiento y cuantía; el sistema experto usa ACI 318 para **clasificar el patrón**, no para sustituir el cálculo de \(V_n\) o \(\phi V_c\). |
| **NEC-SE-HM** | Adopta el marco de hormigón armado para Ecuador: requisitos de **durabilidad**, recubrimiento, exposición y servicio. En este proyecto se alinea con los límites de ACI 224R y se enfatiza el ambiente **marino / agresivo** del litoral manabita. |

### 1.2. Criterios de tolerancia de ancho (ACI 224R-01, Tabla 4.1)

Ancho máximo razonable de fisura de flexión, \(w_{\max}\), en milímetros:

| Condición de exposición | \(w_{\max}\) (mm) | Mapeo de interfaz CrackExpert |
| --- | ---: | --- |
| Aire seco o membrana protectora | 0,41 | Interior seco |
| Humedad, aire húmedo, suelo | 0,30 | Exterior húmedo |
| Productos químicos de deshielo | 0,18 | Marino / agresivo (subcaso sales) |
| Agua de mar, mojado-secado | 0,15 | Marino / agresivo (defecto) |
| Estructuras retenedoras de agua | 0,10 | Marino / agresivo (subcaso depósitos / tanques) |

**Interpretación operacional.** Superar \(w_{\max}\) **no** demuestra colapso; indica incumplimiento del **estado límite de servicio / durabilidad**. Un ancho \(\geq 1{,}0\,\mathrm{mm}\) se trata como fisura grosera y se escala a severidad crítica **hasta** inspección profesional, por el riesgo de sección efectiva reducida, corrosión avanzada o mecanismo no flexional.

### 1.3. Durabilidad y recubrimiento (ACI 318 / NEC-SE-HM)

La fisura es un camino de transporte de agentes agresivos. El motor experto **incrementa** la severidad cuando:

- el ambiente es marino o con cloruros (NEC-SE-HM: exposición agresiva);
- hay indicios de corrosión (manchas, fisuras paralelas a la armadura, desprendimiento);
- la fisura es **pasante** (losa o muro): filtración y lixiviación.

### 1.4. Mecanismos estructurales (ACI 318)

| Patrón | Lectura normativa | Tratamiento en el SE |
| --- | --- | --- |
| Flexión | Fisuras aproximadamente perpendiculares al acero de tracción, zona de momento positivo/negativo | Severidad gobernada principalmente por \(w\) vs \(w_{\max}\) |
| Cortante | Fisuras inclinadas en almas de vigas o nudos; mecanismo potencialmente frágil | Escalado inmediato a **crítica / estructural**, con CF alto |
| Mapa / caimán | Retracción, gradiente térmico o RAS | Durabilidad / material; no se confunde con cortante |
| Corrosión | Fisuras siguiendo barras, manchas de óxido, recubrimiento suelto | Durabilidad elevada; estructural si hay *spalling* |

**Descargo.** El sistema no calcula capacidad residual ni factores \(\phi\). Cualquier conclusión de “crítica / estructural” es una **alerta de protocolo**, no un certificado de inestabilidad.

---

## 2. Variables del dominio

### 2.1. Entradas

| Símbolo | Nombre | Tipo | Dominio | Descripción |
| --- | --- | --- | --- | --- |
| \(P\) | Probabilidad de fisura (ML) | continuo | \([0, 1]\) | Salida de la capa `sigmoid` del clasificador. |
| \(w\) | Ancho estimado | continuo / nulo | \(\mathrm{mm}\) o desconocido | Medición preferente con fisurómetro; si falta, se dispara la regla de evidencia incompleta. |
| \(E\) | Tipo de elemento | enumerado | Viga, Columna, Losa, Muro | Condiciona el peso de cortante, pandeo/hendimiento y filtración. |
| \(A\) | Ambiente de exposición | enumerado | Interior seco; Exterior húmedo; Marino/agresivo | Selecciona \(w_{\max}\) según §1.2. |
| \(Q_w\) | Calidad de la medición de ancho | continuo | \([0, 1]\) | 1,0 si hay lectura instrumental; 0,5 si es estimación visual; 0,0 si \(w\) es nulo. |
| \(\pi\) | Patrón (opcional) | enumerado | desconocido, flexión, cortante, mapa, corrosión, asentamiento, térmica | Si el inspector no lo indica, las reglas de patrón no se disparan (salvo cortante explícito). |

**Umbral de percepción.** Se considera evidencia positiva de fisura si \(P \geq 0{,}50\). El CF asociado a esa evidencia no es binario: se calcula como se indica en §4.3.

**Función \(w_{\max}(A)\):**

\[
w_{\max}(A) =
\begin{cases}
0{,}41 & \text{si } A = \text{Interior seco} \\
0{,}30 & \text{si } A = \text{Exterior húmedo} \\
0{,}15 & \text{si } A = \text{Marino/agresivo}
\end{cases}
\]

### 2.2. Salidas

| Símbolo | Nombre | Dominio | Semántica |
| --- | --- | --- | --- |
| \(S\) | Nivel de severidad | Leve / estética; Moderada / durabilidad; Crítica / estructural | Hipótesis de mayor rango entre las reglas disparadas. |
| \(\mathrm{CF}_{\mathrm{comb}}\) | Factor de certeza combinado | \([-1, 1]\) | Confianza neta en \(S\) tras MYCIN. |
| \(M\) | Medidas de mitigación y reparación | lista de acciones | Plan priorizado, una o más por regla. |
| \(\mathcal{R}\) | Trace | identificadores \(R_i\) | Explicabilidad: por qué se concluyó \(S\). |

**Correspondencia de severidad** (tres niveles de interfaz; el código puede conservar estados internos `NONE` y `CRITICAL` como extremos de “crítica”):

| Nivel de informe | Significado de servicio | Estados internos típicos |
| --- | --- | --- |
| Leve / estética | Fisura dentro de \(w_{\max}\); impacto visual | `AESTHETIC` |
| Moderada / durabilidad | Riesgo de corrosión, filtración o \(w > w_{\max}\) | `SERVICEABILITY` |
| Crítica / estructural | Cortante, *spalling*, \(w \geq 1\,\mathrm{mm}\), alerta de estabilidad | `STRUCTURAL`, `CRITICAL` |
| Sin fisura | \(P < 0{,}50\) | `NONE` |

---

## 3. Matriz de decisión (visión de conjunto)

Filas: combinación \(A\) × relación de \(w\) con \(w_{\max}\) (cuando \(P \geq 0{,}50\) y el patrón no es cortante). Columnas: tipo de elemento. Celda: severidad **base** (el cortante y el *spalling* la sobrescriben).

| Ambiente \(A\) | Ancho | Viga | Columna | Losa | Muro |
| --- | --- | --- | --- | --- | --- |
| Interior seco | \(w \leq 0{,}41\,\mathrm{mm}\) | Leve | Leve | Leve | Leve |
| Interior seco | \(0{,}41 < w < 1{,}0\) | Moderada | Moderada | Moderada | Moderada |
| Exterior húmedo | \(w \leq 0{,}30\,\mathrm{mm}\) | Leve | Leve | Leve\* | Leve\* |
| Exterior húmedo | \(0{,}30 < w < 1{,}0\) | Moderada | Moderada | Moderada | Moderada |
| Marino / agresivo | \(w \leq 0{,}15\,\mathrm{mm}\) | Moderada\*\* | Moderada\*\* | Moderada | Moderada |
| Marino / agresivo | \(0{,}15 < w < 1{,}0\) | Moderada | Moderada | Crítica† | Crítica† |
| Cualquiera | \(w \geq 1{,}0\,\mathrm{mm}\) | Crítica | Crítica | Crítica | Crítica |
| Cualquiera | Patrón cortante | Crítica | Crítica | Crítica | Crítica |
| Cualquiera | \(w\) desconocido | Leve (incompleto) | Leve (incompleto) | Leve (incompleto) | Leve (incompleto) |

\* En losa y muro húmedos, una fisura **pasante** escala a Moderada aunque \(w \leq w_{\max}\).  
\*\* En ambiente marino, incluso anchos “tolerables” se tratan como Moderada por durabilidad (cloruros), no como meramente estéticos.  
† Filtración + cloruros: se recomienda tratamiento de **crítica de durabilidad / uso**, agrupado en el nivel crítico de la interfaz.

---

## 4. Motor de inferencia y certeza estilo MYCIN

### 4.1. Estrategia

- **Encadenamiento hacia adelante** (data-driven): se evalúan todas las reglas cuyas premisas son satisfacibles; no hay backtracking.
- **Conflicto:** la severidad de salida es el **máximo rango** entre hipótesis confirmadas (crítica \(>\) moderada \(>\) leve \(>\) sin fisura).
- **Explicación:** se registran las reglas disparadas y el CF parcial de cada una.

### 4.2. Factor de certeza de una regla

Una regla \(R_i\) tiene un **CF base** \(\mathrm{CF}(R_i) \in [-1, 1]\) (confianza del experto **si** las premisas son verdaderas).

El CF de las premisas es el mínimo de las certezas de cada condición (conjunción clásica MYCIN):

\[
\mathrm{CF}(\text{premisas}_i) = \min_j \mathrm{CF}(c_{i,j})
\]

El CF **aportado** a la conclusión:

\[
\mathrm{CF}_{\text{aportar}}(R_i) = \mathrm{CF}(R_i) \cdot \mathrm{CF}(\text{premisas}_i)
\]

### 4.3. Certeza de las condiciones atómicas

| Condición | \(\mathrm{CF}(c)\) |
| --- | --- |
| Fisura detectada por ML | \(\mathrm{CF}_P = 2P - 1\)  (mapea \(P=0 \rightarrow -1\), \(P=0{,}5 \rightarrow 0\), \(P=1 \rightarrow +1\)) |
| Ancho medido y comparable con \(w_{\max}\) | \(\mathrm{CF}_w = Q_w\) |
| Tipo de elemento declarado | \(1{,}0\) (hecho de inspección) |
| Ambiente declarado | \(1{,}0\) (hecho de inspección) |
| Patrón declarado por el inspector | \(0{,}85\) (observación subjetiva) |
| Patrón desconocido | la regla de patrón **no** se dispara |

Para reglas que exigen \(P \geq 0{,}50\), si \(\mathrm{CF}_P \leq 0\) la regla de patología **no** se dispara; se evalúa \(R0\).

### 4.4. Combinación de dos CF (MYCIN)

Sean \(\mathrm{CF}_1, \mathrm{CF}_2 \in [-1, 1]\) certezas sobre la **misma** hipótesis \(H\) (p. ej. “severidad moderada”).

\[
\mathrm{CF}_{\mathrm{comb}}(\mathrm{CF}_1, \mathrm{CF}_2) =
\begin{cases}
\mathrm{CF}_1 + \mathrm{CF}_2\,(1 - \mathrm{CF}_1) & \text{si } \mathrm{CF}_1 > 0 \land \mathrm{CF}_2 > 0 \\[6pt]
\mathrm{CF}_1 + \mathrm{CF}_2\,(1 + \mathrm{CF}_1) & \text{si } \mathrm{CF}_1 < 0 \land \mathrm{CF}_2 < 0 \\[6pt]
\dfrac{\mathrm{CF}_1 + \mathrm{CF}_2}{1 - \min\bigl(|\mathrm{CF}_1|, |\mathrm{CF}_2|\bigr)} & \text{si } \mathrm{CF}_1 \cdot \mathrm{CF}_2 < 0
\end{cases}
\]

La combinación es **asociativa** en el uso práctico: se reduce la lista de \(\mathrm{CF}_{\text{aportar}}\) de las reglas que votan por la hipótesis ganadora \(S\).

El \(\mathrm{CF}_{\mathrm{comb}}\) reportado en la interfaz es el de la hipótesis \(S\) seleccionada. Si ninguna regla de patología se dispara, se reporta \(\mathrm{CF}_{\mathrm{comb}} = -\mathrm{CF}_P\) sobre la hipótesis “sin fisura” cuando \(P < 0{,}50\) (certeza de ausencia acotada por el clasificador).

### 4.5. Ejemplo numérico

Supóngase \(P = 0{,}92\) \(\Rightarrow \mathrm{CF}_P = 0{,}84\), \(Q_w = 1\), \(A =\) exterior húmedo, \(w = 0{,}45\,\mathrm{mm} > 0{,}30\), elemento = losa.

- \(R1\) (fisura presente), \(\mathrm{CF}(R1)=0{,}90\): \(\mathrm{CF}_{\text{aportar}} = 0{,}90 \times 0{,}84 = 0{,}756\) (hipótesis leve mínima).
- \(R7\) (\(w > w_{\max}\)), \(\mathrm{CF}(R7)=0{,}85\): \(\mathrm{CF}_{\text{aportar}} = 0{,}85 \times \min(0{,}84, 1{,}0) = 0{,}714\) (hipótesis moderada).

Hipótesis ganadora: **Moderada**. Un solo votante dominante \(\Rightarrow \mathrm{CF}_{\mathrm{comb}} = 0{,}714\). Si otra regla de durabilidad en losa húmeda aportara \(0{,}40\):

\[
\mathrm{CF}_{\mathrm{comb}} = 0{,}714 + 0{,}40\,(1-0{,}714) = 0{,}828
\]

---

## 5. Reglas de producción

Formato:

> **SI** \(\langle\)condiciones\(\rangle\) **ENTONCES** \(\langle\)conclusión\(\rangle\)  
> **CF base** = \(\mathrm{CF}(R_i)\)  
> **Acciones** = medidas de mitigación asociadas.

Las condiciones implícitas “hecho declarado” tienen \(\mathrm{CF}=1\) salvo indicación contraria.

### 5.1. Percepción y evidencia incompleta

**R0 — Ausencia visual**  
**SI** \(P < 0{,}50\)  
**ENTONCES** \(S \leftarrow\) Sin fisura; no se evalúa ancho normativo.  
**CF base** = \(0{,}80\)  
**Acciones:** inspección rutinaria según plan de mantenimiento.

**R1 — Detección positiva**  
**SI** \(P \geq 0{,}50\)  
**ENTONCES** existe fisura; \(S\) inicial \(\leftarrow\) Leve / estética.  
**CF base** = \(0{,}90\)  
**Acciones:** fotografiar con escala; registrar localización en el elemento.

**R6b — Ancho no medido**  
**SI** \(P \geq 0{,}50\) **Y** \(w\) es nulo  
**ENTONCES** evidencia incompleta; se mantiene \(S\) no inferior a Leve y se marca el dictamen como no contrastado con ACI 224R.  
**CF base** = \(0{,}55\)  
**Acciones:** medir ancho máximo con fisurómetro y reejecutar el motor.

### 5.2. Durabilidad por ancho y exposición (ACI 224R / NEC-SE-HM)

**R6 — Cómputo del límite**  
**SI** \(A\) es conocido  
**ENTONCES** asignar \(w_{\max}(A)\) según §2.1.  
**CF base** = \(1{,}00\) (definición normativa).

**R7 — Incumplimiento de servicio**  
**SI** \(P \geq 0{,}50\) **Y** \(w\) conocido **Y** \(w_{\max} < w < 1{,}0\,\mathrm{mm}\)  
**ENTONCES** \(S \leftarrow\) Moderada / durabilidad.  
**CF base** = \(0{,}85\)  
**Acciones:** testigos de evolución; sellado elástico; revisar recubrimiento y flecha.

**R8 — Fisura grosera**  
**SI** \(P \geq 0{,}50\) **Y** \(w \geq 1{,}0\,\mathrm{mm}\)  
**ENTONCES** \(S \leftarrow\) Crítica / estructural.  
**CF base** = \(0{,}88\)  
**Acciones:** delimitar zona; restringir uso según criterio del ingeniero responsable; evaluación estructural formal.

**R10 — Ambiente marino aun con \(w \leq w_{\max}\)**  
**SI** \(P \geq 0{,}50\) **Y** \(A =\) Marino/agresivo **Y** \(w\) conocido **Y** \(w \leq w_{\max}\)  
**ENTONCES** \(S \leftarrow\) Moderada / durabilidad (no meramente estética).  
**CF base** = \(0{,}70\)  
**Acciones:** verificar recubrimiento NEC-SE-HM; inspección de manchas de óxido; considerar recubrimientos de protección.

### 5.3. Mecanismos y síntomas (ACI 318)

**R2 — Patrón de cortante**  
**SI** \(P \geq 0{,}50\) **Y** \(\pi =\) cortante  
**ENTONCES** \(S \leftarrow\) Crítica / estructural.  
**CF base** = \(0{,}92\)  
**Acciones:** restringir cargas; evaluación inmediata por profesional calificado; no sellar como única medida.

**R3 — Corrosión de armadura**  
**SI** \(P \geq 0{,}50\) **Y** (\(\pi =\) corrosión **O** manchas de óxido)  
**ENTONCES** \(S \leftarrow\) al menos Moderada / durabilidad.  
**CF base** = \(0{,}80\)  
**Acciones:** carbonatación y cloruros; rehabilitación del recubrimiento.

**R4 — Desprendimiento (*spalling*)**  
**SI** \(P \geq 0{,}50\) **Y** existe *spalling*  
**ENTONCES** \(S \leftarrow\) Crítica / estructural.  
**CF base** = \(0{,}90\)  
**Acciones:** delimitar área; reparación del recubrimiento y evaluación de sección residual.

**R5 — Fisura pasante**  
**SI** \(P \geq 0{,}50\) **Y** fisura pasante **Y** \(E \in \{\)Losa, Muro\(\}\)  
**ENTONCES** \(S \leftarrow\) al menos Moderada / durabilidad.  
**CF base** = \(0{,}78\)  
**Acciones:** sellado; revisión de impermeabilización; control de filtraciones.

**R9 — Fisuración en mapa**  
**SI** \(P \geq 0{,}50\) **Y** \(\pi =\) mapa  
**ENTONCES** hipótesis de retracción / RAS; **no** clasificar como cortante.  
**CF base** = \(0{,}75\)  
**Acciones:** control de humedad; evaluar potencial de reacción álcali-agregado si hay expansión.

### 5.4. Especialización por tipo de elemento

**R11 — Viga, cortante implícito por localización (si el inspector marca “alma / 45°”)**  
**SI** \(P \geq 0{,}50\) **Y** \(E =\) Viga **Y** \(\pi =\) cortante  
**ENTONCES** idéntico a R2; se anota contexto de alma a cortante.  
**CF base** = \(0{,}92\)  
**Acciones:** las de R2; revisar estribos y nudos según ACI 318 / NEC-SE-HM.

**R12 — Columna, fisura vertical grosera**  
**SI** \(P \geq 0{,}50\) **Y** \(E =\) Columna **Y** \(w \geq 0{,}50\,\mathrm{mm}\)  
**ENTONCES** \(S \leftarrow\) al menos Moderada; si \(w \geq 1{,}0\,\mathrm{mm}\) aplica R8.  
**CF base** = \(0{,}72\)  
**Acciones:** revisar carga axial, esbeltez y posible hendimiento; no atribuir automáticamente a “retracción”.

**R13 — Losa, servicio y filtración**  
**SI** \(P \geq 0{,}50\) **Y** \(E =\) Losa **Y** \(A \neq\) Interior seco **Y** \(w > w_{\max}\)  
**ENTONCES** \(S \leftarrow\) Moderada, o Crítica si \(A =\) Marino/agresivo (matriz §3).  
**CF base** = \(0{,}80\)  
**Acciones:** impermeabilización; control de flecha; sellado de fisuras de flexión.

**R14 — Muro, mapa vs asentamiento**  
**SI** \(P \geq 0{,}50\) **Y** \(E =\) Muro **Y** \(\pi =\) asentamiento  
**ENTONCES** \(S \leftarrow\) al menos Moderada.  
**CF base** = \(0{,}77\)  
**Acciones:** topografía de fisuras (apertura y escalonamiento); geotecnia si hay desplazamiento relativo.

---

## 6. Plan de mitigación (catálogo)

Las acciones se **unen** (unión ordenada, sin duplicados) a partir de las reglas disparadas. Prioridad de presentación:

1. Seguridad inmediata (R2, R4, R8).  
2. Durabilidad y sellado (R3, R5, R7, R10, R13).  
3. Completar evidencia (R6b).  
4. Monitoreo (R0, R1, R9).

| Código | Medida |
| --- | --- |
| M1 | Inspección rutinaria / registro fotográfico con escala |
| M2 | Medición de ancho máximo (fisurómetro) |
| M3 | Testigos de yeso o instrumentación de apertura |
| M4 | Sellado elástico / puenteo de fisura de flexión |
| M5 | Impermeabilización y control de filtraciones |
| M6 | Ensayos de carbonatación, cloruros y recubrimiento |
| M7 | Reparación de recubrimiento y *spalling* |
| M8 | Restricción de cargas y evaluación estructural formal |
| M9 | Protección superficial en ambiente marino (NEC-SE-HM) |

---

## 7. Trazabilidad hacia la implementación

| Concepto de esta especificación | Módulo |
| --- | --- |
| \(w_{\max}(A)\), R0–R9 | `src/expert_system.py` (`aci_224r_width_limit`, `evaluate_pathology`) |
| \(P\) desde CNN | Inferencia del `.keras` seleccionado en `app.py` / pipeline de predicción |
| CF MYCIN y R10–R14 | Motor de certeza documentado aquí; debe mantenerse alineado en `expert_system.py` |
| Benchmark visual (no normativo) | `src/train.py`, `src/evaluate.py` |

**Principio de desacople:** modificar una regla ACI no requiere reentrenar la CNN; cambiar el backbone no modifica \(w_{\max}\).

---

## 8. Referencias

1. ACI Committee 224. *ACI 224R-01: Control of Cracking in Concrete Structures*. American Concrete Institute.  
2. ACI Committee 318. *Building Code Requirements for Structural Concrete (ACI 318)*. American Concrete Institute.  
3. MIDUVI. *NEC-SE-HM: Hormigón Armado*. Norma Ecuatoriana de la Construcción.  
4. Shortliffe, E. H.; Buchanan, B. G. A model of inexact reasoning in medicine. *Mathematical Biosciences*, 1975. (combinación de CF de MYCIN.)  
5. Özgenel, Ç. F.; et al. Surface Crack Detection dataset (`arunrk7/surface-crack-detection`), usado únicamente para la capa de percepción.

---

*Fin de la especificación. Cualquier cambio de umbral normativo debe versionarse en este documento y en la bitácora `reports/experiments_log.md` si altera dictámenes de demostración.*
