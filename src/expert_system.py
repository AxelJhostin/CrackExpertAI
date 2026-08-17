"""Motor de inferencia patológica (ACI 224R / ACI 318 / NEC-SE-HM) con CF MYCIN.

Combina P(fisura) de la CNN con variables de campo: elemento, ambiente,
ancho (mm) y patrón/orientación visual. No sustituye un peritaje in situ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReportSeverity(str, Enum):
    NONE = "Sin fisura"
    LEVE = "Leve"
    MODERADA = "Moderada"
    CRITICA = "Crítica"


class ElementType(str, Enum):
    VIGA = "Viga"
    COLUMNA = "Columna"
    LOSA = "Losa"
    MURO = "Muro"


class Exposure(str, Enum):
    INTERIOR_SECO = "Interior seco"
    EXTERIOR_HUMEDO = "Exterior húmedo"
    MARINO = "Marino / agresivo"


class PatronOrientacion(str, Enum):
    """Variable de campo `patron_orientacion` (observación visual)."""

    DIAGONAL_APOYOS = "Diagonal (~45° en apoyos)"
    VERTICAL_COLUMNA = "Vertical (paralela al eje en columna)"
    VERTICAL_VANO = "Vertical / Perpendicular (centro de vano en viga)"
    HORIZONTAL_COLUMNA = "Horizontal transversal (en columna)"
    HELICOIDAL = "Helicoidal / Espiral (45°)"
    MALLA = "Malla / Piel de cocodrilo"
    LONGITUDINAL_REFUERZO = "Longitudinal paralela al refuerzo"


ELEMENT_OPTIONS: tuple[str, ...] = tuple(e.value for e in ElementType)
EXPOSURE_OPTIONS: tuple[str, ...] = tuple(e.value for e in Exposure)
PATRON_ORIENTACION_OPTIONS: tuple[str, ...] = tuple(p.value for p in PatronOrientacion)

ML_CRACK_THRESHOLD: float = 0.50
GROSS_WIDTH_MM: float = 1.0
CF_PATTERN_OBS: float = 0.85
CF_FACT_DECLARED: float = 1.0
CF_FIELD_OBS: float = 0.80
UNKNOWN_ANSWER = "No lo sé"

FIELD_QUESTIONS: tuple[dict[str, object], ...] = (
    {
        "id": "ubicacion",
        "text": "¿Dónde está la grieta?",
        "options": ("En el apoyo", "En el medio", "En una junta", UNKNOWN_ANSWER),
    },
    {
        "id": "malla",
        "text": "¿Es una sola grieta o varias en red?",
        "options": ("Una sola", "Varias en red", UNKNOWN_ANSWER),
    },
    {
        "id": "humedad",
        "text": "¿Está húmeda, gotea o hay mancha de agua?",
        "options": ("Sí", "No", UNKNOWN_ANSWER),
    },
    {
        "id": "oxido_desprendimiento",
        "text": "¿Hay óxido o se cae el concreto?",
        "options": ("Óxido", "Se desprende el concreto", "Ambos", "No", UNKNOWN_ANSWER),
    },
    {
        "id": "antiguedad",
        "text": "¿Hace cuánto la notaste?",
        "options": ("Días", "Semanas o meses", "Años", UNKNOWN_ANSWER),
    },
    {
        "id": "sismo",
        "text": "¿Hubo un temblor reciente?",
        "options": ("Sí", "No", UNKNOWN_ANSWER),
    },
    {
        "id": "carga_arriba",
        "text": "¿Hay más pisos o carga arriba de este elemento?",
        "options": ("Sí", "No", UNKNOWN_ANSWER),
    },
    {
        "id": "evolucion",
        "text": "¿Crece o está igual?",
        "options": ("Crece", "Está igual", UNKNOWN_ANSWER),
    },
    {
        "id": "pasante",
        "text": "¿La grieta pasa al otro lado del elemento?",
        "options": ("Sí", "No", UNKNOWN_ANSWER),
    },
)

ACI_224R_LIMITS_MM: dict[Exposure, float] = {
    Exposure.INTERIOR_SECO: 0.41,
    Exposure.EXTERIOR_HUMEDO: 0.30,
    Exposure.MARINO: 0.15,
}

_SEVERITY_RANK: dict[ReportSeverity, int] = {
    ReportSeverity.NONE: 0,
    ReportSeverity.LEVE: 1,
    ReportSeverity.MODERADA: 2,
    ReportSeverity.CRITICA: 3,
}


@dataclass(frozen=True)
class CrackObservation:
    """Entrada al sistema experto (unidades SI: mm)."""

    ml_probability: float
    elemento: ElementType = ElementType.VIGA
    ambiente: Exposure = Exposure.INTERIOR_SECO
    ancho_mm: float | None = None
    patron_orientacion: PatronOrientacion = PatronOrientacion.VERTICAL_VANO
    calidad_ancho: float = 1.0
    through_crack: bool = False
    rust_stains: bool = False
    spalling: bool = False
    field_answers: dict[str, str] = field(default_factory=dict)


@dataclass
class FiredRule:
    code: str
    statement: str
    cf_base: float
    cf_premises: float
    cf_contrib: float
    severity: ReportSeverity
    mechanism: str
    normative_basis: str
    actions: list[str]


@dataclass
class ExpertVerdict:
    """Dictamen completo para la interfaz y la bitácora."""

    has_crack: bool
    severity: ReportSeverity
    cf_combined: float
    mechanism: str
    normative_basis: str
    action_plan: list[str]
    max_allowed_width_mm: float
    fired_rules: list[FiredRule] = field(default_factory=list)
    notes: str = ""
    headline: str = ""


def cf_from_ml(probability: float) -> float:
    """Mapea P∈[0,1] a CF∈[-1,1]: P=0 → -1, P=0.5 → 0, P=1 → +1."""
    return max(-1.0, min(1.0, 2.0 * float(probability) - 1.0))


def combine_cf(cf1: float, cf2: float) -> float:
    """Combinación MYCIN de dos factores de certeza sobre la misma hipótesis."""
    if cf1 >= 0.0 and cf2 >= 0.0:
        return cf1 + cf2 * (1.0 - cf1)
    if cf1 < 0.0 and cf2 < 0.0:
        return cf1 + cf2 * (1.0 + cf1)
    denom = 1.0 - min(abs(cf1), abs(cf2))
    if denom <= 1e-12:
        return 0.0
    return (cf1 + cf2) / denom


def combine_cf_list(values: list[float]) -> float:
    acc = 0.0
    for value in values:
        acc = combine_cf(acc, value)
    return acc


def wmax_mm(ambiente: Exposure) -> float:
    return ACI_224R_LIMITS_MM[ambiente]


def _coherence(patron: PatronOrientacion, elemento: ElementType) -> float:
    """Atenúa el CF si el patrón es poco coherente con el tipo de elemento."""
    preferred: dict[PatronOrientacion, tuple[ElementType, ...]] = {
        PatronOrientacion.DIAGONAL_APOYOS: (ElementType.VIGA, ElementType.LOSA, ElementType.MURO),
        PatronOrientacion.VERTICAL_COLUMNA: (ElementType.COLUMNA, ElementType.MURO),
        PatronOrientacion.VERTICAL_VANO: (ElementType.VIGA, ElementType.LOSA),
        PatronOrientacion.HORIZONTAL_COLUMNA: (ElementType.COLUMNA,),
        PatronOrientacion.HELICOIDAL: (ElementType.VIGA, ElementType.COLUMNA),
        PatronOrientacion.MALLA: tuple(ElementType),
        PatronOrientacion.LONGITUDINAL_REFUERZO: tuple(ElementType),
    }
    return 1.0 if elemento in preferred[patron] else 0.55


def _emit(
    code: str,
    statement: str,
    cf_base: float,
    premises: list[float],
    severity: ReportSeverity,
    mechanism: str,
    normative_basis: str,
    actions: list[str],
) -> FiredRule:
    cf_prem = min(premises) if premises else 1.0
    cf_prem = max(0.0, cf_prem)
    return FiredRule(
        code=code,
        statement=statement,
        cf_base=cf_base,
        cf_premises=cf_prem,
        cf_contrib=cf_base * cf_prem,
        severity=severity,
        mechanism=mechanism,
        normative_basis=normative_basis,
        actions=actions,
    )


def _parse_enum(enum_cls: type[Enum], value: str | Enum) -> Enum:
    if isinstance(value, enum_cls):
        return value
    for member in enum_cls:
        if member.value == value or member.name == value:
            return member
    raise ValueError(f"Valor no reconocido para {enum_cls.__name__}: {value!r}")


def patron_from_geometry(elemento: str | ElementType, geometry: str) -> PatronOrientacion:
    """Mapea orientación de la foto + tipo de elemento al patrón del motor experto."""
    el = _parse_enum(ElementType, elemento)
    key = (geometry or "").strip().lower()
    if key in {"malla", "mesh"}:
        return PatronOrientacion.MALLA
    if key in {"inclinada", "diagonal", "45"}:
        return PatronOrientacion.DIAGONAL_APOYOS
    if key == "vertical":
        if el == ElementType.COLUMNA:
            return PatronOrientacion.VERTICAL_COLUMNA
        return PatronOrientacion.VERTICAL_VANO
    if key == "horizontal":
        if el == ElementType.COLUMNA:
            return PatronOrientacion.HORIZONTAL_COLUMNA
        return PatronOrientacion.LONGITUDINAL_REFUERZO
    return PatronOrientacion.VERTICAL_VANO


def observation_from_form(
    ml_probability: float,
    *,
    elemento: str,
    ambiente: str,
    patron_orientacion: str,
    ancho_mm: float | None,
    calidad_ancho: float = 1.0,
    through_crack: bool = False,
    rust_stains: bool = False,
    spalling: bool = False,
    field_answers: dict[str, str] | None = None,
) -> CrackObservation:
    answers = dict(field_answers or {})
    flags = flags_from_field_answers(answers)
    return CrackObservation(
        ml_probability=float(ml_probability),
        elemento=_parse_enum(ElementType, elemento),  # type: ignore[arg-type]
        ambiente=_parse_enum(Exposure, ambiente),  # type: ignore[arg-type]
        patron_orientacion=_parse_enum(PatronOrientacion, patron_orientacion),  # type: ignore[arg-type]
        ancho_mm=ancho_mm,
        calidad_ancho=max(0.0, min(1.0, calidad_ancho)),
        through_crack=through_crack or flags["through_crack"],
        rust_stains=rust_stains or flags["rust_stains"],
        spalling=spalling or flags["spalling"],
        field_answers=answers,
    )


def flags_from_field_answers(answers: dict[str, str]) -> dict[str, bool]:
    """Traduce Sí/No de campo a banderas ya usadas por R3b/R4/R5. 'No lo sé' no dispara."""
    q4 = answers.get("oxido_desprendimiento", UNKNOWN_ANSWER)
    return {
        "through_crack": answers.get("pasante") == "Sí",
        "rust_stains": q4 in {"Óxido", "Ambos"},
        "spalling": q4 in {"Se desprende el concreto", "Ambos"},
    }


def patron_with_field_answers(patron: str, answers: dict[str, str] | None) -> str:
    """Si el inspector marca malla, el patrón de foto no manda sobre esa observación."""
    if (answers or {}).get("malla") == "Varias en red":
        return PatronOrientacion.MALLA.value
    return patron


def _ans(answers: dict[str, str] | None, key: str) -> str | None:
    value = (answers or {}).get(key)
    if not value or value == UNKNOWN_ANSWER:
        return None
    return value


def compose_headline(obs: CrackObservation, severity: ReportSeverity, has_crack: bool) -> str:
    """Mensaje llano para la app: mezcla foto, elemento y las 9 respuestas."""
    if not has_crack:
        return "En esta foto no se ve una fisura clara. Siga con la inspección habitual."

    a = obs.field_answers or {}
    el = obs.elemento.value.lower()
    parts: list[str] = []

    if obs.spalling:
        parts.append("Se está cayendo concreto: ya no es solo una fisura de superficie.")
    elif _ans(a, "sismo") == "Sí" and obs.elemento == ElementType.COLUMNA:
        parts.append("Hubo un temblor y la fisura está en una columna: un ingeniero debe verla pronto.")
    elif _ans(a, "sismo") == "Sí":
        parts.append("La relacionó con un temblor reciente: no la trate como una grieta de curado.")
    elif _ans(a, "ubicacion") == "En el apoyo" and obs.elemento == ElementType.VIGA:
        parts.append("Está en el apoyo de la viga: eso apunta más a cortante que a una grieta del medio del vano.")
    elif obs.patron_orientacion == PatronOrientacion.DIAGONAL_APOYOS:
        parts.append(f"El patrón en diagonal en este {el} sugiere un mecanismo serio (cortante). Un ingeniero debe revisarlo.")
    elif obs.rust_stains:
        parts.append("Hay óxido a la vista: el acero puede estar perdiendo protección.")
    elif _ans(a, "humedad") == "Sí" or _ans(a, "pasante") == "Sí":
        parts.append("Hay agua o la grieta atraviesa el elemento: el riesgo es filtración y corrosión, no solo que se vea mal.")
    elif _ans(a, "evolucion") == "Crece":
        parts.append("Dijo que la grieta crece: el mecanismo sigue activo.")
    elif _ans(a, "malla") == "Varias en red":
        parts.append("Varias grietas en red suelen ser de retracción o mal curado, no de cortante.")
    elif _ans(a, "ubicacion") == "En una junta":
        parts.append("Si sigue una junta, no la interprete sola como falla del alma a 45°.")
    elif _ans(a, "carga_arriba") == "Sí" and obs.elemento == ElementType.COLUMNA:
        parts.append("Hay pisos o carga arriba de esta columna: hay que mirar si se está aplastando.")

    if severity == ReportSeverity.CRITICA and not parts:
        parts.append("Con la foto y lo que marcó en campo, el mecanismo se ve serio. Un ingeniero debe revisar el elemento.")
    elif severity == ReportSeverity.MODERADA and not parts:
        parts.append("Hay fisura y, con lo respondido, puede afectar durabilidad (agua, corrosión o que siga abriéndose). Conviene actuar pronto.")
    elif severity == ReportSeverity.LEVE and not parts:
        parts.append("Hay una fisura, pero con lo que respondió el riesgo inmediato parece bajo. Conviene vigilarla.")
    elif severity == ReportSeverity.CRITICA and parts and "ingeniero" not in parts[0].lower():
        parts.append("Un ingeniero debe revisar el elemento; no basta sellar.")
    elif severity == ReportSeverity.MODERADA and len(parts) == 1 and _ans(a, "evolucion") != "Crece":
        if _ans(a, "antiguedad") == "Años":
            parts.append("Lleva años: mire más el deterioro acumulado (óxido, recubrimiento) que un evento de ayer.")
        elif _ans(a, "antiguedad") == "Días" and _ans(a, "sismo") != "Sí":
            parts.append("Es reciente: conviene marcarla y volver a verla en pocos días.")

    return " ".join(parts[:2])


def compose_action_plan(
    obs: CrackObservation,
    severity: ReportSeverity,
    has_crack: bool,
    fired_actions: list[str],
) -> list[str]:
    """Qué hacer, en orden de lo que el inspector acaba de marcar."""
    if not has_crack:
        return ["Mantener la inspección rutinaria; si más adelante aparece una fisura, fotografíela de frente y de cerca."]

    a = obs.field_answers or {}
    out: list[str] = []

    def add(text: str) -> None:
        if text and text not in out:
            out.append(text)

    if severity == ReportSeverity.CRITICA:
        add("Delimite la zona y avise a un ingeniero civil. No se limite a pintar o sellar.")
    if _ans(a, "sismo") == "Sí":
        add("Revise nudos y columnas, y fíjese si hay fisuras horizontales en elementos verticales (posible daño del temblor).")
    if _ans(a, "ubicacion") == "En el apoyo" and obs.elemento == ElementType.VIGA:
        add("En el apoyo mire estribos, alma y el nudo viga-columna; no lo trate como grieta estética del vano.")
    if obs.spalling:
        add("Delimite el concreto suelto y no deje el acero al aire; hay que ver qué sección queda.")
    if obs.rust_stains:
        add("Anote las manchas de óxido y, si puede, el recubrimiento. La corrosión ya se ve.")
    if _ans(a, "humedad") == "Sí" or _ans(a, "pasante") == "Sí":
        add("Busque de dónde viene el agua, mire el otro lado del elemento, seque y selle.")
    if _ans(a, "evolucion") == "Crece":
        add("Ponga testigos (yeso o vidrio) y tome otra foto con una regla en unos días, para ver si sigue abriéndose.")
    if _ans(a, "carga_arriba") == "Sí" and obs.elemento == ElementType.COLUMNA:
        add("Hay carga de pisos arriba: mire desprendimiento o fisuras verticales de aplastamiento.")
    if _ans(a, "malla") == "Varias en red":
        add("Si es malla de retracción, controle humedad y selle si hay filtración; no la confunda con cortante.")
    if _ans(a, "ubicacion") == "En una junta":
        add("Revise el sellado de la junta; una junta abierta no es lo mismo que una grieta a 45° en el alma.")
    if _ans(a, "antiguedad") == "Años" and obs.ambiente != Exposure.INTERIOR_SECO:
        add("Como lleva años y el ambiente no es interior seco, priorice corrosión y recubrimiento.")
    if severity in {ReportSeverity.MODERADA, ReportSeverity.LEVE}:
        add("Tome una foto de frente, de cerca, con una regla o una moneda para comparar en la próxima visita.")
    if severity == ReportSeverity.LEVE and _ans(a, "evolucion") != "Crece":
        add("Si no crece y no hay agua ni óxido, puede esperar a la siguiente inspección, pero no la pierda de vista.")

    for action in fired_actions:
        add(action)
        if len(out) >= 5:
            break
    return out[:5]


def evaluate_pathology(obs: CrackObservation) -> ExpertVerdict:
    """Encadenamiento hacia adelante: P_ML × elemento × ambiente × ancho × patrón."""
    limit = wmax_mm(obs.ambiente)
    cf_p = cf_from_ml(obs.ml_probability)
    cf_el = CF_FACT_DECLARED
    cf_amb = CF_FACT_DECLARED
    cf_pat = CF_PATTERN_OBS * _coherence(obs.patron_orientacion, obs.elemento)
    cf_w = float(obs.calidad_ancho) if obs.ancho_mm is not None else 0.0
    fired: list[FiredRule] = []

    if obs.ml_probability < ML_CRACK_THRESHOLD:
        rule = _emit(
            "R0",
            f"SI P={obs.ml_probability:.3f} < 0.50 ENTONCES sin fisura detectable.",
            0.80,
            [max(0.0, -cf_p)],
            ReportSeverity.NONE,
            "Sin mecanismo patológico atribuible (percepción negativa).",
            "El dictamen normativo de ACI 224R/318 no se activa sin evidencia visual de fisura.",
            ["Mantener inspección rutinaria según el plan de mantenimiento."],
        )
        fired.append(rule)
        return ExpertVerdict(
            has_crack=False,
            severity=ReportSeverity.NONE,
            cf_combined=round(rule.cf_contrib, 4),
            mechanism=rule.mechanism,
            normative_basis=rule.normative_basis,
            action_plan=compose_action_plan(obs, ReportSeverity.NONE, False, rule.actions),
            max_allowed_width_mm=limit,
            fired_rules=fired,
            notes="El motor experto no interpreta ancho ni patrón si la CNN no detecta fisura.",
            headline=compose_headline(obs, ReportSeverity.NONE, False),
        )

    # --- Percepción ---
    fired.append(
        _emit(
            "R1",
            f"SI P={obs.ml_probability:.3f} ≥ 0.50 ENTONCES fisura presente (Leve inicial).",
            0.90,
            [max(0.0, cf_p)],
            ReportSeverity.LEVE,
            "Fisura superficial detectada por visión (CNN); mecanismo aún no especializado.",
            "Capa de percepción; la interpretación estructural se rige por ACI 224R-01, ACI 318 y NEC-SE-HM.",
            ["Fotografiar con escala y registrar localización en el elemento."],
        )
    )

    # --- Ancho / exposición (ACI 224R Tabla 4.1) ---
    fired.append(
        _emit(
            "R6",
            f"SI ambiente={obs.ambiente.value} ENTONCES w_max={limit:.2f} mm (ACI 224R Tabla 4.1).",
            1.00,
            [cf_amb],
            ReportSeverity.LEVE,
            "Límite de servicio por exposición (durabilidad / apariencia).",
            "ACI 224R-01 Tabla 4.1; NEC-SE-HM (durabilidad y recubrimiento según exposición).",
            [],
        )
    )

    if obs.ancho_mm is None:
        fired.append(
            _emit(
                "R6b",
                "SI P≥0.50 Y ancho desconocido ENTONCES evidencia incompleta (Leve, no contrastado).",
                0.55,
                [max(0.0, cf_p)],
                ReportSeverity.LEVE,
                "Mecanismo no contrastado: falta medición de ancho.",
                "ACI 224R-01 requiere ancho máximo para juzgar el estado límite de servicio.",
                ["Medir ancho máximo con fisurómetro y reejecutar el motor."],
            )
        )
    else:
        w = float(obs.ancho_mm)
        if limit < w < GROSS_WIDTH_MM:
            fired.append(
                _emit(
                    "R7",
                    f"SI w={w:.2f} mm > w_max={limit:.2f} mm ENTONCES Moderada (servicio/durabilidad).",
                    0.85,
                    [max(0.0, cf_p), cf_w, cf_amb],
                    ReportSeverity.MODERADA,
                    "Incumplimiento del ancho de servicio; riesgo de ingreso de agentes agresivos.",
                    "ACI 224R-01 Tabla 4.1; NEC-SE-HM (estado límite de servicio y durabilidad).",
                    [
                        "Colocar testigos de evolución y sellar con material elástico compatible.",
                        "Revisar recubrimiento, flecha y humedad en el elemento.",
                    ],
                )
            )
        if w >= GROSS_WIDTH_MM:
            fired.append(
                _emit(
                    "R8",
                    f"SI w={w:.2f} mm ≥ 1.0 mm ENTONCES Crítica (fisura grosera).",
                    0.88,
                    [max(0.0, cf_p), cf_w],
                    ReportSeverity.CRITICA,
                    "Fisura grosera: posible pérdida de sección, corrosión avanzada o mecanismo no flexional.",
                    "ACI 224R-01 (anchos groseros fuera de servicio); ACI 318 / NEC-SE-HM (alerta de integridad).",
                    [
                        "Delimitar la zona y restringir uso según el ingeniero responsable.",
                        "Solicitar evaluación estructural formal (no basta el sellado).",
                    ],
                )
            )
        if obs.ambiente == Exposure.MARINO and w <= limit:
            fired.append(
                _emit(
                    "R10",
                    "SI ambiente marino/agresivo Y w ≤ w_max ENTONCES Moderada (cloruros), no meramente estética.",
                    0.70,
                    [max(0.0, cf_p), cf_w, cf_amb],
                    ReportSeverity.MODERADA,
                    "Durabilidad en exposición a cloruros aun con ancho 'tolerable' de flexión.",
                    "NEC-SE-HM (exposición agresiva / recubrimiento); ACI 224R-01 (agua de mar 0.15 mm).",
                    [
                        "Verificar recubrimiento y manchas de óxido.",
                        "Considerar protección superficial según NEC-SE-HM.",
                    ],
                )
            )

    if obs.spalling:
        fired.append(
            _emit(
                "R4",
                "SI hay desprendimiento (spalling) ENTONCES Crítica.",
                0.90,
                [max(0.0, cf_p)],
                ReportSeverity.CRITICA,
                "Pérdida de recubrimiento y posible reducción de sección de hormigón/acero.",
                "ACI 318 (integridad de la sección); NEC-SE-HM (reparación de recubrimiento).",
                ["Delimitar el área y reparar recubrimiento; evaluar sección residual."],
            )
        )
    if obs.through_crack and obs.elemento in {ElementType.LOSA, ElementType.MURO}:
        fired.append(
            _emit(
                "R5",
                "SI fisura pasante en losa/muro ENTONCES al menos Moderada (filtración).",
                0.78,
                [max(0.0, cf_p), cf_el],
                ReportSeverity.MODERADA,
                "Fisura pasante: camino de filtración y lixiviación.",
                "ACI 224R-01 (fisuras en elementos retenedores / exposición húmeda); NEC-SE-HM.",
                ["Sellar, revisar impermeabilización y controlar filtraciones."],
            )
        )
    if obs.rust_stains:
        fired.append(
            _emit(
                "R3b",
                "SI hay manchas de óxido ENTONCES Moderada (corrosión visible).",
                0.80,
                [max(0.0, cf_p)],
                ReportSeverity.MODERADA,
                "Indicio de corrosión de armadura (despasivación).",
                "ACI 224R / ACI 318 (fisuración asociada a corrosión); NEC-SE-HM (durabilidad).",
                ["Ensayos de carbonatación, cloruros y recubrimiento; rehabilitar recubrimiento."],
            )
        )

    # --- Patrón / orientación (variable patron_orientacion) ---
    fired.extend(_rules_patron(obs, cf_p, cf_el, cf_amb, cf_pat, cf_w, limit))
    fired.extend(_rules_field_interview(obs, cf_p, cf_el, cf_amb))

    winner = max(fired, key=lambda r: (_SEVERITY_RANK[r.severity], r.cf_contrib))
    supporting = [r.cf_contrib for r in fired if r.severity == winner.severity and r.cf_contrib > 0]
    cf_comb = combine_cf_list(supporting) if supporting else winner.cf_contrib

    mechanisms = [r.mechanism for r in fired if r.severity == winner.severity and r.code not in {"R1", "R6"}]
    mechanism = mechanisms[-1] if mechanisms else winner.mechanism
    norms = list(dict.fromkeys(r.normative_basis for r in fired if r.normative_basis))
    actions: list[str] = []
    for rank in (ReportSeverity.CRITICA, ReportSeverity.MODERADA, ReportSeverity.LEVE):
        for rule in fired:
            if rule.severity == rank:
                for action in rule.actions:
                    if action not in actions:
                        actions.append(action)
    if not actions:
        actions.append("Registrar fisura, fotografiar con escala y reevaluar en la próxima inspección.")

    return ExpertVerdict(
        has_crack=True,
        severity=winner.severity,
        cf_combined=round(float(cf_comb), 4),
        mechanism=mechanism,
        normative_basis=" | ".join(norms),
        action_plan=compose_action_plan(obs, winner.severity, True, actions),
        max_allowed_width_mm=limit,
        fired_rules=fired,
        notes=(
            "Dictamen de apoyo a la decisión. No calcula capacidad residual φVn ni sustituye "
            "al ingeniero civil responsable. CF combinado estilo MYCIN sobre la hipótesis de severidad ganadora."
        ),
        headline=compose_headline(obs, winner.severity, True),
    )


def _rules_patron(
    obs: CrackObservation,
    cf_p: float,
    cf_el: float,
    cf_amb: float,
    cf_pat: float,
    cf_w: float,
    limit: float,
) -> list[FiredRule]:
    p = obs.patron_orientacion
    e = obs.elemento
    a = obs.ambiente
    w = obs.ancho_mm
    prem_pat = [max(0.0, cf_p), cf_pat, cf_el]
    rules: list[FiredRule] = []

    if p == PatronOrientacion.DIAGONAL_APOYOS:
        sev = ReportSeverity.CRITICA
        rules.append(
            _emit(
                "R-P1",
                "SI patrón=Diagonal ~45° en apoyos ENTONCES Cortante / tensión diagonal → Crítica.",
                0.92,
                prem_pat,
                sev,
                "Cortante / tensión diagonal (fisuración inclinada en zona de apoyos o alma).",
                "ACI 318 (cortante en vigas, fisuras a ~45°); NEC-SE-HM (resistencia a cortante y estribos).",
                [
                    "Restringir cargas; evaluación estructural inmediata.",
                    "Revisar estribos, nudos y posible punzonamiento; no usar el sellado como única medida.",
                ],
            )
        )
        if e == ElementType.VIGA:
            rules.append(
                _emit(
                    "R-P1b",
                    "SI elemento=Viga Y patrón diagonal en apoyos ENTONCES alma a cortante (ACI 318).",
                    0.93,
                    prem_pat,
                    ReportSeverity.CRITICA,
                    "Cortante en viga: mecanismo potencialmente frágil.",
                    "ACI 318 / NEC-SE-HM: verificar Vc, estribos y anclaje en apoyos.",
                    ["Inspeccionar alma, apoyos y nudos viga-columna."],
                )
            )

    elif p == PatronOrientacion.VERTICAL_COLUMNA:
        sev = ReportSeverity.MODERADA
        if w is not None and w >= 0.50:
            sev = ReportSeverity.CRITICA if w >= GROSS_WIDTH_MM else ReportSeverity.MODERADA
        rules.append(
            _emit(
                "R-P2",
                "SI patrón=Vertical paralela al eje en columna ENTONCES compresión / aplastamiento.",
                0.86,
                prem_pat + ([cf_w] if w is not None else []),
                sev,
                "Compresión pura / aplastamiento (hendimiento paralelo a la carga axial).",
                "ACI 318 (columnas a compresión, fisuración vertical); NEC-SE-HM (elementos a compresión).",
                [
                    "Revisar carga axial, esbeltez y confinamiento; no atribuir automáticamente a retracción.",
                    "Si hay desprendimiento o w≥1 mm, restringir uso y evaluar sección.",
                ],
            )
        )

    elif p == PatronOrientacion.VERTICAL_VANO:
        sev = ReportSeverity.LEVE
        if w is not None and w > limit:
            sev = ReportSeverity.MODERADA
        if w is not None and w >= GROSS_WIDTH_MM:
            sev = ReportSeverity.CRITICA
        if a == Exposure.MARINO and sev == ReportSeverity.LEVE:
            sev = ReportSeverity.MODERADA
        rules.append(
            _emit(
                "R-P3",
                "SI patrón=Vertical/perpendicular en centro de vano ENTONCES flexión pura.",
                0.88,
                prem_pat,
                sev,
                "Flexión pura (fisuras perpendiculares al acero de tracción en zona de momento máximo).",
                "ACI 318 (fisuración por flexión); ACI 224R-01 Tabla 4.1 (ancho de servicio); NEC-SE-HM.",
                [
                    "Contrastar w con w_max de exposición; sellar fisuras de flexión si el elemento es estable.",
                    "Controlar flecha y cuantía de tracción.",
                ],
            )
        )

    elif p == PatronOrientacion.HORIZONTAL_COLUMNA:
        rules.append(
            _emit(
                "R-P4",
                "SI patrón=Horizontal transversal en columna ENTONCES flexo-tracción sísmica / viento → Crítica.",
                0.90,
                prem_pat,
                ReportSeverity.CRITICA,
                "Flexo-tracción por sismo o viento (fisuras horizontales por momento flector en columna).",
                "ACI 318 (columnas a flexocompresión, nudos); NEC-SE-HM y NEC-SE-DS (demanda sísmica).",
                [
                    "Evaluación sísmica/estructural urgente del pórtico.",
                    "Revisar nudos, estribos de confinamiento y posible rótula plástica.",
                ],
            )
        )

    elif p == PatronOrientacion.HELICOIDAL:
        rules.append(
            _emit(
                "R-P5",
                "SI patrón=Helicoidal/espiral 45° ENTONCES falla por torsión → Crítica.",
                0.91,
                prem_pat,
                ReportSeverity.CRITICA,
                "Falla por torsión (fisuración helicoidal a ~45°).",
                "ACI 318 (torsión, estribos cerrados y refuerzo longitudinal extra); NEC-SE-HM.",
                [
                    "Restringir cargas de torsión (voladizos, excentricidades).",
                    "Verificar estribos cerrados y continuidad del refuerzo; evaluación formal.",
                ],
            )
        )

    elif p == PatronOrientacion.MALLA:
        sev = ReportSeverity.LEVE
        if a != Exposure.INTERIOR_SECO:
            sev = ReportSeverity.MODERADA
        rules.append(
            _emit(
                "R-P6",
                "SI patrón=Malla / piel de cocodrilo ENTONCES retracción plástica / curado deficiente.",
                0.75,
                prem_pat + [cf_amb],
                sev,
                "Retracción plástica / curado deficiente (no confundir con cortante).",
                "ACI 224R-01 (fisuración por retracción y curado); NEC-SE-HM (curado y durabilidad).",
                [
                    "Controlar humedad y curado; sellar si hay riesgo de filtración.",
                    "Evaluar RAS solo si hay expansión o gel; no tratar como cortante.",
                ],
            )
        )

    elif p == PatronOrientacion.LONGITUDINAL_REFUERZO:
        sev = ReportSeverity.MODERADA
        if a == Exposure.MARINO or obs.rust_stains or obs.spalling:
            sev = ReportSeverity.CRITICA
        if w is not None and w >= GROSS_WIDTH_MM:
            sev = ReportSeverity.CRITICA
        rules.append(
            _emit(
                "R-P7",
                "SI patrón=Longitudinal paralela al refuerzo ENTONCES corrosión y despasivación.",
                0.84,
                prem_pat + [cf_amb],
                sev,
                "Corrosión y despasivación de armaduras (fisura siguiendo la barra).",
                "ACI 224R / ACI 318 (fisuración por corrosión); NEC-SE-HM (recubrimiento y exposición a cloruros).",
                [
                    "Ensayos de carbonatación, cloruros y espesor de recubrimiento.",
                    "Rehabilitar recubrimiento; protección catódica o inhibidores según el caso.",
                ],
            )
        )

    if e == ElementType.LOSA and a != Exposure.INTERIOR_SECO and w is not None and w > limit:
        sev = ReportSeverity.CRITICA if a == Exposure.MARINO else ReportSeverity.MODERADA
        rules.append(
            _emit(
                "R13",
                "SI losa en ambiente no seco Y w>w_max ENTONCES servicio/filtración.",
                0.80,
                [max(0.0, cf_p), cf_el, cf_amb, cf_w],
                sev,
                "Flexión/servicio en losa con riesgo de filtración.",
                "ACI 224R-01; NEC-SE-HM (losas y durabilidad).",
                ["Impermeabilizar, controlar flecha y sellar fisuras de flexión."],
            )
        )

    return rules


def _known(answers: dict[str, str], key: str) -> str | None:
    return _ans(answers, key)


def _rules_field_interview(
    obs: CrackObservation,
    cf_p: float,
    cf_el: float,
    cf_amb: float,
) -> list[FiredRule]:
    """Reglas de las 9 preguntas de campo. 'No lo sé' no dispara (no inventa evidencia)."""
    answers = obs.field_answers or {}
    if not answers:
        return []
    cf_f = CF_FIELD_OBS
    prem = [max(0.0, cf_p), cf_f, cf_el]
    rules: list[FiredRule] = []
    e = obs.elemento

    loc = _known(answers, "ubicacion")
    if loc == "En el apoyo" and e == ElementType.VIGA:
        sev = (
            ReportSeverity.CRITICA
            if obs.patron_orientacion == PatronOrientacion.DIAGONAL_APOYOS
            else ReportSeverity.MODERADA
        )
        rules.append(
            _emit(
                "R-F1",
                "SI grieta en el apoyo de viga ENTONCES sospecha de cortante (no solo flexión de vano).",
                0.82,
                prem,
                sev,
                "Zona de apoyo: cortante / tensión diagonal más probable que flexión de centro de vano.",
                "ACI 318 (cortante en vigas, zona de apoyos); NEC-SE-HM.",
                ["Revisar alma, estribos y nudo en el apoyo; no tratarlo como fisura estética de vano."],
            )
        )
    elif loc == "En el medio" and e in {ElementType.VIGA, ElementType.LOSA}:
        rules.append(
            _emit(
                "R-F1b",
                "SI grieta en el medio del vano ENTONCES compatible con flexión.",
                0.70,
                prem,
                ReportSeverity.LEVE,
                "Centro de vano: mecanismo flexional más coherente que cortante de apoyo.",
                "ACI 318 (fisuración por flexión); ACI 224R-01.",
                ["Contrastar ancho con w_max de exposición y vigilar flecha."],
            )
        )
    elif loc == "En una junta":
        rules.append(
            _emit(
                "R-F1c",
                "SI la grieta sigue una junta ENTONCES no interpretarla como cortante de alma.",
                0.72,
                prem,
                ReportSeverity.LEVE,
                "Junta de construcción o encuentro: fisura preferencial, no necesariamente mecanismo frágil.",
                "ACI 224R-01 (juntas y fisuración); NEC-SE-HM.",
                ["Revisar sellado de junta; no confundir con cortante a 45° en el alma."],
            )
        )

    if _known(answers, "humedad") == "Sí":
        rules.append(
            _emit(
                "R-F3",
                "SI hay humedad, goteo o mancha de agua ENTONCES al menos Moderada (filtración).",
                0.78,
                prem + [cf_amb],
                ReportSeverity.MODERADA,
                "Humedad visible: camino de filtración y riesgo de corrosión/lixiviación.",
                "ACI 224R-01 (exposición húmeda); NEC-SE-HM (durabilidad).",
                ["Sellar, rastrear la filtración y secar el elemento antes de un recubrimiento."],
            )
        )

    antiguedad = _known(answers, "antiguedad")
    if antiguedad == "Años":
        rules.append(
            _emit(
                "R-F5",
                "SI se observó hace años ENTONCES durabilidad crónica (no emergencia nueva por sí sola).",
                0.65,
                prem,
                ReportSeverity.MODERADA,
                "Fisura antigua: más relevante el deterioro acumulado que un evento único reciente.",
                "ACI 224R-01 (evolución y durabilidad).",
                ["Comparar con fotos viejas; priorizar corrosión y recubrimiento si el ambiente es agresivo."],
            )
        )
    elif antiguedad == "Días" and _known(answers, "sismo") == "Sí":
        rules.append(
            _emit(
                "R-F5b",
                "SI apareció hace días Y hubo temblor reciente ENTONCES posible daño sísmico.",
                0.84,
                prem,
                ReportSeverity.CRITICA if e == ElementType.COLUMNA else ReportSeverity.MODERADA,
                "Aparición post-sismo: no atribuir a retracción de curado.",
                "NEC-SE-DS / NEC-SE-HM (demanda sísmica); ACI 318 (nudos y columnas).",
                ["Inspeccionar nudos y elementos verticales; no limitarse al sellado."],
            )
        )

    if _known(answers, "sismo") == "Sí" and antiguedad != "Días":
        sev = ReportSeverity.CRITICA if e == ElementType.COLUMNA else ReportSeverity.MODERADA
        rules.append(
            _emit(
                "R-F6",
                "SI hubo temblor reciente ENTONCES elevar la alerta (sobre todo en columna).",
                0.80,
                prem,
                sev,
                "Demanda sísmica reciente: posible flexo-tracción o daño en nudos.",
                "NEC-SE-DS / NEC-SE-HM; ACI 318 (columnas y nudos).",
                ["Revisar columnas, nudos y fisuras horizontales; evaluación estructural si hay duda."],
            )
        )

    if _known(answers, "carga_arriba") == "Sí" and e == ElementType.COLUMNA:
        rules.append(
            _emit(
                "R-F7",
                "SI hay pisos o carga arriba Y el elemento es columna ENTONCES atención a compresión.",
                0.76,
                prem,
                ReportSeverity.MODERADA,
                "Columna con carga de pisos superiores: fisuración vertical más coherente con aplastamiento.",
                "ACI 318 (elementos a compresión); NEC-SE-HM.",
                ["Revisar desprendimiento, esbeltez y confinamiento; no tratarlo como retracción de losa."],
            )
        )

    if _known(answers, "evolucion") == "Crece":
        rules.append(
            _emit(
                "R-F8",
                "SI la grieta crece ENTONCES al menos Moderada (mecanismo activo).",
                0.83,
                prem,
                ReportSeverity.MODERADA,
                "Evolución activa: el ancho o la longitud no se han estabilizado.",
                "ACI 224R-01 (monitoreo de fisuras); NEC-SE-HM.",
                ["Colocar testigos, fotografiar con escala y reevaluar; no asumir que está quieta."],
            )
        )

    if _known(answers, "pasante") == "Sí" and e not in {ElementType.LOSA, ElementType.MURO}:
        rules.append(
            _emit(
                "R-F9",
                "SI la grieta es pasante en viga/columna ENTONCES Moderada (sección atravesada).",
                0.74,
                prem,
                ReportSeverity.MODERADA,
                "Fisura pasante: discontinuidad a través del espesor, no solo piel.",
                "ACI 224R-01; ACI 318 (integridad de sección).",
                ["Medir si hay desplazamiento; evaluación formal si hay desalineación o desprendimiento."],
            )
        )

    return rules


def diagnose(
    ml_probability: float,
    *,
    elemento: str,
    ambiente: str,
    patron_orientacion: str,
    ancho_mm: float | None,
    calidad_ancho: float = 1.0,
    through_crack: bool = False,
    rust_stains: bool = False,
    spalling: bool = False,
    field_answers: dict[str, str] | None = None,
) -> ExpertVerdict:
    """API de la interfaz: cadenas del formulario → dictamen."""
    obs = observation_from_form(
        ml_probability,
        elemento=elemento,
        ambiente=ambiente,
        patron_orientacion=patron_orientacion,
        ancho_mm=ancho_mm,
        calidad_ancho=calidad_ancho,
        through_crack=through_crack,
        rust_stains=rust_stains,
        spalling=spalling,
        field_answers=field_answers,
    )
    return evaluate_pathology(obs)


def verdict_to_dict(verdict: ExpertVerdict) -> dict[str, object]:
    return {
        "has_crack": verdict.has_crack,
        "severity": verdict.severity.value,
        "cf_combined": verdict.cf_combined,
        "mechanism": verdict.mechanism,
        "normative_basis": verdict.normative_basis,
        "action_plan": verdict.action_plan,
        "max_allowed_width_mm": verdict.max_allowed_width_mm,
        "fired_rules": [
            {
                "code": r.code,
                "statement": r.statement,
                "cf_contrib": round(r.cf_contrib, 4),
                "severity": r.severity.value,
            }
            for r in verdict.fired_rules
        ],
        "notes": verdict.notes,
        "headline": verdict.headline,
    }
