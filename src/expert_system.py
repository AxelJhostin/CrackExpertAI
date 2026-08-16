"""Motor de reglas de patología estructural en hormigón (ACI 224R / ACI 318 / NEC-SE-HM).

El clasificador visual estima la presencia de fisura. Este módulo traduce
medidas (ancho, patrón, exposición) a un dictamen de servicio y urgencia,
sin sustituir un peritaje estructural in situ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    NONE = "sin_fisura"
    AESTHETIC = "estetica"
    SERVICEABILITY = "servicio"
    STRUCTURAL = "estructural"
    CRITICAL = "critica"


class Pattern(str, Enum):
    UNKNOWN = "desconocido"
    FLEXURAL = "flexion"
    SHEAR = "cortante"
    MAP = "mapa"
    CORROSION = "corrosion"
    SETTLEMENT = "asentamiento"
    THERMAL = "termica"


@dataclass(frozen=True)
class CrackObservation:
    """Entrada al sistema experto (unidades SI: mm)."""

    ml_probability: float
    width_mm: float | None = None
    length_mm: float | None = None
    pattern: Pattern = Pattern.UNKNOWN
    wet_environment: bool = False
    deicing_salts: bool = False
    seawater: bool = False
    water_retaining: bool = False
    through_crack: bool = False
    rust_stains: bool = False
    spalling: bool = False


@dataclass
class ExpertVerdict:
    """Salida trazable: reglas disparadas + dictamen."""

    has_crack: bool
    severity: Severity
    max_allowed_width_mm: float
    actions: list[str]
    fired_rules: list[str] = field(default_factory=list)
    notes: str = ""


# ACI 224R-01 Table 4.1 — anchos máximos de fisura (mm) según exposición.
ACI_224R_LIMITS_MM: dict[str, float] = {
    "dry_air": 0.41,
    "humidity_moist_soil": 0.30,
    "deicing": 0.18,
    "seawater": 0.15,
    "water_retaining": 0.10,
}

# Umbral operativo de detección visual (probabilidad sigmoid).
ML_CRACK_THRESHOLD: float = 0.50
_SEVERITY_RANK: dict[Severity, int] = {s: i for i, s in enumerate(Severity)}


def _raise_severity(current: Severity, candidate: Severity) -> Severity:
    return candidate if _SEVERITY_RANK[candidate] > _SEVERITY_RANK[current] else current


def aci_224r_width_limit(obs: CrackObservation) -> float:
    """Selecciona el límite de ancho de fisura según exposición (ACI 224R / NEC-SE-HM)."""
    if obs.water_retaining:
        return ACI_224R_LIMITS_MM["water_retaining"]
    if obs.seawater:
        return ACI_224R_LIMITS_MM["seawater"]
    if obs.deicing_salts:
        return ACI_224R_LIMITS_MM["deicing"]
    if obs.wet_environment:
        return ACI_224R_LIMITS_MM["humidity_moist_soil"]
    return ACI_224R_LIMITS_MM["dry_air"]


def evaluate_pathology(obs: CrackObservation) -> ExpertVerdict:
    """Aplica reglas deterministas sobre la observación + score del modelo.

    Orden de precedencia: indicios de fallo estructural > corrosión/desprendimiento
    > incumplimiento de ancho ACI > patrón de cortante > fisura estética.
    """
    fired: list[str] = []
    actions: list[str] = []
    limit = aci_224r_width_limit(obs)
    has_crack = obs.ml_probability >= ML_CRACK_THRESHOLD

    if not has_crack:
        fired.append("R0: P(fisura) < 0.50 → elemento visualmente sano para el detector.")
        return ExpertVerdict(
            has_crack=False,
            severity=Severity.NONE,
            max_allowed_width_mm=limit,
            actions=["Mantener inspección rutinaria según plan de mantenimiento."],
            fired_rules=fired,
            notes="El sistema experto no evalúa ancho si el clasificador no detecta fisura.",
        )

    fired.append(f"R1: P(fisura)={obs.ml_probability:.3f} ≥ 0.50 → fisura detectada.")
    severity = Severity.AESTHETIC

    if obs.pattern == Pattern.SHEAR:
        severity = Severity.STRUCTURAL
        fired.append("R2 (ACI 318 / NEC-SE-HM): patrón de cortante → posible mecanismo frágil.")
        actions.append("Restringir cargas; evaluación estructural inmediata por profesional calificado.")

    if obs.pattern == Pattern.CORROSION or obs.rust_stains:
        severity = _raise_severity(severity, Severity.SERVICEABILITY)
        fired.append("R3: indicios de corrosión de armadura (manchas / patrón paralelo a barras).")
        actions.append("Verificar recubrimiento, carbonatación y cloruros; considerar rehabilitación.")

    if obs.spalling:
        severity = Severity.STRUCTURAL
        fired.append("R4: desprendimiento (spalling) → pérdida de sección o recubrimiento.")
        actions.append("Delimitar zona afectada y programar reparación del recubrimiento.")

    if obs.through_crack:
        severity = _raise_severity(severity, Severity.SERVICEABILITY)
        fired.append("R5: fisura pasante — riesgo de filtración y durabilidad.")
        actions.append("Sellar fisura y revisar impermeabilización.")

    if obs.width_mm is not None:
        fired.append(f"R6: ancho medido={obs.width_mm:.2f} mm; límite ACI 224R={limit:.2f} mm.")
        if obs.width_mm > limit:
            severity = _raise_severity(severity, Severity.SERVICEABILITY)
            fired.append("R7: ancho supera el límite de servicio de ACI 224R / NEC-SE-HM.")
            actions.append("Cuantificar evolución (testigos) y sellar; revisar flechas y recubrimiento.")
        if obs.width_mm >= 1.0:
            severity = Severity.CRITICAL
            fired.append("R8: ancho ≥ 1.0 mm → fisura grosera; posible compromiso de integridad.")
            actions.append("Evacuar/apuntar según criterio del ingeniero responsable.")
    else:
        fired.append("R6b: sin medición de ancho; dictamen limitado a patrón y probabilidad ML.")
        actions.append("Medir ancho máximo (fisurómetro) para contrastar con ACI 224R Tabla 4.1.")

    if obs.pattern == Pattern.MAP:
        fired.append("R9: fisuración en mapa — típica de retracción / álcali-agregado; no es cortante.")
        actions.append("Controlar humedad y evaluar potencial de RAS si hay expansión.")

    if not actions:
        actions.append("Registrar fisura, fotografiar con escala y reevaluar en la próxima inspección.")

    unique_actions = list(dict.fromkeys(actions))
    return ExpertVerdict(
        has_crack=True,
        severity=severity,
        max_allowed_width_mm=limit,
        actions=unique_actions,
        fired_rules=fired,
        notes=(
            "Referencias: ACI 224R-01 (control of cracking), ACI 318 (mecanismos de fallo), "
            "NEC-SE-HM (hormigón armado, Ecuador) para límites de servicio y durabilidad."
        ),
    )


def verdict_to_dict(verdict: ExpertVerdict) -> dict[str, object]:
    """Serializa el dictamen para la bitácora o una API posterior."""
    return {
        "has_crack": verdict.has_crack,
        "severity": verdict.severity.value,
        "max_allowed_width_mm": verdict.max_allowed_width_mm,
        "actions": verdict.actions,
        "fired_rules": verdict.fired_rules,
        "notes": verdict.notes,
    }
