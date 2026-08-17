"""Orientación geométrica de la fisura a partir de la foto (OpenCV).

No sustituye a la CNN: solo estima si el trazo dominante es vertical,
horizontal, inclinado (~45°) o tipo malla. Ángulos en el plano de la imagen.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

VERTICAL = "vertical"
HORIZONTAL = "horizontal"
INCLINADA = "inclinada"
MALLA = "malla"
DESCONOCIDA = "desconocida"

DISPLAY_NAME: dict[str, str] = {
    VERTICAL: "Vertical",
    HORIZONTAL: "Horizontal",
    INCLINADA: "Inclinada (~45°)",
    MALLA: "Malla / varias direcciones",
    DESCONOCIDA: "No clara",
}

_BIN_HALF_WIDTH = 22.0


@dataclass(frozen=True)
class GeometryResult:
    label: str
    display_name: str
    confidence: float
    angle_deg: float | None
    notes: str


def _to_gray(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim == 2:
        gray = image_rgb
    elif image_rgb.shape[2] == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb[:, :, 0]
    if gray.shape[0] > 720 or gray.shape[1] > 720:
        scale = 720 / max(gray.shape[:2])
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def _crack_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def _fold_angle_deg(dx: float, dy: float) -> float:
    ang = float(np.degrees(np.arctan2(dy, dx))) % 180.0
    return ang


def _bin_label(angle: float) -> str:
    a = angle % 180.0
    dist_h = min(a, 180.0 - a)
    dist_v = abs(a - 90.0)
    dist_d1 = abs(a - 45.0)
    dist_d2 = abs(a - 135.0)
    dist_d = min(dist_d1, dist_d2)
    best = min(dist_h, dist_v, dist_d)
    if best > _BIN_HALF_WIDTH:
        return DESCONOCIDA
    if best == dist_d:
        return INCLINADA
    if best == dist_v:
        return VERTICAL
    return HORIZONTAL


def _hough_votes(mask: np.ndarray) -> tuple[dict[str, float], float | None]:
    edges = cv2.Canny(mask, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=28,
        minLineLength=max(24, min(mask.shape[:2]) // 12),
        maxLineGap=12,
    )
    votes = {VERTICAL: 0.0, HORIZONTAL: 0.0, INCLINADA: 0.0}
    angles: list[tuple[float, float]] = []
    if lines is None:
        return votes, None
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 8:
            continue
        ang = _fold_angle_deg(float(x2 - x1), float(y2 - y1))
        label = _bin_label(ang)
        if label in votes:
            votes[label] += length
            angles.append((ang, length))
    if not angles:
        return votes, None
    total = sum(w for _, w in angles)
    mean = sum(a * w for a, w in angles) / total
    return votes, mean


def _pca_angle(mask: np.ndarray) -> float | None:
    ys, xs = np.where(mask > 0)
    if xs.size < 80:
        return None
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    _mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    vx, vy = float(eigenvectors[0, 0]), float(eigenvectors[0, 1])
    return _fold_angle_deg(vx, vy)


def analyze_orientation(image_rgb: np.ndarray, *, assume_crack: bool) -> GeometryResult:
    """Estima la orientación dominante. Si assume_crack es False, no fuerza un patrón."""
    if not assume_crack:
        return GeometryResult(
            DESCONOCIDA,
            DISPLAY_NAME[DESCONOCIDA],
            0.0,
            None,
            "La red no detecta fisura; no se interpreta la orientación.",
        )

    gray = _to_gray(np.asarray(image_rgb))
    mask = _crack_mask(gray)
    coverage = float(np.count_nonzero(mask)) / float(mask.size)
    votes, hough_angle = _hough_votes(mask)
    total = sum(votes.values())
    pca_ang = _pca_angle(mask)

    if total < 40 or coverage < 0.004:
        if pca_ang is None:
            return GeometryResult(
                DESCONOCIDA,
                DISPLAY_NAME[DESCONOCIDA],
                0.15,
                None,
                "No se encontró un trazo dominante en la foto.",
            )
        label = _bin_label(pca_ang)
        return GeometryResult(
            label,
            DISPLAY_NAME[label],
            0.40,
            round(pca_ang, 1),
            "Orientación aproximada (poca evidencia de líneas).",
        )

    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ranked[0]
    second = ranked[1][1]
    dominance = top_score / total if total else 0.0

    if second > 0 and top_score > 0 and (second / top_score) > 0.72 and dominance < 0.62:
        conf = min(0.85, 0.45 + coverage * 8)
        return GeometryResult(
            MALLA,
            DISPLAY_NAME[MALLA],
            round(conf, 3),
            round(hough_angle, 1) if hough_angle is not None else None,
            "Hay más de una dirección fuerte (posible malla o piel de cocodrilo).",
        )

    angle = hough_angle if hough_angle is not None else pca_ang
    conf = min(0.92, 0.50 + 0.45 * dominance)
    return GeometryResult(
        top_label,
        DISPLAY_NAME[top_label],
        round(float(conf), 3),
        round(float(angle), 1) if angle is not None else None,
        "Dirección estimada sobre el trazo más visible de la foto.",
    )
