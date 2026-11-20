"""Sonuçları görüntü üzerine çizer. Sınıf başına ayrı renk: configs/classes.yaml."""

from __future__ import annotations

import cv2
import numpy as np

from dentai.config import load_classes
from dentai.inference.results import AnalysisResult

_C = load_classes()
# Katman adı: "teeth", bir sınıf adı ("filling") veya bir grup adı ("restoration").
ALL_LAYERS = frozenset({"teeth", *_C.groups, *_C.finding_names})
_TR = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
_FALLBACK_BGR = (200, 200, 200)
_TEETH_BGR = (215, 215, 215)


def ascii_tr(text: str) -> str:
    """cv2.putText Hershey fontları Türkçe karakterleri çizemez."""
    return text.translate(_TR)


def _text(img, text, org, scale, color, thick, bg=None):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x, y - th - base), (x + tw + 4, y + base // 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - 1), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_analysis(
    img_bgr: np.ndarray,
    result: AnalysisResult,
    layers: set[str] | frozenset[str] = ALL_LAYERS,
    show_conf: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    s = max(w, h) / 1500
    thick = max(1, round(2 * s))
    fs = max(0.4, 0.5 * s)
    classes = load_classes()

    if "teeth" in layers:
        for t in result.teeth:
            x1, y1, x2, y2 = (int(v) for v in t.xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), _TEETH_BGR, 1, cv2.LINE_AA)
            upper = t.name[0] in "12"
            ty = y1 + int(16 * s) + 2 if upper else y2 - 6
            _text(out, t.name, (x1 + 1, ty), fs, (255, 255, 255), max(1, thick - 1), bg=(40, 40, 40))

    for det, fdi in zip(result.findings, result.tooth_of):
        if det.name not in layers and det.group not in layers:
            continue
        spec = classes.by_name(det.name)
        color = spec.color_bgr if spec else _FALLBACK_BGR
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
        if show_labels or show_conf:
            parts = []
            if show_labels:
                parts.append(ascii_tr(spec.label if spec else det.name))
                if fdi:
                    parts.insert(0, fdi)
            if show_conf:
                parts.append(f"{det.conf:.0%}")
            label = " ".join(parts)
            lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
            ty = y1 - 4 if y1 > 28 * s else y2 + int(18 * s)
            _text(out, label, (x1, ty), fs, (0, 0, 0) if lum > 140 else (255, 255, 255), max(1, thick - 1), bg=color)
    return out
