"""Görüntü G/Ç, kontrast iyileştirme ve oran koruyan letterbox.

cv2.imread / imwrite Windows'ta ASCII olmayan yollarda (ör. "Masaüstü") sessizce
başarısız olur; bu yüzden np.fromfile / tofile üzerinden okunup yazılır.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def imread(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"Görüntü kodlanamadı: {path}")
    buf.tofile(str(path))


def enhance_contrast(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE: düşük kontrastlı röntgende erken çürük kenarlarını belirginleştirir."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


@dataclass(frozen=True)
class LetterboxInfo:
    """letterbox() dönüşümünün tersini almak için gereken bilgi."""

    scale: float
    left: int
    top: int
    src_hw: tuple[int, int]
    dst_hw: tuple[int, int]

    def to_letterbox(self, xyxy: np.ndarray) -> np.ndarray:
        out = np.asarray(xyxy, dtype=np.float32).copy()
        out[..., [0, 2]] = out[..., [0, 2]] * self.scale + self.left
        out[..., [1, 3]] = out[..., [1, 3]] * self.scale + self.top
        return out

    def to_original(self, xyxy: np.ndarray) -> np.ndarray:
        out = np.asarray(xyxy, dtype=np.float32).copy()
        out[..., [0, 2]] = (out[..., [0, 2]] - self.left) / self.scale
        out[..., [1, 3]] = (out[..., [1, 3]] - self.top) / self.scale
        h, w = self.src_hw
        out[..., [0, 2]] = out[..., [0, 2]].clip(0, w)
        out[..., [1, 3]] = out[..., [1, 3]].clip(0, h)
        return out


def letterbox(
    img: np.ndarray,
    new_hw: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
    scale_up: bool = True,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Görüntüyü en-boy oranını bozmadan new_hw (yükseklik, genişlik) içine sığdırır, kalanı doldurur."""
    h, w = img.shape[:2]
    H, W = new_hw
    if h == 0 or w == 0 or H <= 0 or W <= 0:
        raise ValueError("Geçersiz görüntü ya da hedef boyut")

    r = min(H / h, W / w)
    if not scale_up:
        r = min(r, 1.0)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))

    if (nw, nh) != (w, h):
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (nw, nh), interpolation=interp)

    left, top = (W - nw) // 2, (H - nh) // 2
    right, bottom = W - nw - left, H - nh - top
    value = color if img.ndim == 3 else color[0]
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
    return out, LetterboxInfo(r, left, top, (h, w), (H, W))


def fit_to_screen(
    img: np.ndarray, max_w: int = 1600, max_h: int = 900, color: tuple[int, int, int] = (24, 24, 24)
) -> np.ndarray:
    """Ekrana sığdırılmış, oranı korunmuş önizleme (yalnızca küçültür, asla büyütmez)."""
    h, w = img.shape[:2]
    r = min(max_w / w, max_h / h, 1.0)
    target = (max(1, round(h * r)), max(1, round(w * r)))
    out, _ = letterbox(img, target, color=color, scale_up=False)
    return out
