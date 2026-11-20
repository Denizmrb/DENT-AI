"""İki aşamalı + yaş regresörlü çıkarım hattı.

  teeth modeli    -> 32 FDI sınıfı (diş kutuları + numara)
  findings modeli -> çürük / dolgu / kuron / implant / gömülü diş (+ pozisyon)
  age modeli      -> yıl cinsinden dental yaş
  birleştirme     -> her bulgu, kutusu en çok örtüştüğü dişe (FDI) atanır
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from dentai.config import DEFAULT_WEIGHTS, load_classes
from dentai.imaging import enhance_contrast
from dentai.inference.results import AnalysisResult, Detection, RawResult

MIN_IOA = 0.25   # bulgu alanının en az bu kadarı dişin içinde olmalı


def _load_meta(weights: Path) -> dict:
    p = weights.with_suffix(".json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _intersection(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def dedupe_teeth(teeth: list[Detection]) -> list[Detection]:
    """Aynı FDI numarası birden fazla çıkarsa en güvenilir olanı tutar."""
    best: dict[str, Detection] = {}
    for t in teeth:
        if t.name not in best or t.conf > best[t.name].conf:
            best[t.name] = t
    return sorted(best.values(), key=lambda t: t.center[0])


def assign_findings(findings: list[Detection], teeth: list[Detection]) -> list[str | None]:
    out: list[str | None] = []
    for f in findings:
        best, best_ioa = None, 0.0
        for t in teeth:
            ioa = _intersection(f.xyxy, t.xyxy) / f.area if f.area else 0.0
            if ioa > best_ioa:
                best, best_ioa = t.name, ioa
        out.append(best if best_ioa >= MIN_IOA else None)
    return out


class DentalAnalyzer:
    def __init__(self, teeth: str | Path | None = DEFAULT_WEIGHTS["teeth"],
                 findings: str | Path | None = DEFAULT_WEIGHTS["findings"],
                 age: str | Path | None = DEFAULT_WEIGHTS["age"],
                 device: str | None = None, imgsz: int | None = None, raw_conf: float = 0.10,
                 iou: float = 0.5, clahe: bool | None = None):
        self.paths = {k: (Path(p) if p and Path(p).exists() else None)
                      for k, p in (("teeth", teeth), ("findings", findings), ("age", age))}
        self.device, self.imgsz_override, self.raw_conf, self.iou = device, imgsz, raw_conf, iou
        self.clahe_override = clahe
        self._models: dict = {}
        self._loaded = False

    # ---- yükleme -----------------------------------------------------------------
    @property
    def available(self) -> dict[str, bool]:
        return {k: v is not None for k, v in self.paths.items()}

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        if self._loaded:
            return
        from ultralytics import YOLO

        self.device = self._resolve_device()
        for key in ("teeth", "findings"):
            if self.paths[key]:
                self._models[key] = YOLO(str(self.paths[key]))
        if self.paths["age"]:
            from dentai.age.model import AgeEstimator

            self._models["age"] = AgeEstimator(self.paths["age"], self.device)
        self._loaded = True

    # ---- çıkarım -----------------------------------------------------------------
    def _detect(self, key: str, img: np.ndarray) -> list[Detection]:
        model = self._models.get(key)
        if model is None:
            return []
        meta = _load_meta(self.paths[key])
        clahe = self.clahe_override if self.clahe_override is not None else meta.get("clahe", False)
        imgsz = self.imgsz_override or meta.get("imgsz", 640)
        src = enhance_contrast(img) if clahe else img
        res = model.predict(src, imgsz=imgsz, conf=self.raw_conf, iou=self.iou, device=self.device,
                            verbose=False, agnostic_nms=(key == "teeth"), max_det=300)[0]
        b = res.boxes
        if b is None or len(b) == 0:
            return []
        classes = load_classes()
        dets = []
        for xyxy, conf, cls in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()):
            name = str(res.names[int(cls)])
            group = ""
            if key == "findings":
                spec = classes.by_name(name)
                group = spec.group if spec else "other"
            dets.append(Detection(tuple(float(v) for v in xyxy), name, float(conf), group))
        return dets

    def infer(self, img_bgr: np.ndarray) -> RawResult:
        self.load()
        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]
        age = self._models["age"].predict(img_bgr) if "age" in self._models else None
        return RawResult(
            teeth=self._detect("teeth", img_bgr),
            findings=self._detect("findings", img_bgr),
            age=age,
            size=(w, h),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    @staticmethod
    def assemble(raw: RawResult, conf: float = 0.25) -> AnalysisResult:
        """Eşik + birleştirme. Ucuzdur; UI'daki eşik kaydırıcısı bunu yeniden çağırır."""
        teeth = dedupe_teeth([t for t in raw.teeth if t.conf >= conf])
        findings = sorted((f for f in raw.findings if f.conf >= conf), key=lambda f: f.xyxy[0])
        return AnalysisResult(teeth, findings, assign_findings(findings, teeth), raw.age,
                              raw.size, raw.elapsed_ms, conf)

    def analyze(self, img_bgr: np.ndarray, conf: float = 0.25) -> AnalysisResult:
        return self.assemble(self.infer(img_bgr), conf)
