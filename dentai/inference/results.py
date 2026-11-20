from __future__ import annotations

from dataclasses import dataclass, field

DISCLAIMER = "Bu çıktı bir karar destek aracıdır; tanı yerine geçmez. Nihai karar hekime aittir."


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    name: str            # teeth: FDI kodu ("36"), findings: sınıf adı ("caries_early")
    conf: float
    group: str = ""      # findings: caries | restoration | impacted

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class RawResult:
    """Güven eşiği uygulanmamış (düşük eşikli) ham model çıktısı."""

    teeth: list[Detection]
    findings: list[Detection]
    age: float | None
    size: tuple[int, int]            # (genişlik, yükseklik)
    elapsed_ms: float = 0.0


@dataclass
class AnalysisResult:
    teeth: list[Detection]
    findings: list[Detection]
    tooth_of: list[str | None]       # findings[i] hangi FDI dişine ait
    age: float | None
    size: tuple[int, int]
    elapsed_ms: float = 0.0
    conf_threshold: float = 0.25

    def by_tooth(self) -> dict[str, list[Detection]]:
        out: dict[str, list[Detection]] = {}
        for det, fdi in zip(self.findings, self.tooth_of):
            if fdi:
                out.setdefault(fdi, []).append(det)
        return out

    def counts(self) -> dict[str, int]:
        c = {"caries": 0, "restoration": 0, "impacted": 0}
        for d in self.findings:
            if d.group in c:
                c[d.group] += 1
        return c

    def to_dict(self, image_name: str = "") -> dict:
        return {
            "image": image_name,
            "size": {"width": self.size[0], "height": self.size[1]},
            "estimated_dental_age_years": None if self.age is None else round(self.age, 1),
            "counts": self.counts(),
            "teeth": [{"fdi": t.name, "conf": round(t.conf, 3), "box": [round(v, 1) for v in t.xyxy]}
                      for t in self.teeth],
            "findings": [
                {"type": d.name, "group": d.group, "tooth": fdi, "conf": round(d.conf, 3),
                 "box": [round(v, 1) for v in d.xyxy]}
                for d, fdi in zip(self.findings, self.tooth_of)
            ],
            "confidence_threshold": self.conf_threshold,
            "disclaimer": DISCLAIMER,
        }
