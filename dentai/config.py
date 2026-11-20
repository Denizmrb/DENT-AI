from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
MODELS_DIR = ROOT / "models"

DEFAULT_WEIGHTS = {
    "teeth": MODELS_DIR / "teeth_best.pt",
    "findings": MODELS_DIR / "findings_best.pt",
    "age": MODELS_DIR / "age_best.pt",
}


@dataclass(frozen=True)
class GroupSpec:
    key: str
    label: str
    color_rgb: tuple[int, int, int]

    @property
    def color_bgr(self) -> tuple[int, int, int]:
        r, g, b = self.color_rgb
        return (b, g, r)


@dataclass(frozen=True)
class ClassSpec:
    id: int
    name: str
    label: str
    group: str
    color_rgb: tuple[int, int, int]

    @property
    def color_bgr(self) -> tuple[int, int, int]:
        r, g, b = self.color_rgb
        return (b, g, r)


@dataclass(frozen=True)
class Classes:
    groups: dict[str, GroupSpec]
    findings: list[ClassSpec]

    @property
    def finding_names(self) -> list[str]:
        return [c.name for c in self.findings]

    def by_name(self, name: str) -> ClassSpec | None:
        return next((c for c in self.findings if c.name == name), None)


@lru_cache(maxsize=1)
def load_classes() -> Classes:
    raw = yaml.safe_load((CONFIG_DIR / "classes.yaml").read_text(encoding="utf-8"))
    groups = {
        k: GroupSpec(k, v["label"], tuple(v["color"])) for k, v in raw["groups"].items()
    }
    findings = [
        ClassSpec(i, c["name"], c["label"], c["group"], tuple(c["color"]))
        for i, c in enumerate(raw["findings"])
    ]
    return Classes(groups, findings)


def load_augment(task: str) -> dict:
    raw = yaml.safe_load((CONFIG_DIR / "augment.yaml").read_text(encoding="utf-8"))
    if task not in raw:
        raise KeyError(f"augment.yaml içinde '{task}' bölümü yok (teeth|findings)")
    return dict(raw[task])
