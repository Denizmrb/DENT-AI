"""Hasta kayıtları, hekim notları ve arayüz ayarları (yerel JSON dosyaları, kayitlar/ klasörü).

Röntgenler kopyalanmaz; kayıtta dosyanın yolu tutulur. Notlar röntgen dosyasının yoluna bağlıdır:
aynı röntgen yeniden açıldığında genel not ve dişlere yazılan notlar geri gelir.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from dentai.config import ROOT

DATA_DIR = ROOT / "kayitlar"
GENERAL = "genel"   # röntgenin geneline yazılan notun anahtarı


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError):
        # bozuk dosyayı ezmeden kenara al, boş başla
        try:
            path.replace(path.with_suffix(f".bozuk-{datetime.now():%Y%m%d%H%M%S}.json"))
        except OSError:
            pass
        return default


def _write(path: Path, data) -> None:
    """Önce geçici dosyaya yazar, sonra yer değiştirir: yarıda kesilirse eski kayıt bozulmaz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def image_key(path: str | Path) -> str:
    p = Path(path)
    try:
        p = p.resolve()
    except OSError:
        pass
    return os.path.normcase(str(p))


@dataclass
class Patient:
    id: str
    name: str
    birth_year: str = ""
    created: str = field(default_factory=_now)
    xrays: list[str] = field(default_factory=list)

    @property
    def latest_xray(self) -> Path | None:
        return Path(self.xrays[-1]) if self.xrays else None


class PatientStore:
    def __init__(self, path: Path | None = None):
        self.path = path or DATA_DIR / "hastalar.json"
        raw = _read(self.path, {"patients": []})
        self.patients: list[Patient] = [Patient(**p) for p in raw.get("patients", [])]

    def save(self) -> None:
        _write(self.path, {"patients": [asdict(p) for p in self.patients]})

    def get(self, pid: str) -> Patient | None:
        return next((p for p in self.patients if p.id == pid), None)

    def add(self, name: str, birth_year: str = "", xray: str | Path | None = None) -> Patient:
        name = " ".join(name.split())
        if not name:
            raise ValueError("Hasta adı boş olamaz.")
        p = Patient(uuid.uuid4().hex[:12], name, birth_year.strip())
        if xray:
            p.xrays.append(str(Path(xray)))
        self.patients.append(p)
        self.save()
        return p

    def remove(self, pid: str) -> None:
        self.patients = [p for p in self.patients if p.id != pid]
        self.save()

    def attach_xray(self, pid: str, xray: str | Path) -> bool:
        """Röntgeni hastaya ekler; zaten varsa listenin sonuna (en güncel) taşır. Değişiklik olduysa True."""
        p = self.get(pid)
        if p is None:
            return False
        key = image_key(xray)
        before = list(p.xrays)
        p.xrays = [x for x in p.xrays if image_key(x) != key] + [str(Path(xray))]
        if p.xrays != before:
            self.save()
            return True
        return False


class NotesStore:
    """{röntgen anahtarı: {"path": ..., "notes": {"genel" | FDI: metin}, "updated": ...}}"""

    def __init__(self, path: Path | None = None):
        self.path = path or DATA_DIR / "notlar.json"
        self.data: dict = _read(self.path, {})

    def get(self, image: str | Path, target: str = GENERAL) -> str:
        return self.data.get(image_key(image), {}).get("notes", {}).get(target, "")

    def set(self, image: str | Path, target: str, text: str) -> None:
        key = image_key(image)
        entry = self.data.setdefault(key, {"path": str(image), "notes": {}})
        text = text.rstrip()
        if text:
            entry["notes"][target] = text
        else:
            entry["notes"].pop(target, None)
        entry["updated"] = _now()
        if not entry["notes"]:
            self.data.pop(key, None)
        _write(self.path, self.data)

    def noted_teeth(self, image: str | Path) -> set[str]:
        notes = self.data.get(image_key(image), {}).get("notes", {})
        return {k for k in notes if k != GENERAL}


def load_settings() -> dict:
    return _read(DATA_DIR / "ayarlar.json", {})


def save_setting(key: str, value) -> None:
    s = load_settings()
    s[key] = value
    try:
        _write(DATA_DIR / "ayarlar.json", s)
    except OSError:
        pass
