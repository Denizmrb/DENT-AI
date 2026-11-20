"""Uygulama girişi.

Kullanıcılar kodda değil, git'e girmeyen kayitlar/kullanicilar.json dosyasında tutulur.
Şifreler düz metin olarak değil, kullanıcıya özel rastgele tuzla PBKDF2-SHA256 özeti olarak saklanır.

Kullanıcı ekleme / şifre değiştirme (şifre ekranda görünmeden sorulur):

    python -m dentai.auth ekle <kullanici_adi>
    python -m dentai.auth sil <kullanici_adi>
    python -m dentai.auth liste
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sys
from pathlib import Path

from dentai.records import DATA_DIR

USERS_FILE = DATA_DIR / "kullanicilar.json"
_ITER = 200_000


def _path(path: Path | None) -> Path:
    return Path(path) if path is not None else USERS_FILE


def _norm(username: str) -> str:
    return username.strip().lower()


def load_users(path: Path | None = None) -> dict:
    try:
        return json.loads(_path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def has_users(path: Path | None = None) -> bool:
    return bool(load_users(path))


def _digest(password: str, salt: bytes, iterations: int) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations).hex()


def set_user(username: str, password: str, path: Path | None = None) -> None:
    """Kullanıcı ekler; varsa şifresini değiştirir."""
    name = _norm(username)
    if not name:
        raise ValueError("Kullanıcı adı boş olamaz.")
    if not password:
        raise ValueError("Şifre boş olamaz.")
    users = load_users(path)
    salt = secrets.token_bytes(16)
    users[name] = {"salt": salt.hex(), "iter": _ITER, "hash": _digest(password, salt, _ITER)}
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def remove_user(username: str, path: Path | None = None) -> bool:
    users = load_users(path)
    if users.pop(_norm(username), None) is None:
        return False
    _path(path).write_text(json.dumps(users, indent=2), encoding="utf-8")
    return True


def check_login(username: str, password: str, path: Path | None = None) -> bool:
    entry = load_users(path).get(_norm(username))
    # kullanıcı yoksa da aynı hesabı yap: yanıt süresi hangi kullanıcıların var olduğunu ele vermesin
    salt = bytes.fromhex(entry["salt"]) if entry else b"-"
    iters = int(entry.get("iter", _ITER)) if entry else _ITER
    got = _digest(password, salt, iters)
    return entry is not None and hmac.compare_digest(got, entry["hash"])


def password_strength(pw: str) -> int:
    """0 (boş/çok zayıf) … 4 (güçlü)."""
    if not pw:
        return 0
    score = 0
    score += len(pw) >= 8
    score += len(pw) >= 12
    kinds = sum(bool(re.search(p, pw)) for p in (r"[a-zçğıöşü]", r"[A-ZÇĞİÖŞÜ]", r"\d", r"[^\w\s]"))
    score += kinds >= 3
    score += kinds == 4 or (kinds >= 3 and len(pw) >= 10)
    return max(1, min(4, score))


STRENGTH_LABELS = ["", "Zayıf", "Orta", "İyi", "Güçlü"]


def _cli(argv: list[str]) -> int:
    from getpass import getpass

    if len(argv) >= 2 and argv[0] == "ekle":
        name = argv[1]
        pw = getpass(f"'{_norm(name)}' için yeni şifre: ")
        if pw != getpass("Şifre (tekrar): "):
            print("Şifreler aynı değil, değişiklik yapılmadı.")
            return 1
        existed = _norm(name) in load_users()
        set_user(name, pw)
        print(f"{'Şifre değiştirildi' if existed else 'Kullanıcı eklendi'}: {_norm(name)}  "
              f"(güç: {STRENGTH_LABELS[password_strength(pw)]})  →  {USERS_FILE}")
        return 0
    if len(argv) >= 2 and argv[0] == "sil":
        ok = remove_user(argv[1])
        print("Silindi." if ok else "Böyle bir kullanıcı yok.")
        return 0 if ok else 1
    if argv[:1] == ["liste"]:
        users = load_users()
        print("\n".join(users) if users else "Kayıtlı kullanıcı yok.")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
