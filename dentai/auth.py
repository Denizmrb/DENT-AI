"""Uygulama girişi. Şifre düz metin olarak değil, PBKDF2 özeti olarak tutulur."""

from __future__ import annotations

import hashlib
import hmac
import re

_ITER = 200_000
# kullanıcı adı -> (tuz, PBKDF2-SHA256 özeti)
USERS: dict[str, tuple[bytes, str]] = {
    "deniz123456": (b"dentai-deniz123456-2026", "e48eda876c21e7f1eb6e69385e24930fa086da3bae06cbab2b748129af843335"),
}


def check_login(username: str, password: str) -> bool:
    entry = USERS.get(username.strip().lower())
    # kullanıcı yoksa da aynı hesabı yap: yanıt süresi hangi kullanıcıların var olduğunu ele vermesin
    salt, digest = entry if entry else (b"-", "0" * 64)
    got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITER).hex()
    return entry is not None and hmac.compare_digest(got, digest)


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
