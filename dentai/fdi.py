"""FDI (ISO 3950) diş numaralandırma yardımcıları. Yalnızca kalıcı dişler (32 sınıf)."""

from __future__ import annotations

# Sınıf id'si = indeks. 1x üst sağ, 2x üst sol, 3x alt sol, 4x alt sağ (hastanın sağı/solu).
FDI_NAMES: list[str] = [f"{q}{n}" for q in (1, 2, 3, 4) for n in range(1, 9)]
FDI_INDEX: dict[str, int] = {name: i for i, name in enumerate(FDI_NAMES)}

# Röntgende soldan sağa görünüm sırası (hastanın sağı ekranın solunda).
UPPER_ROW: list[str] = [f"1{n}" for n in range(8, 0, -1)] + [f"2{n}" for n in range(1, 9)]
LOWER_ROW: list[str] = [f"4{n}" for n in range(8, 0, -1)] + [f"3{n}" for n in range(1, 9)]


def is_upper(fdi: str) -> bool:
    return fdi[0] in "12"


def is_third_molar(fdi: str) -> bool:
    return fdi.endswith("8")
