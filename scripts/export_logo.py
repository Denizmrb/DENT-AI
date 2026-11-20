"""DENT-AI logosunu dosyaya yazar (sunum, README, masaüstü kısayolu için).

    python scripts/export_logo.py      →  assets/dentai_logo.png (1024), dentai_logo_512.png, dentai.ico,
                                            dentai_logo_yazili.png (logo + yazı, yatay)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from dentai.ui.logo import draw_logo, logo_pixmap  # noqa: E402


def main() -> None:
    QApplication([])
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    logo_pixmap(1024).save(str(out / "dentai_logo.png"))
    logo_pixmap(512).save(str(out / "dentai_logo_512.png"))
    logo_pixmap(256).save(str(out / "dentai.ico"))

    img = QImage(1400, 420, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw_logo(p, QRectF(40, 50, 320, 320))
    f = QFont("Segoe UI", 110)
    f.setWeight(QFont.Weight.ExtraBold)
    p.setFont(f)
    p.setPen(QColor("#12395B"))
    p.drawText(QPointF(410, 255), "DENT-")
    w = p.fontMetrics().horizontalAdvance("DENT-")
    p.setPen(QColor("#2BB8A6"))
    p.drawText(QPointF(410 + w, 255), "AI")
    f2 = QFont("Segoe UI", 30)
    p.setFont(f2)
    p.setPen(QColor("#6B7C8C"))
    p.drawText(QPointF(418, 330), "Panoramik röntgen için ikinci göz")
    p.end()
    img.save(str(out / "dentai_logo_yazili.png"))
    print("kaydedildi:", *sorted(x.name for x in out.iterdir()))


if __name__ == "__main__":
    main()
