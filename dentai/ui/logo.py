"""DENT-AI logosu (vektör, her boyutta net).

Lacivert yuvarlatılmış kare içinde beyaz azı dişi; dişin kronundan geçen turkuaz tarama/nabız
çizgisi röntgenin "ikinci göz" tarafından okunmasını, sağ üstteki ışıltı yapay zekâyı simgeler.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient

TEAL = "#2BB8A6"
TEAL_LIGHT = "#5FE0CF"
NAVY = "#12395B"
NAVY_DEEP = "#0B2740"


def tooth_path() -> QPainterPath:
    """Azı dişi silueti, (0,0) merkezli ~92×124 birim."""
    p = QPainterPath(QPointF(-40, -20))
    p.cubicTo(-46, -50, -22, -60, -10, -47)
    p.cubicTo(-4, -42, 4, -42, 10, -47)
    p.cubicTo(22, -60, 46, -50, 40, -20)
    p.cubicTo(38, 0, 34, 14, 30, 30)
    p.cubicTo(27, 44, 26, 58, 18, 62)
    p.cubicTo(10, 64, 8, 50, 6, 38)
    p.cubicTo(4, 30, -4, 30, -6, 38)
    p.cubicTo(-8, 50, -10, 64, -18, 62)
    p.cubicTo(-26, 58, -27, 44, -30, 30)
    p.cubicTo(-34, 14, -38, 0, -40, -20)
    p.closeSubpath()
    return p


def _spark(p: QPainter, c: QPointF, r: float, color: QColor) -> None:
    path = QPainterPath(QPointF(c.x(), c.y() - r))
    k = r * 0.2
    for dx, dy, ex, ey in ((k, -k, r, 0), (k, k, 0, r), (-k, k, -r, 0), (-k, -k, 0, -r)):
        path.quadTo(QPointF(c.x() + dx, c.y() + dy), QPointF(c.x() + ex, c.y() + ey))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawPath(path)


def draw_logo(p: QPainter, rect: QRectF, tile: bool = True, stroke: str = "#FFFFFF", alpha: float = 1.0) -> None:
    """tile=True: lacivert karo üzerinde (uygulama ikonu). tile=False: yalnızca diş işareti (stroke renginde)."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setOpacity(p.opacity() * alpha)
    s = min(rect.width(), rect.height())
    r = QRectF(rect.center().x() - s / 2, rect.center().y() - s / 2, s, s)
    if tile:
        g = QLinearGradient(r.topLeft(), r.bottomRight())
        g.setColorAt(0, QColor("#1B5283"))
        g.setColorAt(1, QColor(NAVY_DEEP))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawRoundedRect(r, s * 0.24, s * 0.24)
        glow = QRadialGradient(QPointF(r.right() - s * 0.15, r.bottom() - s * 0.1), s * 0.6)
        c = QColor(TEAL)
        c.setAlphaF(0.35)
        glow.setColorAt(0, c)
        c2 = QColor(TEAL)
        c2.setAlphaF(0.0)
        glow.setColorAt(1, c2)
        clip = QPainterPath()
        clip.addRoundedRect(r, s * 0.24, s * 0.24)
        p.setClipPath(clip)
        p.setBrush(glow)
        p.drawRect(r)
        p.setClipping(False)

    k = s * (0.0056 if tile else 0.0074)          # birim → piksel
    p.translate(r.center().x() - (s * 0.02 if tile else 0), r.center().y() + s * 0.03)
    p.scale(k, k)
    tooth = tooth_path()
    fill = QColor(stroke)
    fill.setAlphaF(0.10)
    p.setBrush(fill)
    p.setPen(QPen(QColor(stroke), 8.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.drawPath(tooth)

    # tarama / nabız çizgisi: kronun içinden geçer
    pulse = QPainterPath(QPointF(-27, -16))
    for x, y in ((-12, -16), (-6, -31), (2, -2), (8, -16), (27, -16)):
        pulse.lineTo(QPointF(x, y))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(TEAL_LIGHT if tile else TEAL), 7.5, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.drawPath(pulse)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(TEAL_LIGHT if tile else TEAL))
    for x in (-27, 27):
        p.drawEllipse(QPointF(x, -16), 6.5, 6.5)
    # yapay zekâ ışıltısı
    _spark(p, QPointF(48, -58), 15, QColor(TEAL_LIGHT if tile else TEAL))
    _spark(p, QPointF(62, -34), 7, QColor(stroke))
    p.restore()


def logo_pixmap(size: int, tile: bool = True) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    draw_logo(p, QRectF(0, 0, size, size), tile)
    p.end()
    return pm


def app_icon() -> QIcon:
    icon = QIcon()
    for sz in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(logo_pixmap(sz))
    return icon
