"""Animasyonlu, kendini çizen arayüz kontrolleri: ikonlar, düğmeler, çipler, kaydırıcı, tema anahtarı, açılır panel."""

from __future__ import annotations

import math

from PySide6.QtCore import QEasingCurve, QPoint, QPointF, QPropertyAnimation, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractButton, QFrame, QGraphicsDropShadowEffect, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from dentai.ui import theme


def qc(hex_or_color, alpha: float | None = None) -> QColor:
    c = QColor(hex_or_color)
    if alpha is not None:
        c.setAlphaF(max(0.0, min(1.0, alpha)))
    return c


def mix(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor.fromRgbF(a.redF() + (b.redF() - a.redF()) * t, a.greenF() + (b.greenF() - a.greenF()) * t,
                           a.blueF() + (b.blueF() - a.blueF()) * t, a.alphaF() + (b.alphaF() - a.alphaF()) * t)


# ---------------------------------------------------------------------------------------------------------------
# İkonlar (vektör, her boyutta net)
# ---------------------------------------------------------------------------------------------------------------
def draw_icon(p: QPainter, name: str, r: QRectF, color: QColor, width: float = 1.8) -> None:
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    s = min(r.width(), r.height())
    c = r.center()
    u = s / 20.0                       # 20×20 birimlik ızgara

    def P(x, y):
        return QPointF(c.x() + (x - 10) * u, c.y() + (y - 10) * u)

    if name in ("plus", "minus", "zoom_in", "zoom_out"):
        if name.startswith("zoom"):
            p.drawEllipse(P(8.5, 8.5), 6 * u, 6 * u)
            p.drawLine(P(13, 13), P(17.5, 17.5))
            p.drawLine(P(5.8, 8.5), P(11.2, 8.5))
            if name == "zoom_in":
                p.drawLine(P(8.5, 5.8), P(8.5, 11.2))
        else:
            p.drawLine(P(4, 10), P(16, 10))
            if name == "plus":
                p.drawLine(P(10, 4), P(10, 16))
    elif name == "fit":
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            cx, cy = 10 - 6.5 * sx, 10 - 6.5 * sy
            p.drawLine(P(cx, cy), P(cx + 4 * sx, cy))
            p.drawLine(P(cx, cy), P(cx, cy + 4 * sy))
        p.drawEllipse(P(10, 10), 1.6 * u, 1.6 * u)
    elif name == "upload":
        p.drawLine(P(10, 13), P(10, 3.5))
        p.drawLine(P(6, 7.5), P(10, 3.5))
        p.drawLine(P(14, 7.5), P(10, 3.5))
        path = QPainterPath(P(3.5, 12))
        path.lineTo(P(3.5, 16.5))
        path.lineTo(P(16.5, 16.5))
        path.lineTo(P(16.5, 12))
        p.drawPath(path)
    elif name == "download":
        p.drawLine(P(10, 3.5), P(10, 12.5))
        p.drawLine(P(6, 8.5), P(10, 12.5))
        p.drawLine(P(14, 8.5), P(10, 12.5))
        path = QPainterPath(P(3.5, 12))
        path.lineTo(P(3.5, 16.5))
        path.lineTo(P(16.5, 16.5))
        path.lineTo(P(16.5, 12))
        p.drawPath(path)
    elif name == "sliders":
        for y, k in ((5, 13), (10, 6.5), (15, 11)):
            p.drawLine(P(3, y), P(17, y))
            p.setBrush(theme_panel())
            p.drawEllipse(P(k, y), 1.9 * u, 1.9 * u)
            p.setBrush(Qt.BrushStyle.NoBrush)
    elif name == "sun":
        p.drawEllipse(P(10, 10), 3.4 * u, 3.4 * u)
        for i in range(8):
            a = i * math.pi / 4
            p.drawLine(P(10 + 6 * math.cos(a), 10 + 6 * math.sin(a)), P(10 + 8 * math.cos(a), 10 + 8 * math.sin(a)))
    elif name == "moon":
        path = QPainterPath()
        path.addEllipse(P(10, 10), 7 * u, 7 * u)
        cut = QPainterPath()
        cut.addEllipse(P(14, 6.5), 6 * u, 6 * u)
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(path.subtracted(cut))
    elif name == "contrast":
        p.drawEllipse(P(10, 10), 7 * u, 7 * u)
        half = QPainterPath(P(10, 3))
        half.arcTo(QRectF(P(3, 3), P(17, 17)), 90, 180)
        half.closeSubpath()
        p.setBrush(color)
        p.drawPath(half)
    elif name == "brightness":
        p.drawEllipse(P(10, 10), 3 * u, 3 * u)
        for i in range(8):
            a = i * math.pi / 4
            p.drawLine(P(10 + 5.6 * math.cos(a), 10 + 5.6 * math.sin(a)), P(10 + 7.4 * math.cos(a), 10 + 7.4 * math.sin(a)))
    elif name == "play":
        path = QPainterPath(P(6.5, 4))
        path.lineTo(P(16, 10))
        path.lineTo(P(6.5, 16))
        path.closeSubpath()
        p.setBrush(color)
        p.drawPath(path)
    elif name == "user_plus":
        p.drawEllipse(P(8, 6.5), 3.2 * u, 3.2 * u)
        arc = QPainterPath(P(2, 17))
        arc.cubicTo(P(2, 11), P(14, 11), P(14, 17))
        p.drawPath(arc)
        p.drawLine(P(16.5, 5), P(16.5, 11))
        p.drawLine(P(13.5, 8), P(19.5, 8))
    elif name == "eye" or name == "eye_off":
        path = QPainterPath(P(2, 10))
        path.quadTo(P(10, 2), P(18, 10))
        path.quadTo(P(10, 18), P(2, 10))
        p.drawPath(path)
        p.drawEllipse(P(10, 10), 2.6 * u, 2.6 * u)
        if name == "eye_off":
            p.drawLine(P(3.5, 16.5), P(16.5, 3.5))
    elif name == "logout":
        p.drawLine(P(8, 10), P(17, 10))
        p.drawLine(P(14, 7), P(17, 10))
        p.drawLine(P(14, 13), P(17, 10))
        path = QPainterPath(P(11, 4))
        path.lineTo(P(4, 4))
        path.lineTo(P(4, 16))
        path.lineTo(P(11, 16))
        p.drawPath(path)
    elif name == "tooth":
        path = QPainterPath(P(4, 6))
        path.cubicTo(P(4, 2.5), P(8, 2.5), P(10, 4))
        path.cubicTo(P(12, 2.5), P(16, 2.5), P(16, 6))
        path.cubicTo(P(16, 10), P(14.5, 12), P(14, 15))
        path.cubicTo(P(13.5, 18), P(11.8, 18), P(11.5, 15))
        path.cubicTo(P(11.2, 12.5), P(8.8, 12.5), P(8.5, 15))
        path.cubicTo(P(8.2, 18), P(6.5, 18), P(6, 15))
        path.cubicTo(P(5.5, 12), P(4, 10), P(4, 6))
        p.drawPath(path)
    elif name == "check":
        path = QPainterPath(P(4.5, 10.5))
        path.lineTo(P(8.5, 14.5))
        path.lineTo(P(15.5, 5.5))
        p.drawPath(path)
    p.restore()


def theme_panel() -> QColor:
    return QColor(theme.PANEL)


# ---------------------------------------------------------------------------------------------------------------
# Basit animasyon motoru: değerler hedeflerine yumuşakça yaklaşır, oturunca zamanlayıcı durur
# ---------------------------------------------------------------------------------------------------------------
class _Anim:
    def _anim_init(self, **values):
        self._v = dict(values)
        self._t = dict(values)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._anim_tick)

    def target(self, **kw):
        self._t.update(kw)
        if not self._timer.isActive():
            self._timer.start()

    def v(self, key: str) -> float:
        return self._v[key]

    def _anim_extra(self) -> bool:      # alt sınıflar sürekli animasyonlar için True döndürür
        return False

    def _anim_tick(self):
        busy = self._anim_extra()
        for k, tgt in self._t.items():
            cur = self._v[k]
            nxt = cur + (tgt - cur) * 0.22
            if abs(nxt - tgt) < 0.004:
                nxt = tgt
            else:
                busy = True
            self._v[k] = nxt
        self.update()
        if not busy:
            self._timer.stop()


# ---------------------------------------------------------------------------------------------------------------
# Düğme
# ---------------------------------------------------------------------------------------------------------------
class AnimatedButton(QPushButton, _Anim):
    """Üzerine gelince renk yumuşakça değişir ve hafifçe yükselir, basınca küçülür ve dalga (ripple) yayılır.
    kind: "ghost" (çerçeveli), "primary" (dolu ana renk), "icon" (yalnızca ikon, kare)."""

    def __init__(self, text: str = "", icon: str | None = None, kind: str = "ghost", tooltip: str = ""):
        super().__init__(text)
        self._icon, self.kind = icon, kind
        self._anim_init(hover=0.0, press=0.0, check=0.0)
        self._ripple: QPointF | None = None
        self._ripple_t = 1.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.toggled.connect(lambda on: self.target(check=1.0 if on else 0.0))

    def set_icon_name(self, name: str) -> None:
        self._icon = name
        self.update()

    def sizeHint(self) -> QSize:
        h = 38
        if self.kind == "icon" or not self.text():
            return QSize(h, h)
        f = QFont(self.font())
        f.setWeight(QFont.Weight.DemiBold)
        w = QFontMetrics(f).horizontalAdvance(self.text()) + 30 + (24 if self._icon else 0)
        return QSize(w, h)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def enterEvent(self, e):
        self.target(hover=1.0)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.target(hover=0.0, press=0.0)
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._ripple, self._ripple_t = e.position(), 0.0
            self.target(press=1.0)
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        self.target(press=0.0)
        super().mouseReleaseEvent(e)

    def _anim_extra(self) -> bool:
        if self._ripple_t < 1.0:
            self._ripple_t = min(1.0, self._ripple_t + 0.045)
            return True
        return False

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        hv, pr, ck = self.v("hover"), self.v("press"), self.v("check")
        en = self.isEnabled()
        scale = 1.0 - 0.04 * pr
        lift = -1.2 * hv * (1 - pr)
        r = QRectF(self.rect()).adjusted(1.5, 2.5, -1.5, -1.5)
        w, h = r.width() * scale, r.height() * scale
        r = QRectF(r.center().x() - w / 2, r.center().y() - h / 2 + lift, w, h)
        radius = 10

        primary = qc(theme.PRIMARY)
        if self.kind == "primary":
            bg = mix(primary, qc(theme.PRIMARY_HOVER), hv)
            fg = qc("#FFFFFF")
            border = Qt.PenStyle.NoPen
            if not en:
                bg = mix(primary, qc(theme.PANEL), 0.55)
        else:
            bg = mix(qc(theme.PANEL), qc(theme.PRIMARY_SOFT), max(hv, ck))
            fg = mix(qc(theme.TEXT), qc(theme.PRIMARY) if not theme.DARK_MODE else qc(theme.NAVY), max(hv * 0.8, ck))
            border = QPen(mix(qc(theme.BORDER), primary, max(hv, ck)), 1)
            if not en:
                fg = qc(theme.MUTED, 0.6)
                bg = qc(theme.PANEL_HI)
        # gölge
        if hv > 0.01 and en:
            sh = QColor(theme.NAVY if not theme.DARK_MODE else "#000000")
            sh.setAlphaF(0.10 * hv)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(sh)
            p.drawRoundedRect(r.translated(0, 2.5), radius, radius)
        p.setBrush(bg)
        p.setPen(border)
        p.drawRoundedRect(r, radius, radius)
        # dalga
        if self._ripple is not None and self._ripple_t < 1.0:
            clip = QPainterPath()
            clip.addRoundedRect(r, radius, radius)
            p.save()
            p.setClipPath(clip)
            rc = QColor("#FFFFFF" if self.kind == "primary" else theme.PRIMARY)
            rc.setAlphaF(0.28 * (1 - self._ripple_t))
            p.setBrush(rc)
            p.setPen(Qt.PenStyle.NoPen)
            rad = max(r.width(), r.height()) * 1.2 * self._ripple_t
            p.drawEllipse(self._ripple, rad, rad)
            p.restore()
        # içerik
        f = QFont(self.font())
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        text = self.text()
        isz = 18
        if text and self.kind != "icon":
            tw = QFontMetrics(f).horizontalAdvance(text)
            total = tw + (isz + 7 if self._icon else 0)
            x = r.center().x() - total / 2
            if self._icon:
                draw_icon(p, self._icon, QRectF(x, r.center().y() - isz / 2, isz, isz), fg)
                x += isz + 7
            p.setPen(fg)
            p.drawText(QRectF(x, r.top(), tw + 2, r.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
        elif self._icon:
            isz = 18 + 1.5 * hv
            draw_icon(p, self._icon, QRectF(r.center().x() - isz / 2, r.center().y() - isz / 2, isz, isz), fg)


# ---------------------------------------------------------------------------------------------------------------
# Görünüm çipi (renk noktalı açma/kapama düğmesi)
# ---------------------------------------------------------------------------------------------------------------
class ToggleChip(QAbstractButton, _Anim):
    def __init__(self, text: str, color: str | None = None, checked: bool = True):
        super().__init__()
        self.setText(text)
        self.setCheckable(True)
        self.setChecked(checked)
        self._color = color
        self._anim_init(hover=0.0, check=1.0 if checked else 0.0)
        self.toggled.connect(lambda on: self.target(check=1.0 if on else 0.0))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(QFontMetrics(self.font()).horizontalAdvance(self.text()) + 46, 34)

    def enterEvent(self, e):
        self.target(hover=1.0)

    def leaveEvent(self, e):
        self.target(hover=0.0)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        ck, hv = self.v("check"), self.v("hover")
        accent = qc(self._color or theme.PRIMARY)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        soft = QColor(accent)
        soft.setAlphaF(0.16 if not theme.DARK_MODE else 0.24)
        bg = mix(qc(theme.PANEL), soft, ck)
        if hv > 0:
            bg = mix(bg, qc(theme.PRIMARY_SOFT), 0.5 * hv * (1 - ck))
        p.setBrush(bg)
        p.setPen(QPen(mix(qc(theme.BORDER), accent, max(ck, hv * 0.6)), 1.2))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        # nokta → işaretli olunca dolar ve içinde tik belirir
        cx, cy, rad = r.left() + 17, r.center().y(), 7.0
        p.setPen(QPen(accent, 1.6))
        p.setBrush(mix(qc(theme.PANEL), accent, ck))
        p.drawEllipse(QPointF(cx, cy), rad, rad)
        if ck > 0.05:
            path = QPainterPath(QPointF(cx - 3.2, cy + 0.2))
            path.lineTo(QPointF(cx - 0.9, cy + 2.6))
            path.lineTo(QPointF(cx + 3.4, cy - 2.6))
            p.setPen(QPen(qc("#FFFFFF", ck), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        p.setPen(mix(qc(theme.MUTED), qc(theme.TEXT), 0.4 + 0.6 * ck))
        f = QFont(self.font())
        if ck > 0.5:
            f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.drawText(r.adjusted(31, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())


# ---------------------------------------------------------------------------------------------------------------
# Kaydırıcı
# ---------------------------------------------------------------------------------------------------------------
class AnimSlider(QSlider, _Anim):
    """Tıklanan yere atlar; tutamaç üzerine gelince büyür. Çift tık varsayılan değere döner."""

    def __init__(self, lo: int, hi: int, value: int, default: int | None = None, centered: bool = False):
        super().__init__(Qt.Orientation.Horizontal)
        self.setRange(lo, hi)
        self.setValue(value)
        self.default = value if default is None else default
        self.centered = centered          # dolgu varsayılandan başlar (parlaklık gibi ± değerler için)
        self._anim_init(hover=0.0, press=0.0, pos=float(value))
        self.valueChanged.connect(lambda v: self.target(pos=float(v)))
        self.setMinimumHeight(24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def enterEvent(self, e):
        self.target(hover=1.0)

    def leaveEvent(self, e):
        self.target(hover=0.0)

    def _x_to_value(self, x: float) -> int:
        m = 9
        t = (x - m) / max(1.0, self.width() - 2 * m)
        return round(self.minimum() + max(0.0, min(1.0, t)) * (self.maximum() - self.minimum()))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(True)
            self.setValue(self._x_to_value(e.position().x()))
            self.target(press=1.0)
            e.accept()

    def mouseMoveEvent(self, e):
        if self.isSliderDown():
            self.setValue(self._x_to_value(e.position().x()))

    def mouseReleaseEvent(self, e):
        if self.isSliderDown():
            self.setSliderDown(False)
            self.target(press=0.0)

    def mouseDoubleClickEvent(self, e):
        self.setValue(self.default)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m = 9
        y = self.height() / 2
        span = max(1, self.maximum() - self.minimum())
        x_of = lambda v: m + (v - self.minimum()) / span * (self.width() - 2 * m)
        x = x_of(self.v("pos"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(theme.BORDER))
        p.drawRoundedRect(QRectF(m, y - 2, self.width() - 2 * m, 4), 2, 2)
        x0 = x_of(self.default) if self.centered else m
        p.setBrush(qc(theme.PRIMARY) if self.isEnabled() else qc(theme.MUTED))
        p.drawRoundedRect(QRectF(min(x0, x), y - 2, abs(x - x0), 4), 2, 2)
        rad = 6.5 + 1.8 * self.v("hover") + 1.2 * self.v("press")
        if self.v("hover") > 0.01:
            halo = qc(theme.PRIMARY, 0.18 * self.v("hover"))
            p.setBrush(halo)
            p.drawEllipse(QPointF(x, y), rad + 5, rad + 5)
        p.setBrush(qc(theme.PANEL))
        p.setPen(QPen(qc(theme.PRIMARY) if self.isEnabled() else qc(theme.MUTED), 2.4))
        p.drawEllipse(QPointF(x, y), rad, rad)


# ---------------------------------------------------------------------------------------------------------------
# Aydınlık / karanlık tema anahtarı
# ---------------------------------------------------------------------------------------------------------------
class ThemeSwitch(QAbstractButton, _Anim):
    def __init__(self, dark: bool = False):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(dark)
        self._anim_init(pos=1.0 if dark else 0.0, hover=0.0)
        self.toggled.connect(lambda on: self.target(pos=1.0 if on else 0.0))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Aydınlık / karanlık tema")
        self.setFixedSize(64, 32)

    def enterEvent(self, e):
        self.target(hover=1.0)

    def leaveEvent(self, e):
        self.target(hover=0.0)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = self.v("pos")
        r = QRectF(self.rect()).adjusted(1, 2, -1, -2)
        track = mix(qc("#D6F2EE"), qc("#12395B"), t)
        p.setPen(QPen(mix(qc(theme.BORDER), qc(theme.PRIMARY), self.v("hover")), 1))
        p.setBrush(track)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        # zemindeki küçük yıldızlar karanlıkta belirir
        if t > 0.05:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc("#FFFFFF", 0.7 * t))
            for sx, sy, sr in ((14, 10, 1.2), (22, 19, 0.9), (11, 21, 0.8)):
                p.drawEllipse(QPointF(r.left() + sx, r.top() + sy - 2), sr, sr)
        d = r.height() - 6
        kx = r.left() + 3 + (r.width() - d - 6) * (t * t * (3 - 2 * t))
        knob = QRectF(kx, r.top() + 3, d, d)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(mix(qc("#FFFFFF"), qc("#E9F3F7"), t))
        p.drawEllipse(knob)
        ic = knob.adjusted(4, 4, -4, -4)
        p.save()
        p.translate(ic.center())
        p.rotate(90 * t)
        p.translate(-ic.center())
        if t < 0.5:
            draw_icon(p, "sun", ic, qc("#E9A23B", 1 - 2 * t), 1.6)
        else:
            draw_icon(p, "moon", ic, qc("#12395B", 2 * t - 1), 1.6)
        p.restore()


# ---------------------------------------------------------------------------------------------------------------
# Açılır panel (Hassasiyet)
# ---------------------------------------------------------------------------------------------------------------
class PopupPanel(QWidget):
    """Bir düğmenin altında kayarak ve belirerek açılan küçük panel. Dışarı tıklanınca kapanır."""

    closed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 18)
        self.frame = QFrame()
        self.frame.setObjectName("popup")
        sh = QGraphicsDropShadowEffect(self.frame)
        sh.setBlurRadius(28)
        sh.setOffset(0, 6)
        sh.setColor(QColor(18, 57, 91, 60))
        self.frame.setGraphicsEffect(sh)
        outer.addWidget(self.frame)
        self.body = QVBoxLayout(self.frame)
        self.body.setContentsMargins(16, 14, 16, 14)
        self.body.setSpacing(8)
        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(170)
        self._slide = QPropertyAnimation(self, b"pos", self)
        self._slide.setDuration(220)
        self._slide.setEasingCurve(QEasingCurve.Type.OutCubic)

    def show_below(self, anchor: QWidget) -> None:
        self.adjustSize()
        g = anchor.mapToGlobal(QPoint(anchor.width() // 2, anchor.height()))
        end = QPoint(g.x() - self.width() // 2, g.y() - 6)
        scr = anchor.screen().availableGeometry() if anchor.screen() else None
        if scr is not None:
            end.setX(max(scr.left() + 4, min(end.x(), scr.right() - self.width() - 4)))
        self.move(end - QPoint(0, 10))
        self.setWindowOpacity(0.0)
        self.show()
        self._fade.stop()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()
        self._slide.stop()
        self._slide.setStartValue(end - QPoint(0, 10))
        self._slide.setEndValue(end)
        self._slide.start()

    def hideEvent(self, e):
        super().hideEvent(e)
        self.closed.emit()
