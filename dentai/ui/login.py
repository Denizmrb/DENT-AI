"""Giriş ekranı: imleci izleyen diş maskotu, şifre gücüne göre parlayan diş, diş fırçalı yükleme çubuğu.

Maskot sade çizgilerle çizilir (çizgi film abartısı yok):
  * kullanıcı adı yazılırken gözleri yazılan yeri, boşta fare imlecini izler
  * şifre alanına geçince elleriyle gözlerini kapatır; şifre görünür yapılınca elleri iner ve bakar
  * şifre güçlendikçe diş temizlenir ve parlar, zayıf şifrede hafif sararır
  * her hatalı girişte biraz daha sinirlenir (kaşlar, ağız, kızarma, sarsılma, en sonda buhar)
  * başarılı girişte ışıltı ve ardından gülümseyen bir rozet
"""

from __future__ import annotations

import math

from PySide6.QtCore import QElapsedTimer, QEvent, QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton, QDialog, QHBoxLayout, QLabel, QLineEdit, QSizePolicy, QVBoxLayout, QWidget,
)

from dentai.auth import STRENGTH_LABELS, check_login, has_users, password_strength
from dentai.ui import theme
from dentai.ui.controls import _Anim, draw_icon, mix, qc
from dentai.ui.logo import app_icon, draw_logo, tooth_path

NAVY = "#12395B"


def _star(p: QPainter, c: QPointF, r: float, color: QColor) -> None:
    """Dört köşeli ışıltı yıldızı."""
    path = QPainterPath(QPointF(c.x(), c.y() - r))
    k = r * 0.22
    for dx, dy, ex, ey in ((k, -k, r, 0), (k, k, 0, r), (-k, k, -r, 0), (-k, -k, 0, -r)):
        path.quadTo(QPointF(c.x() + dx, c.y() + dy), QPointF(c.x() + ex, c.y() + ey))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawPath(path)


class ToothMascot(QWidget, _Anim):
    EYES = ((-15.0, -20.0), (15.0, -20.0))
    TYPE_HOLD = 1.6        # son tuştan sonra gözlerin yazıda kalma süresi (sn)

    def __init__(self):
        super().__init__()
        self.setFixedSize(260, 200)
        self._anim_init(lx=0.0, ly=0.0, cover=0.0, anger=0.0, clean=0.66, smile=0.15, success=0.0, badge=0.0)
        self.mode = "mouse"                  # mouse | text | cover | peek | wait
        self._look_target: QPoint | None = None
        self._typed_at = -10.0
        self._mouse_at = -10.0
        self._last_mouse = QCursor.pos()
        self._clock = QElapsedTimer()
        self._clock.start()
        self._shake_start = -10.0
        self._shake_amp = 0.0
        self._success_start = -10.0
        self._path = tooth_path()
        self._timer.start()

    # ---- durum ------------------------------------------------------------------------
    def _now(self) -> float:
        return self._clock.elapsed() / 1000.0

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.target(cover=1.0 if mode == "cover" else 0.0)

    def look_at(self, global_pt: QPoint | None) -> None:
        """Yazı yazılırken çağrılır: gözler bir süre son yazılan harfe bakar, sonra yine fareyi izler."""
        self._look_target = global_pt
        self._typed_at = self._now()

    def set_cleanliness(self, c: float) -> None:
        self.target(clean=max(0.0, min(1.0, c)))

    def set_anger(self, a: float) -> None:
        a = max(0.0, min(1.0, a))
        self.target(anger=a, smile=0.15 - 1.0 * a)

    def shake(self, amount: float = 1.0) -> None:
        self._shake_start, self._shake_amp = self._now(), amount

    def celebrate(self) -> None:
        self.set_mode("mouse")
        self._success_start = self._now()
        self.target(anger=0.0, smile=1.0, clean=1.0, success=1.0)
        QTimer.singleShot(650, lambda: self.target(badge=1.0))

    # ---- zamanlayıcı ------------------------------------------------------------------
    def _anim_extra(self) -> bool:
        now = self._now()
        mouse = QCursor.pos()
        if mouse != self._last_mouse:
            self._last_mouse, self._mouse_at = mouse, now
        # Hangisi daha yeniyse ona bak: yazı yazılıyorsa son harfe, yazı durup fare oynatılınca fareye.
        # Yazmayı bıraktıktan TYPE_HOLD saniye sonra da fareye döner.
        typing = (self.mode in ("text", "peek") and self._look_target is not None
                  and now - self._typed_at < self.TYPE_HOLD and self._typed_at >= self._mouse_at)
        pt = self._look_target if typing or (self.mode == "wait" and self._look_target is not None) else mouse
        eye = self.mapToGlobal(QPoint(self.width() // 2, int(self.height() * 0.42)))
        dx, dy = pt.x() - eye.x(), pt.y() - eye.y()
        # yazı alanı ~360 px genişliğinde: son harf soldaysa sola, sağdaysa sağa bakar
        self.target(lx=max(-1.0, min(1.0, dx / 165.0)), ly=max(-1.0, min(1.0, dy / 150.0)))
        return True                           # göz kırpma, ışıltı ve buhar için sürekli çizim

    # ---- çizim ------------------------------------------------------------------------
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        now = self._now()
        anger, clean, cover = self.v("anger"), self.v("clean"), self.v("cover")
        s = min(self.width() / 200.0, self.height() / 160.0)
        shake_t = max(0.0, 1.0 - (now - self._shake_start) / 0.55)
        sx = math.sin(now * 48) * 6.0 * self._shake_amp * shake_t ** 1.5
        bob = math.sin(now * 1.6) * 1.2 * (1 - anger)      # sakin nefes alma
        p.translate(self.width() / 2 + sx, self.height() / 2 + 10 + bob)
        p.scale(s * 0.95, s * 0.95)

        # zemin gölgesi
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(NAVY, 0.10 if not theme.DARK_MODE else 0.35))
        p.drawEllipse(QPointF(0, 72 - bob / s), 34, 5)

        # gövde rengi: sarımsı → beyaz (temizlik), sonra kırmızıya (sinir)
        base = mix(qc("#E8D8A4"), qc("#FFFFFF"), clean)
        base = mix(base, qc("#F6B9B1"), anger * 0.55)
        grad = QLinearGradient(QPointF(0, -60), QPointF(0, 64))
        grad.setColorAt(0, mix(base, qc("#FFFFFF"), 0.35))
        grad.setColorAt(1, mix(base, qc("#B9C9D3"), 0.35))
        p.setBrush(grad)
        p.setPen(QPen(qc(NAVY), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(self._path)

        # zayıf şifrede lekeler
        stain = max(0.0, 0.55 - clean) / 0.55
        if stain > 0.01:
            p.setPen(Qt.PenStyle.NoPen)
            for x, y, rx, ry in ((-27, -34, 5, 3.2), (24, -36, 4, 2.6), (-6, 18, 6, 2.4), (20, 8, 3.2, 2.2)):
                p.setBrush(qc("#B8955A", 0.32 * stain))
                p.drawEllipse(QPointF(x, y), rx, ry)

        # güçlü şifrede parlama
        shine = max(0.0, clean - 0.7) / 0.3 * (1 - anger)
        if shine > 0.01:
            arc = QPainterPath(QPointF(-31, -18))
            arc.cubicTo(-35, -38, -26, -48, -15, -44)
            p.setPen(QPen(qc("#FFFFFF", 0.9 * shine), 4.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(arc)
            for i, (x, y, r) in enumerate(((44, -52, 7), (52, -30, 4.5), (-48, -44, 4))):
                tw = 0.5 + 0.5 * math.sin(now * 3.2 + i * 2.1)
                _star(p, QPointF(x, y), r * (0.7 + 0.3 * tw), qc("#2BB8A6", shine * (0.35 + 0.65 * tw)))

        # yanaklar (sinirde kızarır)
        if anger > 0.05:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc("#E5584F", 0.38 * anger))
            p.drawEllipse(QPointF(-25, -4), 6.5, 3.6)
            p.drawEllipse(QPointF(25, -4), 6.5, 3.6)

        self._draw_face(p, now, anger, cover)
        if cover > 0.02:
            self._draw_hands(p, cover, base)
        if anger > 0.55:
            self._draw_steam(p, now, (anger - 0.55) / 0.45)
        self._draw_success(p, now)

    def _draw_face(self, p: QPainter, now: float, anger: float, cover: float) -> None:
        lx, ly = self.v("lx"), self.v("ly")
        # göz kırpma: ~3.8 sn'de bir, 0.13 sn
        phase = now % 3.8
        blink = 1.0 - (math.sin(phase / 0.13 * math.pi) if phase < 0.13 else 0.0)
        squint = 1.0 - 0.38 * anger
        for i, (ex, ey) in enumerate(self.EYES):
            ry = 7.4 * blink * squint
            if ry < 1.0:
                p.setPen(QPen(qc(NAVY), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(ex - 6.2, ey), QPointF(ex + 6.2, ey))
            else:
                # göz akı + içinde hareket eden göz bebeği (yazılan son harfe / imlece bakar)
                white = QPainterPath()
                white.addEllipse(QPointF(ex, ey), 6.4, ry)
                p.setPen(QPen(qc(NAVY), 1.6))
                p.setBrush(qc("#FFFFFF"))
                p.drawPath(white)
                p.save()
                p.setClipPath(white)
                px, py = ex + lx * 3.1, ey + ly * min(3.2, ry * 0.45)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(qc(NAVY))
                p.drawEllipse(QPointF(px, py), 3.5, 3.5)
                p.setBrush(qc("#FFFFFF", 0.95))
                p.drawEllipse(QPointF(px + 1.2, py - 1.3), 1.1, 1.1)
                p.restore()
            # kaşlar: sinir arttıkça içe doğru eğilir ve alçalır
            side = -1 if i == 0 else 1
            inner = QPointF(ex - side * 2.5, ey - 11.5 + 4.0 * anger)
            outer = QPointF(ex + side * 7.5, ey - 13.0 - 1.0 * anger)
            p.setPen(QPen(qc(NAVY, 0.55 + 0.45 * min(1, anger * 2)), 1.8 + 1.4 * anger, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap))
            p.drawLine(inner, outer)
        # ağız
        smile = self.v("smile")
        w = 8.0 + 3.0 * self.v("success")
        y0 = -2.0
        mouth = QPainterPath(QPointF(-w, y0))
        mouth.quadTo(QPointF(0, y0 + smile * 9.0), QPointF(w, y0))
        if self.v("success") > 0.3:
            mouth.quadTo(QPointF(0, y0 + smile * 3.0), QPointF(-w, y0))
            p.setBrush(qc(NAVY))
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(qc(NAVY), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(mouth)

    def _draw_hands(self, p: QPainter, cover: float, fill: QColor) -> None:
        t = cover * cover * (3 - 2 * cover)
        alpha = min(1.0, cover * 2.5)
        for side, (ex, ey) in zip((-1, 1), self.EYES):
            start = QPointF(side * 36, 8)
            rest = QPointF(side * 46, 30)
            hand = QPointF(rest.x() + (ex + side * 1.5 - rest.x()) * t, rest.y() + (ey - rest.y()) * t)
            arm = QPainterPath(start)
            ctrl = QPointF(side * (46 - 8 * t), 6 - 14 * t)
            arm.quadTo(ctrl, hand)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(qc(NAVY, alpha), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPath(arm)
            p.save()
            p.translate(hand)
            p.rotate(side * (30 - 22 * t))
            c = mix(QColor(fill), qc("#DCE6EC"), 0.25)
            c.setAlphaF(alpha)
            # avuç + dört parmak (parmak uçları yukarıda, gözün üstünü örter)
            palm = QPainterPath()
            palm.addRoundedRect(QRectF(-10, -4, 20, 15), 7, 7)
            for fx, fh in ((-7.2, 9.5), (-2.4, 11.5), (2.4, 11.0), (7.2, 8.5)):
                finger = QPainterPath()
                finger.addRoundedRect(QRectF(fx - 2.5, -4 - fh, 5, fh + 6), 2.5, 2.5)
                palm = palm.united(finger)
            p.setBrush(c)
            p.setPen(QPen(qc(NAVY, alpha), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.drawPath(palm)
            p.restore()

    def _draw_steam(self, p: QPainter, now: float, k: float) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        for side in (-1, 1):
            for j in range(3):
                ph = (now * 0.9 + j / 3 + (0.5 if side > 0 else 0)) % 1.0
                x = side * (30 + 6 * ph) + math.sin(now * 4 + j) * 2
                y = -58 - 26 * ph
                r = 3.5 + 5 * ph
                p.setBrush(qc("#9FB2C2", k * 0.55 * (1 - ph)))
                p.drawEllipse(QPointF(x, y), r, r)

    def _draw_success(self, p: QPainter, now: float) -> None:
        dt = now - self._success_start
        if 0 <= dt < 1.6:
            for i in range(7):
                a = i / 7 * 2 * math.pi + 0.4
                k = min(1.0, dt / 0.6)
                rad = 58 + 22 * k
                fade = 1.0 - max(0.0, (dt - 0.7) / 0.9)
                size = (5 + 4 * math.sin(dt * 9 + i)) * (0.4 + 0.6 * k)
                col = qc("#F5C34B" if i % 2 else "#2BB8A6", max(0.0, fade))
                _star(p, QPointF(math.cos(a) * rad, math.sin(a) * rad * 0.8 - 6), size, col)
        b = self.v("badge")
        if b > 0.01:
            c = QPointF(42, -50)
            r = 13 * (b * (1.25 - 0.25 * b))       # hafif büyüyüp yerine oturur
            p.setPen(QPen(qc("#FFFFFF"), 2.2))
            p.setBrush(qc(theme.SUCCESS))
            p.drawEllipse(c, r, r)
            p.setPen(QPen(qc("#FFFFFF"), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.setBrush(Qt.BrushStyle.NoBrush)
            sm = QPainterPath(QPointF(c.x() - r * 0.45, c.y() + r * 0.12))
            sm.quadTo(QPointF(c.x(), c.y() + r * 0.62), QPointF(c.x() + r * 0.45, c.y() + r * 0.12))
            p.drawPath(sm)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc("#FFFFFF"))
            p.drawEllipse(QPointF(c.x() - r * 0.3, c.y() - r * 0.22), r * 0.1, r * 0.13)
            p.drawEllipse(QPointF(c.x() + r * 0.3, c.y() - r * 0.22), r * 0.1, r * 0.13)


class StrengthBar(QWidget, _Anim):
    COLORS = ["#E5584F", "#F0A044", "#2BB8A6", "#3DBE7A"]

    def __init__(self):
        super().__init__()
        self.setFixedHeight(6)
        self._anim_init(level=0.0)

    def set_level(self, n: int) -> None:
        self.target(level=float(n))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        lvl = self.v("level")
        gap = 5
        w = (self.width() - 3 * gap) / 4
        col = qc(self.COLORS[max(0, min(3, round(lvl) - 1))]) if lvl > 0.05 else qc(theme.BORDER)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(4):
            r = QRectF(i * (w + gap), 0, w, self.height())
            p.setBrush(qc(theme.BORDER))
            p.drawRoundedRect(r, 3, 3)
            fillk = max(0.0, min(1.0, lvl - i))
            if fillk > 0:
                p.setBrush(col)
                p.drawRoundedRect(QRectF(r.x(), 0, r.width() * fillk, r.height()), 3, 3)


class LoginButton(QAbstractButton, _Anim):
    """Basılınca diş fırçasının ilerlediği bir yükleme çubuğuna dönüşür; sonunda finished sinyali."""

    finished = Signal()
    DURATION = 1.5

    def __init__(self, text: str = "Giriş Yap"):
        super().__init__()
        self.setText(text)
        self.setFixedHeight(52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.state = "idle"                  # idle | loading | success
        self._anim_init(hover=0.0, morph=0.0, ok=0.0)
        self._clock = QElapsedTimer()
        self._progress = 0.0

    def sizeHint(self) -> QSize:
        return QSize(320, 52)

    def enterEvent(self, e):
        self.target(hover=1.0)

    def leaveEvent(self, e):
        self.target(hover=0.0)

    def start_loading(self) -> None:
        self.state = "loading"
        self._progress = 0.0
        self._clock.start()
        self.setCursor(Qt.CursorShape.BusyCursor)
        self.target(morph=1.0)

    def show_success(self) -> None:
        self.state = "success"
        self.target(morph=0.0, ok=1.0)

    def reset(self) -> None:
        self.state = "idle"
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.target(morph=0.0, ok=0.0)

    def _anim_extra(self) -> bool:
        if self.state == "loading":
            t = min(1.0, self._clock.elapsed() / 1000.0 / self.DURATION)
            self._progress = 1 - (1 - t) ** 2.2 if t < 1 else 1.0
            if t >= 1.0:
                self.state = "checking"
                QTimer.singleShot(0, self.finished.emit)
            return True
        return self.state == "checking"

    def mousePressEvent(self, e):
        if self.state == "idle":
            super().mousePressEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m, hv, ok = self.v("morph"), self.v("hover"), self.v("ok")
        full = QRectF(self.rect()).adjusted(1, 3, -1, -3)
        bh = full.height() + (14 - full.height()) * m
        cy = full.center().y() + 9 * m
        bar = QRectF(full.left(), cy - bh / 2, full.width(), bh)
        rad = 12 + (bh / 2 - 12) * m
        primary = mix(qc(theme.PRIMARY), qc(theme.PRIMARY_HOVER), hv * (1 - m))
        primary = mix(primary, qc(theme.SUCCESS), ok)
        track = mix(primary, qc(theme.PRIMARY_SOFT), m)
        if hv > 0.01 and m < 0.5:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc(theme.PRIMARY, 0.18 * hv * (1 - 2 * m)))
            p.drawRoundedRect(bar.translated(0, 3), rad, rad)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(bar, rad, rad)

        if self.state in ("loading", "checking") or m > 0.02:
            prog = self._progress if self.state in ("loading", "checking") else 1.0
            fill = QRectF(bar.left(), bar.top(), max(bh, bar.width() * prog), bh)
            p.setBrush(qc(theme.PRIMARY, m))
            p.drawRoundedRect(fill, bh / 2, bh / 2)
            # köpük
            now = self._clock.elapsed() / 1000.0 if self._clock.isValid() else 0.0
            for i in range(7):
                bx = fill.right() - 10 - i * 11 - (now * 30 + i * 7) % 8
                if bx < bar.left() + 6:
                    continue
                by = bar.top() - 2 - (i % 3) * 3.2
                p.setBrush(qc("#FFFFFF", 0.9 * m * (1 - i / 7)))
                p.setPen(QPen(qc(theme.PRIMARY, 0.55 * m), 0.8))
                p.drawEllipse(QPointF(bx, by), 2.6 - i * 0.22, 2.6 - i * 0.22)
            self._draw_brush(p, fill.right() + math.sin(now * 26) * 3.0, bar.top(), m)

        text_alpha = max(1 - m * 2.5, ok)
        if text_alpha > 0.01:
            f = QFont("Segoe UI", 11)
            f.setWeight(QFont.Weight.Bold)
            p.setFont(f)
            p.setPen(qc("#FFFFFF", text_alpha))
            label = "Hoş geldiniz" if self.state == "success" else self.text()
            if self.state == "success":
                tw = p.fontMetrics().horizontalAdvance(label)
                ic = QRectF(full.center().x() - tw / 2 - 26, full.center().y() - 9, 18, 18)
                draw_icon(p, "check", ic, qc("#FFFFFF", text_alpha), 2.4)
                p.drawText(full.translated(12, 0), Qt.AlignmentFlag.AlignCenter, label)
            else:
                p.drawText(full, Qt.AlignmentFlag.AlignCenter, label)
        if m > 0.5 and self.state in ("loading", "checking"):
            f = QFont("Segoe UI", 8)
            p.setFont(f)
            p.setPen(qc(theme.MUTED, (m - 0.5) * 2))
            p.drawText(QRectF(full.left(), full.top() - 2, full.width(), 16), Qt.AlignmentFlag.AlignCenter,
                       "Kimlik doğrulanıyor…")

    def _draw_brush(self, p: QPainter, x: float, bar_top: float, m: float) -> None:
        if m < 0.05:
            return
        p.save()
        p.translate(x, bar_top - 1)
        p.scale(m, m)
        p.setPen(Qt.PenStyle.NoPen)
        # kıllar
        p.setBrush(qc("#FFFFFF"))
        p.setPen(QPen(qc(theme.PRIMARY), 1.0))
        p.drawRoundedRect(QRectF(-21, -6, 20, 6), 2, 2)
        p.setPen(QPen(qc(theme.PRIMARY_HOVER), 1.0))
        for i in range(6):
            bx = -19 + i * 3.4
            p.drawLine(QPointF(bx, -5), QPointF(bx, -1))
        # baş ve sap
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(NAVY))
        p.drawRoundedRect(QRectF(-23, -10, 24, 4.5), 2.2, 2.2)
        path = QPainterPath(QPointF(-23, -9))
        path.cubicTo(QPointF(-30, -10), QPointF(-38, -14), QPointF(-56, -15))
        path.lineTo(QPointF(-56, -10.5))
        path.cubicTo(QPointF(-40, -9.5), QPointF(-31, -6), QPointF(-23, -5.5))
        path.closeSubpath()
        p.drawPath(path)
        p.setBrush(qc(theme.PRIMARY))
        p.drawRoundedRect(QRectF(-50, -14.6, 9, 3.4), 1.6, 1.6)
        p.restore()


class BrandPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(400)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g = QLinearGradient(QPointF(0, 0), QPointF(self.width(), self.height()))
        g.setColorAt(0, qc("#12395B"))
        g.setColorAt(1, qc("#0B2740" if not theme.DARK_MODE else "#081B2C"))
        p.fillRect(self.rect(), g)
        # arka planda soluk diş desenleri
        for i, (x, y, sz, a) in enumerate(((300, 90, 120, 0.06), (60, 470, 150, 0.05), (330, 520, 80, 0.07),
                                            (210, 300, 60, 0.05))):
            draw_icon(p, "tooth", QRectF(x - sz / 2, y - sz / 2, sz, sz), qc("#FFFFFF", a), 1.2 * sz / 40)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc("#2BB8A6", 0.10))
        p.drawEllipse(QPointF(self.width() + 40, -40), 190, 190)

        x = 48
        y0 = self.height() / 2 - 130                  # logo + başlık dikeyde ortalanır
        draw_logo(p, QRectF(x, y0, 92, 92))
        f = QFont("Segoe UI", 30)
        f.setWeight(QFont.Weight.ExtraBold)
        p.setFont(f)
        p.setPen(qc("#FFFFFF"))
        p.drawText(QPointF(x, y0 + 160), "DENT-")
        dash_w = p.fontMetrics().horizontalAdvance("DENT-")
        p.setPen(qc("#2BB8A6"))
        p.drawText(QPointF(x + dash_w, y0 + 160), "AI")
        f2 = QFont("Segoe UI", 11)
        p.setFont(f2)
        p.setPen(qc("#FFFFFF", 0.72))
        p.drawText(QPointF(x, y0 + 192), "Panoramik röntgen için ikinci göz")

        p.setPen(QPen(qc("#2BB8A6"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(x, y0 + 218), QPointF(x + 44, y0 + 218))

        f4 = QFont("Segoe UI", 8)
        p.setFont(f4)
        p.setPen(qc("#FFFFFF", 0.45))
        p.drawText(QRectF(x, self.height() - 70, self.width() - 2 * x, 44), Qt.TextFlag.TextWordWrap,
                   "Karar destek aracıdır; tanı yerine geçmez. Nihai karar hekime aittir.")


def _eye_icon(off: bool) -> QIcon:
    pm = QPixmap(40, 40)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    draw_icon(p, "eye_off" if off else "eye", QRectF(6, 6, 28, 28), qc(theme.MUTED), 2.6)
    p.end()
    return QIcon(pm)


class LoginDialog(QDialog):
    MESSAGES = [
        "Kullanıcı adı veya şifre hatalı.",
        "Yine olmadı. Bilgileri kontrol edin.",
        "3. hatalı deneme — büyük/küçük harfe dikkat edin.",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DENT-AI — Giriş")
        self.setWindowIcon(app_icon())
        self.setFixedSize(960, 640)
        self.username = ""
        self.fails = 0

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(BrandPanel())

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 24, 0, 24)
        rl.addStretch(1)
        form = QWidget()
        form.setFixedWidth(360)
        fl = QVBoxLayout(form)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(6)

        self.mascot = ToothMascot()
        fl.addWidget(self.mascot, 0, Qt.AlignmentFlag.AlignHCenter)
        t = QLabel("Hoş geldiniz")
        t.setObjectName("formTitle")
        t.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        sub = QLabel("Devam etmek için hesabınızla giriş yapın")
        sub.setObjectName("muted")
        sub.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        fl.addWidget(t)
        fl.addWidget(sub)
        fl.addSpacing(14)

        fl.addWidget(self._caption("Kullanıcı adı"))
        self.user = QLineEdit()
        self.user.setObjectName("login")
        self.user.setPlaceholderText("kullanıcı adınız")
        fl.addWidget(self.user)
        fl.addSpacing(6)
        fl.addWidget(self._caption("Şifre"))
        self.pw = QLineEdit()
        self.pw.setObjectName("login")
        self.pw.setPlaceholderText("••••••••")
        self.pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.eye = QAction(_eye_icon(False), "Şifreyi göster", self.pw)
        self.eye.triggered.connect(self._toggle_pw)
        self.pw.addAction(self.eye, QLineEdit.ActionPosition.TrailingPosition)
        fl.addWidget(self.pw)

        srow = QHBoxLayout()
        srow.setSpacing(10)
        self.strength = StrengthBar()
        self.strength_lbl = QLabel("")
        self.strength_lbl.setObjectName("muted")
        self.strength_lbl.setFixedWidth(84)
        srow.addWidget(self.strength, 1)
        srow.addWidget(self.strength_lbl)
        fl.addSpacing(2)
        fl.addLayout(srow)

        self.error = QLabel("")
        self.error.setObjectName("error")
        self.error.setFixedHeight(22)
        self.error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fl.addWidget(self.error)
        self.btn = LoginButton("Giriş Yap")
        self.btn.clicked.connect(self.submit)
        self.btn.finished.connect(self._verify)
        fl.addWidget(self.btn)

        rl.addWidget(form, 0, Qt.AlignmentFlag.AlignHCenter)
        rl.addStretch(1)
        lay.addWidget(right, 1)

        self.user.installEventFilter(self)
        self.pw.installEventFilter(self)
        self.user.textEdited.connect(self._follow_user_caret)
        self.user.cursorPositionChanged.connect(lambda *_: self._follow_user_caret())
        self.pw.textChanged.connect(self._pw_changed)
        self.pw.cursorPositionChanged.connect(lambda *_: self._follow_pw_caret())
        self.user.returnPressed.connect(self.pw.setFocus)
        self.pw.returnPressed.connect(self.submit)
        self._pw_changed("")
        QTimer.singleShot(0, self.user.setFocus)
        if not has_users():
            self._no_users_hint()

    def _no_users_hint(self):
        self.error.setText("Kullanıcı yok: python -m dentai.auth ekle <ad>")
        self.error.setToolTip("Kullanıcılar kayitlar/kullanicilar.json dosyasında tutulur.")

    @staticmethod
    def _caption(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("h")
        return lbl

    # ---- maskot bağlantıları -----------------------------------------------------------
    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.FocusIn:
            if obj is self.user:
                self.mascot.set_mode("text")
                self._follow_user_caret()
            elif obj is self.pw:
                self._pw_mode()
        elif ev.type() == QEvent.Type.FocusOut and obj in (self.user, self.pw):
            QTimer.singleShot(0, self._focus_check)
        return super().eventFilter(obj, ev)

    def _focus_check(self):
        if not (self.user.hasFocus() or self.pw.hasFocus()) and self.btn.state == "idle":
            self.mascot.set_mode("mouse")

    def _caret_global(self, le: QLineEdit) -> QPoint:
        return le.mapToGlobal(le.cursorRect().center())

    def _follow_user_caret(self, *_):
        if self.user.hasFocus():
            self.mascot.look_at(self._caret_global(self.user))

    def _follow_pw_caret(self):
        if self.pw.hasFocus() and self.pw.echoMode() == QLineEdit.EchoMode.Normal:
            self.mascot.look_at(self._caret_global(self.pw))

    def _pw_mode(self):
        if self.pw.echoMode() == QLineEdit.EchoMode.Normal:
            self.mascot.set_mode("peek")
            self._follow_pw_caret()
        else:
            self.mascot.set_mode("cover")

    def _toggle_pw(self):
        show = self.pw.echoMode() == QLineEdit.EchoMode.Password
        self.pw.setEchoMode(QLineEdit.EchoMode.Normal if show else QLineEdit.EchoMode.Password)
        self.eye.setIcon(_eye_icon(show))
        self.eye.setText("Şifreyi gizle" if show else "Şifreyi göster")
        self.pw.setFocus()
        self._pw_mode()

    def _pw_changed(self, text: str):
        n = password_strength(text)
        self.strength.set_level(n)
        self.strength_lbl.setText(f"{STRENGTH_LABELS[n]}" if text else "Şifre gücü")
        # boş: nötr; zayıf: sararmış; güçlü: parlak beyaz
        self.mascot.set_cleanliness(0.66 if not text else [0.0, 0.08, 0.45, 0.82, 1.0][n])
        if self.error.text() and text:
            self.error.setText("")
        self._follow_pw_caret()

    # ---- giriş ------------------------------------------------------------------------
    def submit(self):
        if self.btn.state != "idle":
            return
        if not has_users():
            self._no_users_hint()
            return
        if not self.user.text().strip() or not self.pw.text():
            self.error.setText("Kullanıcı adı ve şifre gerekli.")
            self.mascot.shake(0.4)
            (self.user if not self.user.text().strip() else self.pw).setFocus()
            return
        self.error.setText("")
        for w in (self.user, self.pw):
            w.setEnabled(False)
        self.mascot.set_mode("wait")
        self.mascot.look_at(self.btn.mapToGlobal(QPoint(self.btn.width() // 2, self.btn.height() // 2)))
        self.btn.start_loading()

    def _verify(self):
        if check_login(self.user.text(), self.pw.text()):
            self.username = self.user.text().strip().lower()
            self.btn.show_success()
            self.mascot.celebrate()
            QTimer.singleShot(1700, self.accept)
            return
        self.fails += 1
        self.mascot.set_anger(min(1.0, self.fails / 4))
        self.mascot.shake(0.6 + 0.15 * min(self.fails, 4))
        self.error.setText(self.MESSAGES[self.fails - 1] if self.fails <= len(self.MESSAGES)
                           else f"{self.fails}. hatalı deneme. Caps Lock açık olabilir mi?")
        self.btn.reset()
        for w in (self.user, self.pw):
            w.setEnabled(True)
        self.pw.clear()
        self.pw.setFocus()
        self._pw_mode()
