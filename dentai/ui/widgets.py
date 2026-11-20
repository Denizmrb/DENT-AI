from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QSize, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QLabel, QSizePolicy, QToolTip, QVBoxLayout, QWidget,
)

from dentai.config import load_classes
from dentai.fdi import LOWER_ROW, UPPER_ROW
from dentai.imaging import IMAGE_EXTS
from dentai.inference.results import AnalysisResult
from dentai.ui import theme
from dentai.ui.controls import _Anim, mix, qc
from dentai.ui.logo import draw_logo


def bgr_to_pixmap(img: np.ndarray) -> QPixmap:
    h, w = img.shape[:2]
    img = np.ascontiguousarray(img)
    qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


def dropped_image_path(mime) -> Path | None:
    """Sürüklenen veride ilk yerel görüntü dosyası (yoksa None)."""
    if not mime.hasUrls():
        return None
    for url in mime.urls():
        p = Path(url.toLocalFile())
        if url.isLocalFile() and p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


# Bir dişte birden çok bulgu varsa haritada hangisinin rengi görünsün (önemliden önemsize)
FINDING_PRIORITY = ["caries", "impacted", "root_canal_treatment", "crown", "implant", "filling"]


def _lerp_rect(a: QRectF, b: QRectF, t: float) -> QRectF:
    return QRectF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t,
                  a.width() + (b.width() - a.width()) * t, a.height() + (b.height() - a.height()) * t)


class ImageViewer(QGraphicsView):
    """Tekerlek ile imleç merkezli zoom, sürükleyerek kaydırma, çift tık = sığdır (oran korunur).
    zoom_by / zoom_to_box / reset_view yumuşak animasyonla çalışır.
    Görüntü dosyası üzerine sürüklenip bırakılabilir (fileDropped sinyali)."""

    fileDropped = Signal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._drag_ok: bool | None = None   # None: sürükleme yok, True/False: geçerli/geçersiz dosya
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._fit = True
        self._focus: QRectF | None = None      # zoom yapılan dişin kutusu (vurgulanır)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(450)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.valueChanged.connect(self._anim_step)
        self._anim.finished.connect(self._anim_done)
        self._from = self._to = QRectF()
        self._fit_after = False
        self.apply_theme()

    def apply_theme(self) -> None:
        self.setBackgroundBrush(QColor(theme.STAGE))
        self.viewport().update()

    def has_image(self) -> bool:
        return not self._item.pixmap().isNull()

    def image_rect(self) -> QRectF:
        return self._item.boundingRect()

    def set_image(self, bgr: np.ndarray, reset_view: bool = False) -> None:
        self._item.setPixmap(bgr_to_pixmap(bgr))
        self._scene.setSceneRect(self._item.boundingRect())
        if reset_view:
            self._focus = None
        if reset_view or self._fit:
            self.fit()

    def fit(self) -> None:
        if self.has_image():
            self._anim.stop()
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
            self._fit = True

    def visible_rect(self) -> QRectF:
        return self.mapToScene(self.viewport().rect()).boundingRect()

    # ---- animasyonlu zoom -------------------------------------------------------------
    def animate_to(self, target: QRectF, fit_after: bool = False) -> None:
        if not self.has_image():
            return
        self._anim.stop()
        self._from, self._to, self._fit_after = self.visible_rect(), QRectF(target), fit_after
        self._fit = False
        self._anim.start()

    def _anim_step(self, t) -> None:
        self.fitInView(_lerp_rect(self._from, self._to, float(t)), Qt.AspectRatioMode.KeepAspectRatio)

    def _anim_done(self) -> None:
        if self._fit_after:
            self.fit()

    def zoom_by(self, factor: float) -> None:
        if not self.has_image():
            return
        r = self.visible_rect() if self._anim.state() != QVariantAnimation.State.Running else self._to
        img = self.image_rect()
        w, h = r.width() / factor, r.height() / factor
        if factor < 1 and w >= img.width() and h >= img.height():
            self.reset_view()
            return
        if w < 40 or h < 40:
            return
        c = r.center()
        self.animate_to(QRectF(c.x() - w / 2, c.y() - h / 2, w, h))

    def zoom_to_box(self, xyxy, context: float = 0.45) -> None:
        """Kutuya yakınlaşır; context: kutunun her yanında bırakılan pay (kutu boyunun oranı)."""
        x1, y1, x2, y2 = xyxy
        self._focus = QRectF(x1, y1, x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w = max((x2 - x1) * (1 + 2 * context), 140)
        h = max((y2 - y1) * (1 + context * 0.5), 140)
        self.animate_to(QRectF(cx - w / 2, cy - h / 2, w, h))
        self.viewport().update()

    def reset_view(self) -> None:
        self._focus = None
        if self.has_image():
            r = self.image_rect()
            # sığdırma oranında, görüntü ortalanmış hedef dikdörtgen
            vr = self.viewport().rect()
            s = min(vr.width() / max(1, r.width()), vr.height() / max(1, r.height()))
            w, h = vr.width() / s, vr.height() / s
            self.animate_to(QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h), fit_after=True)
        self.viewport().update()

    def wheelEvent(self, e):
        if not self.has_image():
            return
        self._anim.stop()
        factor = 1.18 if e.angleDelta().y() > 0 else 1 / 1.18
        img = self.image_rect()
        vr = self.visible_rect()
        if factor < 1 and vr.width() / factor > img.width() * 1.6 and vr.height() / factor > img.height() * 1.6:
            return
        if factor > 1 and vr.width() < 50:
            return
        self.scale(factor, factor)
        self._fit = False

    def mouseDoubleClickEvent(self, e):
        self.reset_view()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self.fit()

    # ---- sürükle-bırak ----------------------------------------------------------------
    # QGraphicsView sürükleme olaylarını sahneye iletir ve reddeder; burada doğrudan ele alınır.
    def _set_drag(self, state: bool | None) -> None:
        self._drag_ok = state
        self.viewport().update()

    def dragEnterEvent(self, e):
        ok = dropped_image_path(e.mimeData()) is not None
        self._set_drag(ok)
        if ok:
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if self._drag_ok:
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self._set_drag(None)

    def dropEvent(self, e):
        p = dropped_image_path(e.mimeData())
        self._set_drag(None)
        if p is not None:
            e.acceptProposedAction()
            self.fileDropped.emit(p)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        if self._focus is not None and self.has_image() and self._drag_ok is None:
            painter.save()
            pen = QPen(QColor(theme.PRIMARY), 0)     # kozmetik kalem: zoomdan bağımsız 2 px
            pen.setWidthF(2.5)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self._focus.adjusted(-6, -6, 6, 6), 8, 8)
            painter.restore()
        if self._drag_ok is None and self.has_image():
            return
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = QRectF(self.viewport().rect()).adjusted(14, 14, -14, -14)
        f = QFont("Segoe UI", 12)
        if self._drag_ok is not None:
            accent = QColor(theme.PRIMARY if self._drag_ok else theme.DANGER)
            fill = QColor(accent)
            fill.setAlpha(45)
            painter.setBrush(fill)
            painter.setPen(QPen(accent, 2, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(area, 14, 14)
            f.setPointSize(15)
            f.setBold(True)
            painter.setFont(f)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter,
                             "Röntgeni bırakın" if self._drag_ok else "Desteklenmeyen dosya türü")
        else:
            painter.setPen(QPen(QColor(theme.STAGE_LINE), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(area, 14, 14)
            c = area.center()
            draw_logo(painter, QRectF(c.x() - 48, c.y() - 128, 96, 96), tile=False,
                      stroke=theme.STAGE_TEXT, alpha=0.55)
            painter.setFont(f)
            painter.setPen(QColor(theme.STAGE_TEXT))
            painter.drawText(area.adjusted(0, 60, 0, 0), Qt.AlignmentFlag.AlignCenter,
                             "Panoramik röntgeni buraya sürükleyip bırakın\n"
                             "veya üstteki  “Röntgen Yükle”  düğmesine basın")
        painter.restore()


class Card(QFrame):
    def __init__(self, title: str | None = None):
        super().__init__()
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(16, 14, 16, 14)
        self.body.setSpacing(8)
        self.title_label = None
        if title:
            self.title_label = QLabel(title.upper())
            self.title_label.setObjectName("h")
            self.body.addWidget(self.title_label)


class StatCard(QFrame):
    def __init__(self, caption: str, color: str):
        super().__init__()
        self.setObjectName("card")
        self._color = color
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 10, 10, 10)
        lay.setSpacing(0)
        self.value = QLabel("–")
        self.value.setObjectName("big")
        cap = QLabel(caption)
        cap.setObjectName("muted")
        lay.addWidget(self.value)
        lay.addWidget(cap)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, text: str) -> None:
        self.value.setText(text)

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(self._color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(0, 12, 4, self.height() - 24), 2, 2)


class FdiBar(QWidget, _Anim):
    """Röntgenin altındaki FDI diş çubuğu. Bulgulu dişler bulgu renginde çerçevelenir,
    algılanmayan dişler soluk görünür, not yazılmış dişte küçük nokta olur. Tıklanan diş: toothClicked."""

    toothClicked = Signal(str)

    N, GAP, MID_GAP, LABEL_W = 16, 5, 16, 46

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFixedHeight(112)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._result: AnalysisResult | None = None
        self._notes: set[str] = set()
        self.selected: str | None = None
        self._hover: str | None = None
        self._anim_init(**{f: 0.0 for f in UPPER_ROW + LOWER_ROW})

    def sizeHint(self) -> QSize:
        return QSize(760, 112)

    def set_result(self, r: AnalysisResult | None) -> None:
        self._result = r
        self.update()

    def set_noted(self, teeth: set[str]) -> None:
        self._notes = set(teeth)
        self.update()

    def set_selected(self, fdi: str | None) -> None:
        self.selected = fdi
        self.update()

    # ---- geometri ---------------------------------------------------------------------
    def _geom(self):
        avail = self.width() - self.LABEL_W - 24
        cw = min(42.0, (avail - (self.N - 2) * self.GAP - self.MID_GAP) / self.N)
        total = self.N * cw + (self.N - 2) * self.GAP + self.MID_GAP
        x0 = self.LABEL_W + 12 + max(0.0, (avail - total) / 2)
        return x0, cw, 30.0

    def _cells(self):
        x0, cw, ch = self._geom()
        for row, names in enumerate((UPPER_ROW, LOWER_ROW)):
            y = 36 + row * (ch + 8)
            for i, fdi in enumerate(names):
                x = x0 + i * (cw + self.GAP) + (self.MID_GAP - self.GAP if i >= 8 else 0)
                yield fdi, QRectF(x, y, cw, ch)

    def tooth_at(self, pos: QPointF) -> str | None:
        for fdi, r in self._cells():
            if r.adjusted(-2, -3, 2, 3).contains(pos):
                return fdi
        return None

    def _finding_info(self, fdi: str):
        r = self._result
        if r is None:
            return None, False, []
        present = any(t.name == fdi for t in r.teeth)
        dets = r.by_tooth().get(fdi, [])
        names = {d.name for d in dets}
        top = next((n for n in FINDING_PRIORITY if n in names), None)
        if top is None and dets:
            top = dets[0].name
        return top, present or bool(dets), dets

    # ---- olaylar ----------------------------------------------------------------------
    def mouseMoveEvent(self, e):
        fdi = self.tooth_at(e.position())
        if fdi != self._hover:
            if self._hover:
                self.target(**{self._hover: 0.0})
            if fdi:
                self.target(**{fdi: 1.0})
            self._hover = fdi
            QToolTip.showText(e.globalPosition().toPoint(), self._tip(fdi) if fdi else "", self)

    def leaveEvent(self, e):
        if self._hover:
            self.target(**{self._hover: 0.0})
        self._hover = None

    def mousePressEvent(self, e):
        # Seçimi pencere yönetir: aynı dişe ikinci tıklama seçimi kaldırır (MainWindow.toggle_tooth).
        fdi = self.tooth_at(e.position())
        if fdi and e.button() == Qt.MouseButton.LeftButton:
            self.toothClicked.emit(fdi)

    def _tip(self, fdi: str) -> str:
        classes = load_classes()
        top, present, dets = self._finding_info(fdi)
        lines = [f"Diş {fdi}"]
        if self._result is not None:
            if dets:
                lines += [f"• {(classes.by_name(d.name).label if classes.by_name(d.name) else d.name)} ({d.conf:.0%})"
                          for d in dets]
            elif present:
                lines.append("Bulgu yok")
            else:
                lines.append("Bu diş röntgende algılanmadı")
        if fdi in self._notes:
            lines.append("✎ Hekim notu var")
        lines.append("Tekrar tıklayın: seçimi kaldır" if fdi == self.selected else "Tıklayın: dişe yakınlaştır")
        return "\n".join(lines)

    # ---- çizim ------------------------------------------------------------------------
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(theme.BAR_BG))
        p.drawRoundedRect(bg, 12, 12)

        x0, cw, ch = self._geom()
        mid = x0 + 8 * (cw + self.GAP) - self.GAP + (self.MID_GAP - self.GAP) / 2 + self.GAP / 2
        left_c = (x0 + mid) / 2
        right_c = (mid + x0 + self.N * cw + (self.N - 2) * self.GAP + self.MID_GAP) / 2
        f = QFont("Segoe UI", 9)
        f.setBold(True)
        p.setFont(f)
        p.setPen(qc("#FFFFFF", 0.85))
        p.drawText(QRectF(left_c - 120, 6, 240, 22), Qt.AlignmentFlag.AlignCenter, "R (Hasta Sağı)")
        p.drawText(QRectF(right_c - 120, 6, 240, 22), Qt.AlignmentFlag.AlignCenter, "L (Hasta Solu)")
        p.setPen(QPen(qc("#FFFFFF", 0.35), 1.2))
        p.drawLine(QPointF(mid, 9), QPointF(mid, 25))
        p.setPen(QPen(qc("#FFFFFF", 0.14), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(mid, 32), QPointF(mid, self.height() - 6))
        f.setPointSize(9)
        p.setFont(f)
        p.setPen(qc("#FFFFFF", 0.75))
        p.drawText(QRectF(8, 36, self.LABEL_W, ch), Qt.AlignmentFlag.AlignCenter, "ÜST")
        p.drawText(QRectF(8, 36 + ch + 8, self.LABEL_W, ch), Qt.AlignmentFlag.AlignCenter, "ALT")

        classes = load_classes()
        fnum = QFont("Segoe UI", 9)
        fnum.setBold(True)
        p.setFont(fnum)
        for fdi, r in self._cells():
            hv = self.v(fdi)
            top, present, _ = self._finding_info(fdi)
            sel = fdi == self.selected
            grow = 1.5 * hv
            rr = r.adjusted(-grow, -grow, grow, grow).translated(0, -1.5 * hv)
            cell = qc(theme.CELL)
            txt = qc(theme.CELL_TEXT)
            border = QPen(Qt.PenStyle.NoPen)
            if self._result is not None and not present:
                cell.setAlphaF(0.28)
                txt = qc("#FFFFFF", 0.55)
            if top:
                spec = classes.by_name(top)
                col = QColor(*spec.color_rgb) if spec else qc(theme.DANGER)
                border = QPen(col, 2.6)
                if top == "caries":
                    txt = col
            if sel:
                cell = qc(theme.PRIMARY)
                txt = qc("#FFFFFF")
            if hv > 0 and not sel:
                cell = mix(cell, qc("#FFFFFF"), 0.35 * hv)
            p.setBrush(cell)
            p.setPen(border)
            p.drawRoundedRect(rr, 6, 6)
            p.setPen(txt)
            p.drawText(rr, Qt.AlignmentFlag.AlignCenter, fdi)
            if fdi in self._notes:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(qc("#FFFFFF") if sel else qc(theme.PRIMARY))
                p.drawEllipse(QPointF(rr.right() - 5, rr.top() + 5), 2.6, 2.6)
