from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFontMetrics, QIntValidator, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListView, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
    QProgressBar, QScrollArea, QSizePolicy, QSplitter, QStyle, QStyledItemDelegate, QVBoxLayout, QWidget,
)

from dentai.config import DEFAULT_WEIGHTS, load_classes
from dentai.fdi import FDI_NAMES
from dentai.imaging import IMAGE_EXTS, imread, imwrite
from dentai.inference.draw import draw_analysis
from dentai.inference.pipeline import DentalAnalyzer
from dentai.inference.results import DISCLAIMER, AnalysisResult, RawResult
from dentai.records import GENERAL, NotesStore, PatientStore, image_key, save_setting
from dentai.ui import theme
from dentai.ui.controls import AnimatedButton, AnimSlider, PopupPanel, ThemeSwitch, ToggleChip, draw_icon, qc
from dentai.ui.logo import app_icon, draw_logo
from dentai.ui.widgets import FINDING_PRIORITY, Card, FdiBar, ImageViewer, StatCard, dropped_image_path


ERROR_LOG = Path(__file__).resolve().parents[2] / "logs" / "hata.log"


def log_error(where: str, detail: str) -> str:
    """Hatanın tam ayrıntısını logs/hata.log'a ekler; kullanıcıya gösterilecek kısa metni döndürür."""
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG.open("a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S}  {where} =====\n{detail}\n")
    except OSError:
        pass
    return detail.strip().splitlines()[-1] if detail.strip() else where


def tr_fold(s: str) -> str:
    """Türkçe büyük/küçük harf duyarsız arama için."""
    return s.replace("I", "ı").replace("İ", "i").lower()


def adjust_image(img: np.ndarray, brightness: int, contrast: int) -> np.ndarray:
    """brightness: -100…100, contrast: yüzde (100 = değişmez). Orta gri etrafında germe + kaydırma."""
    if brightness == 0 and contrast == 100:
        return img
    lut = np.clip((np.arange(256, dtype=np.float32) - 128) * (contrast / 100) + 128 + brightness * 1.28, 0, 255)
    return cv2.LUT(img, lut.astype(np.uint8))


class WarmupWorker(QThread):
    """Pencere açılınca modelleri arka planda yükler ve boş bir görüntüyle bir kez çalıştırır.

    İlk tahmin; torch/torchvision modüllerinin yüklenmesi ve ağırlıkların hazırlanması nedeniyle
    çok daha uzun sürer. Bu iş kullanıcı "Analiz Et"e basmadan önce bitmiş olur.
    """

    failed = Signal(str)

    def __init__(self, analyzer: DentalAnalyzer):
        super().__init__()
        self.analyzer = analyzer

    def run(self):
        try:
            self.analyzer.load()
            self.analyzer.infer(np.zeros((640, 1280, 3), np.uint8))
        except Exception:
            self.failed.emit(log_error("model hazırlığı", traceback.format_exc()))


class AnalysisWorker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, analyzer: DentalAnalyzer, image):
        super().__init__()
        self.analyzer, self.image = analyzer, image

    def run(self):
        try:
            self.done.emit(self.analyzer.infer(self.image))
        except Exception:  # modeli/CUDA'yı kullanıcıya anlaşılır göstermek için
            self.failed.emit(log_error(f"analiz (görüntü {self.image.shape})", traceback.format_exc()))


class IconMark(QWidget):
    """Araç çubuğundaki küçük, temaya uyan ikon."""

    def __init__(self, name: str, tip: str = ""):
        super().__init__()
        self.name = name
        self.setFixedSize(20, 20)
        self.setToolTip(tip)

    def paintEvent(self, e):
        draw_icon(QPainter(self), self.name, QRectF(1, 1, 18, 18), qc(theme.MUTED), 1.7)


class LogoMark(QWidget):
    def __init__(self, size: int):
        super().__init__()
        self.setFixedSize(size, size)

    def paintEvent(self, e):
        draw_logo(QPainter(self), QRectF(self.rect()))


def _vsep() -> QWidget:
    w = QWidget()
    w.setObjectName("vsep")
    w.setFixedSize(1, 26)
    return w


COLOR_ROLE = Qt.ItemDataRole.UserRole + 2     # bulgu satırının çerçeve rengi (#RRGGBB)


class FindingDelegate(QStyledItemDelegate):
    """Klinik bulgular: her satır, bulgunun renginde kalın çerçeveli yuvarlatılmış bir kutu."""

    PAD_X, PAD_Y, MARGIN = 14, 9, 4

    def __init__(self, view: QListWidget):
        super().__init__(view)
        self.view = view

    def _text_rect(self, width: int) -> QRect:
        return QRect(0, 0, max(60, width - 2 * (self.PAD_X + self.MARGIN) - 4), 10_000)

    def sizeHint(self, opt, idx) -> QSize:
        w = self.view.viewport().width()
        fm = QFontMetrics(opt.font)
        br = fm.boundingRect(self._text_rect(w), int(Qt.TextFlag.TextWordWrap), idx.data() or "")
        return QSize(w, br.height() + 2 * self.PAD_Y + 2 * self.MARGIN)

    def paint(self, p: QPainter, opt, idx) -> None:
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(opt.rect).adjusted(self.MARGIN, self.MARGIN, -self.MARGIN, -self.MARGIN)
        color = idx.data(COLOR_ROLE)
        hover = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        sel = bool(opt.state & QStyle.StateFlag.State_Selected)
        if color:
            c = QColor(color)
            fill = QColor(c)
            fill.setAlphaF(0.20 if sel else 0.13 if hover else 0.07)
            p.setBrush(fill)
            p.setPen(QPen(c, 3.0 if (hover or sel) else 2.6))
            p.drawRoundedRect(r.adjusted(1.5, 1.5, -1.5, -1.5), 10, 10)
            p.setPen(QColor(theme.TEXT))
        else:                                    # bilgi satırı (bulgu yok / analiz yapılmadı)
            p.setPen(QColor(theme.MUTED))
        p.setFont(opt.font)
        p.drawText(r.adjusted(self.PAD_X, self.PAD_Y - 1, -self.PAD_X, -self.PAD_Y + 1),
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
                   idx.data() or "")
        p.restore()


class AddPatientDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Yeni hasta")
        self.setMinimumWidth(420)
        self.xray: Path | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(8)
        t = QLabel("Yeni hasta ekle")
        t.setObjectName("formTitle")
        lay.addWidget(t)
        lay.addSpacing(6)
        for caption, attr, ph in (("Ad soyad", "name", "örn. Ayşe Yılmaz"), ("Doğum yılı (isteğe bağlı)", "year", "örn. 1994")):
            c = QLabel(caption)
            c.setObjectName("h")
            lay.addWidget(c)
            le = QLineEdit()
            le.setPlaceholderText(ph)
            setattr(self, attr, le)
            lay.addWidget(le)
        self.year.setValidator(QIntValidator(1900, 2100, self))
        c = QLabel("Panoramik röntgen (isteğe bağlı)")
        c.setObjectName("h")
        lay.addWidget(c)
        row = QHBoxLayout()
        self.xray_lbl = QLabel("Seçilmedi")
        self.xray_lbl.setObjectName("muted")
        pick = AnimatedButton("Dosya seç", "upload")
        pick.clicked.connect(self._pick)
        row.addWidget(self.xray_lbl, 1)
        row.addWidget(pick)
        lay.addLayout(row)
        self.err = QLabel("")
        self.err.setObjectName("error")
        lay.addWidget(self.err)
        bb = QDialogButtonBox()
        ok = AnimatedButton("Hastayı ekle", "user_plus", kind="primary")
        cancel = AnimatedButton("Vazgeç")
        bb.addButton(cancel, QDialogButtonBox.ButtonRole.RejectRole)
        bb.addButton(ok, QDialogButtonBox.ButtonRole.AcceptRole)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.name.setFocus()

    def _pick(self):
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        path, _ = QFileDialog.getOpenFileName(self, "Panoramik röntgen seç", "", f"Görüntü ({exts})")
        if path:
            self.xray = Path(path)
            self.xray_lbl.setText(self.xray.name)

    def _accept(self):
        if not self.name.text().strip():
            self.err.setText("Hasta adı gerekli.")
            self.name.setFocus()
            return
        self.accept()


class MainWindow(QMainWindow):
    logoutRequested = Signal()

    def __init__(self, weights: dict[str, Path] | None = None, user: str = "",
                 patients: PatientStore | None = None, notes: NotesStore | None = None):
        super().__init__()
        self.setWindowTitle("DENT-AI — Panoramik Röntgen Karar Destek")
        self.setWindowIcon(app_icon())
        # Pencere ekrandan büyük açılırsa alt kısım (FDI çubuğu) ekranın dışında kalır: ekrana sığdır.
        scr = QApplication.primaryScreen()
        avail = scr.availableGeometry() if scr is not None else None
        if avail is not None and avail.width() > 0:
            self.resize(min(1600, avail.width() - 40), min(960, avail.height() - 50))
        else:
            self.resize(1600, 960)
        self.setAcceptDrops(True)

        self.user = user
        self.logout_requested = False
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.image = None
        self.image_path: Path | None = None
        self.raw: RawResult | None = None
        self.result: AnalysisResult | None = None
        self.worker: AnalysisWorker | None = None
        self.warmer: WarmupWorker | None = None
        self._old_threads: list[QThread] = []   # model değişince yarıda kalan hazırlıklar
        self._pending_run = False
        self.classes = load_classes()
        self.analyzer = self._make_analyzer()
        self.patients = patients or PatientStore()
        self.notes = notes or NotesStore()
        self.current_patient: str | None = None
        self._note_ctx: tuple[Path, str] | None = None   # editördeki metnin ait olduğu (röntgen, hedef)
        self._note_dirty = False

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._build_sidebar())

        content = QWidget()
        content.setObjectName("center")
        v = QVBoxLayout(content)
        v.setContentsMargins(16, 14, 16, 8)
        v.setSpacing(12)
        v.addWidget(self._build_toolbar())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(10)
        stage = QFrame()
        stage.setObjectName("card")
        sl = QVBoxLayout(stage)
        sl.setContentsMargins(0, 0, 0, 0)
        self.viewer = ImageViewer()
        self.viewer.fileDropped.connect(self.load_image)
        self.viewer.setStyleSheet("border-radius: 12px;")
        sl.addWidget(self.viewer)
        cl.addWidget(stage, 1)
        self.fdi_bar = FdiBar()
        self.fdi_bar.toothClicked.connect(self.toggle_tooth)
        cl.addWidget(self.fdi_bar)
        split.addWidget(center)
        split.addWidget(self._build_side())
        split.setStretchFactor(0, 1)
        split.setSizes([1000, 400])
        v.addWidget(split, 1)
        h.addWidget(content, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(160)
        self.progress.hide()
        self.lbl_ready = QLabel("")
        self.statusBar().addPermanentWidget(self.lbl_ready)
        self.statusBar().addPermanentWidget(self.progress)

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(12)
        self._render_timer.timeout.connect(self._render_now)
        self._note_timer = QTimer(self)
        self._note_timer.setSingleShot(True)
        self._note_timer.setInterval(700)
        self._note_timer.timeout.connect(self._save_note)

        self._build_menu()
        self._fill_patients()
        self._update_model_status()
        self._load_note_editor()
        self.refresh()
        self._start_warmup()

    # ---- kurulum -------------------------------------------------------------------
    def _make_analyzer(self) -> DentalAnalyzer:
        return DentalAnalyzer(self.weights["teeth"], self.weights["findings"], self.weights["age"])

    def _start_warmup(self) -> None:
        if self.warmer is not None:          # eski analizörün hazırlığı bitene kadar referansı tut
            old = self.warmer
            self._old_threads.append(old)
            old.finished.connect(lambda t=old: self._old_threads.remove(t))
            old.finished.disconnect(self._warmup_done)
            self.warmer = None
        a = self.analyzer.available
        if not (a["teeth"] or a["findings"] or a["age"]):
            return
        self._warm_error = None
        self.warmer = WarmupWorker(self.analyzer)
        self.warmer.finished.connect(self._warmup_done)
        self.warmer.failed.connect(self._warmup_failed)
        self.warmer.start()
        self.lbl_ready.setText("Modeller hazırlanıyor…")

    def _warmup_failed(self, msg: str) -> None:
        self._warm_error = msg
        self.statusBar().showMessage(f"Model hazırlığı başarısız: {msg}")

    def _warmup_done(self) -> None:
        self.warmer = None
        self.lbl_ready.setText("Model hazırlığı başarısız" if self._warm_error
                               else f"Modeller hazır · {self.analyzer.device}")
        if self._pending_run:
            self._pending_run = False
            self.run_analysis()

    # ---- sol panel: hastalar -----------------------------------------------------------
    def _build_sidebar(self) -> QWidget:
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(272)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(18, 18, 18, 14)
        lay.setSpacing(10)

        brand = QHBoxLayout()
        brand.setSpacing(10)
        brand.addWidget(LogoMark(44))
        names = QVBoxLayout()
        names.setSpacing(0)
        trow = QHBoxLayout()
        trow.setSpacing(0)
        t1 = QLabel("DENT-")
        t1.setObjectName("title")
        t2 = QLabel("AI")
        t2.setObjectName("brandAccent")
        trow.addWidget(t1)
        trow.addWidget(t2)
        trow.addStretch(1)
        names.addLayout(trow)
        sub = QLabel("Panoramik röntgen · ikinci göz")
        sub.setObjectName("subtitle")
        names.addWidget(sub)
        brand.addLayout(names, 1)
        lay.addLayout(brand)
        lay.addSpacing(8)

        self.btn_add_patient = AnimatedButton("Hasta Ekle", "user_plus", kind="primary")
        self.btn_add_patient.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_add_patient.clicked.connect(self.add_patient)
        lay.addWidget(self.btn_add_patient)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Hasta ara…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_patients)
        lay.addWidget(self.search)

        self.lbl_patients = QLabel("HASTALAR")
        self.lbl_patients.setObjectName("h")
        lay.addWidget(self.lbl_patients)
        self.patient_list = QListWidget()
        self.patient_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.patient_list.customContextMenuRequested.connect(self._patient_menu)
        self.patient_list.itemClicked.connect(self._patient_clicked)
        lay.addWidget(self.patient_list, 1)

        self.lbl_xrays = QLabel("RÖNTGENLER")
        self.lbl_xrays.setObjectName("h")
        lay.addWidget(self.lbl_xrays)
        self.xray_list = QListWidget()
        self.xray_list.setMaximumHeight(150)
        self.xray_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.xray_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.patient_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.xray_list.itemClicked.connect(self._xray_clicked)
        lay.addWidget(self.xray_list)

        foot = QHBoxLayout()
        who = QLabel(f"Oturum: <b>{self.user}</b>" if self.user else "Oturum açılmadı")
        who.setObjectName("muted")
        foot.addWidget(who, 1)
        out = AnimatedButton("", "logout", kind="icon", tooltip="Oturumu kapat")
        out.clicked.connect(self.logout)
        out.setVisible(bool(self.user))
        foot.addWidget(out)
        lay.addLayout(foot)
        return side

    def _fill_patients(self, select: str | None = None) -> None:
        select = select or self.current_patient
        self.patient_list.clear()
        for p in reversed(self.patients.patients):            # en yeni üstte
            meta = []
            if p.birth_year:
                meta.append(f"Doğum {p.birth_year}")
            meta.append(f"{len(p.xrays)} röntgen" if p.xrays else "röntgen yok")
            it = QListWidgetItem(f"{p.name}\n{' · '.join(meta)}")
            it.setData(Qt.ItemDataRole.UserRole, p.id)
            self.patient_list.addItem(it)
            if p.id == select:
                it.setSelected(True)
                self.patient_list.setCurrentItem(it)
        self.lbl_patients.setText(f"HASTALAR  ·  {len(self.patients.patients)}")
        self._filter_patients(self.search.text())
        self._fill_xrays()

    def _filter_patients(self, text: str) -> None:
        q = tr_fold(text.strip())
        for i in range(self.patient_list.count()):
            it = self.patient_list.item(i)
            it.setHidden(bool(q) and q not in tr_fold(it.text()))

    def _fill_xrays(self) -> None:
        self.xray_list.clear()
        p = self.patients.get(self.current_patient) if self.current_patient else None
        self.lbl_xrays.setVisible(p is not None)
        self.xray_list.setVisible(p is not None)
        if p is None:
            return
        cur = image_key(self.image_path) if self.image_path else None
        for x in reversed(p.xrays):
            path = Path(x)
            it = QListWidgetItem(path.name + ("" if path.exists() else "  (bulunamadı)"))
            it.setData(Qt.ItemDataRole.UserRole, x)
            it.setToolTip(x)
            self.xray_list.addItem(it)
            if cur and image_key(x) == cur:
                it.setSelected(True)
        if not p.xrays:
            it = QListWidgetItem("Henüz röntgen yok — Röntgen Yükle")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.xray_list.addItem(it)

    def add_patient(self) -> None:
        dlg = AddPatientDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        p = self.patients.add(dlg.name.text(), dlg.year.text(), dlg.xray)
        self.current_patient = p.id
        self.search.clear()
        self._fill_patients(p.id)
        self.statusBar().showMessage(f"Hasta eklendi: {p.name}")
        if dlg.xray:
            self.load_image(dlg.xray)

    def _patient_clicked(self, it: QListWidgetItem) -> None:
        self.open_patient(it.data(Qt.ItemDataRole.UserRole))

    def open_patient(self, pid: str) -> None:
        p = self.patients.get(pid)
        if p is None:
            return
        self.current_patient = pid
        self._fill_xrays()
        latest = p.latest_xray
        if latest is None:
            self.statusBar().showMessage(f"{p.name}: kayıtlı röntgen yok. “Röntgen Yükle” ile ya da sürükleyerek ekleyin.")
            self._update_title()
            return
        if not latest.exists():
            QMessageBox.warning(self, "DENT-AI", f"Röntgen dosyası bulunamadı:\n{latest}")
            return
        self.load_image(latest)

    def _xray_clicked(self, it: QListWidgetItem) -> None:
        x = it.data(Qt.ItemDataRole.UserRole)
        if not x:
            return
        if not Path(x).exists():
            QMessageBox.warning(self, "DENT-AI", f"Röntgen dosyası bulunamadı:\n{x}")
            return
        self.load_image(Path(x))

    def _patient_menu(self, pos: QPoint) -> None:
        it = self.patient_list.itemAt(pos)
        if it is None:
            return
        pid = it.data(Qt.ItemDataRole.UserRole)
        p = self.patients.get(pid)
        m = QMenu(self)
        a_add = m.addAction("Röntgen ekle…")
        a_del = m.addAction("Hastayı sil")
        act = m.exec(self.patient_list.mapToGlobal(pos))
        if act == a_add:
            self.current_patient = pid
            self._fill_patients(pid)
            self.open_dialog()
        elif act == a_del:
            ok = QMessageBox.question(self, "Hastayı sil",
                                      f"{p.name} kaydı silinsin mi?\n(Röntgen dosyaları ve notlar silinmez.)")
            if ok == QMessageBox.StandardButton.Yes:
                self.patients.remove(pid)
                if self.current_patient == pid:
                    self.current_patient = None
                self._fill_patients()
                self._update_title()

    def logout(self) -> None:
        self.logout_requested = True
        self.logoutRequested.emit()
        self.close()

    # ---- üst araç çubuğu ---------------------------------------------------------------
    def _build_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("toolbar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)

        self.btn_open = AnimatedButton("Röntgen Yükle", "upload", tooltip="Panoramik röntgen aç (sürükle-bırak da olur)")
        self.btn_open.clicked.connect(self.open_dialog)
        lay.addWidget(self.btn_open)
        lay.addWidget(_vsep())

        lay.addWidget(IconMark("brightness", "Parlaklık"))
        self.sl_bright = AnimSlider(-100, 100, 0, centered=True)
        self.sl_bright.setFixedWidth(104)
        self.sl_bright.setToolTip("Parlaklık  (çift tık: sıfırla)")
        self.sl_bright.valueChanged.connect(self._schedule_render)
        lay.addWidget(self.sl_bright)
        lay.addSpacing(4)
        lay.addWidget(IconMark("contrast", "Kontrast"))
        self.sl_contrast = AnimSlider(40, 220, 100, centered=True)
        self.sl_contrast.setFixedWidth(104)
        self.sl_contrast.setToolTip("Kontrast  (çift tık: sıfırla)")
        self.sl_contrast.valueChanged.connect(self._schedule_render)
        lay.addWidget(self.sl_contrast)
        lay.addWidget(_vsep())

        self.btn_zoom_out = AnimatedButton("", "minus", kind="icon", tooltip="Uzaklaştır")
        self.btn_zoom_in = AnimatedButton("", "plus", kind="icon", tooltip="Yakınlaştır")
        self.btn_fit = AnimatedButton("Sıfırla", "fit", tooltip="Röntgeni alana sığdır")
        self.btn_zoom_out.clicked.connect(lambda: self.viewer.zoom_by(1 / 1.35))
        self.btn_zoom_in.clicked.connect(lambda: self.viewer.zoom_by(1.35))
        self.btn_fit.clicked.connect(self._reset_view)
        for b in (self.btn_zoom_out, self.btn_zoom_in, self.btn_fit):
            lay.addWidget(b)
        lay.addWidget(_vsep())

        self.btn_sens = AnimatedButton("Hassasiyet · %25", "sliders", tooltip="Güven eşiği")
        self.btn_sens.clicked.connect(self._toggle_sens)
        lay.addWidget(self.btn_sens)
        self._build_sens_popup()

        lay.addStretch(1)
        self.btn_export = AnimatedButton("", "download", kind="icon", tooltip="Dışa aktar (PNG + JSON rapor)")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export)
        self.btn_run = AnimatedButton("Analiz Et", "play", kind="primary")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_analysis)
        self.theme_switch = ThemeSwitch(theme.DARK_MODE)
        self.theme_switch.toggled.connect(self.set_dark)
        lay.addWidget(self.btn_export)
        lay.addWidget(self.btn_run)
        lay.addSpacing(4)
        lay.addWidget(self.theme_switch)
        return bar

    def _build_sens_popup(self) -> None:
        self.sens_popup = PopupPanel(self)
        b = self.sens_popup.body
        row = QHBoxLayout()
        cap = QLabel("GÜVEN EŞİĞİ")
        cap.setObjectName("h")
        self.conf_val = QLabel("%25")
        self.conf_val.setObjectName("big")
        self.conf_val.setStyleSheet("font-size: 18px;")
        row.addWidget(cap)
        row.addStretch(1)
        row.addWidget(self.conf_val)
        b.addLayout(row)
        self.slider = AnimSlider(10, 90, 25)
        self.slider.setFixedWidth(250)
        self.slider.valueChanged.connect(self._conf_changed)
        b.addWidget(self.slider)
        hint = QLabel("Düşük eşik daha çok bulgu gösterir (yanlış alarm artar).\n"
                      "Yüksek eşik yalnızca modelin emin olduklarını.\nÇift tık: varsayılan (%25)")
        hint.setObjectName("muted")
        b.addWidget(hint)
        self.sens_popup.closed.connect(lambda: self.btn_sens.target(check=0.0))

    def _toggle_sens(self) -> None:
        if self.sens_popup.isVisible():
            self.sens_popup.hide()
            return
        self.btn_sens.target(check=1.0)
        self.sens_popup.show_below(self.btn_sens)

    def _reset_view(self) -> None:
        self.fdi_bar.set_selected(None)
        self.viewer.reset_view()

    def set_dark(self, dark: bool) -> None:
        theme.set_mode(dark)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(theme.qss())
        self.viewer.apply_theme()
        for w in self.findChildren(QWidget):
            w.update()
        self._fill_findings()
        save_setting("theme", "dark" if dark else "light")

    # ---- sağ panel ------------------------------------------------------------------
    def _build_side(self) -> QWidget:
        side = QWidget()
        side.setObjectName("side")
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(12)

        stats = QHBoxLayout()
        g = self.classes.groups
        self.card_caries = StatCard("Çürük", "#%02X%02X%02X" % g["caries"].color_rgb)
        self.card_resto = StatCard("Restorasyon", "#%02X%02X%02X" % g["restoration"].color_rgb)
        self.card_imp = StatCard("Gömülü diş", "#%02X%02X%02X" % g["impacted"].color_rgb)
        for c in (self.card_caries, self.card_resto, self.card_imp):
            stats.addWidget(c)
        lay.addLayout(stats)

        view = Card("Görünüm")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self.layer_boxes: dict[str, ToggleChip] = {}
        chips = [(c.name, c.label, "#%02X%02X%02X" % c.color_rgb) for c in self.classes.findings]
        chips.append(("teeth", "Diş numaraları", "#8DA3B5"))
        for i, (key, label, color) in enumerate(chips):
            chip = ToggleChip(label, color)
            chip.toggled.connect(self._schedule_render)
            self.layer_boxes[key] = chip
            grid.addWidget(chip, i // 2, i % 2)
        view.body.addLayout(grid)
        cap = QLabel("YAZILAR")
        cap.setObjectName("h")
        view.body.addSpacing(2)
        view.body.addWidget(cap)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.cb_labels = ToggleChip("Etiket", theme.PRIMARY)
        self.cb_conf = ToggleChip("Güven yüzdesi", "#5B84B1", checked=False)
        for cb in (self.cb_labels, self.cb_conf):
            cb.toggled.connect(self._schedule_render)
            row.addWidget(cb)
        view.body.addLayout(row)
        lay.addWidget(view)

        finds = Card("Klinik bulgular")
        self.findings_list = QListWidget()
        self.findings_list.setObjectName("findings")
        self.findings_list.setWordWrap(True)
        self.findings_list.setMinimumHeight(200)
        self.findings_list.setMouseTracking(True)
        self.findings_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.findings_list.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.findings_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.findings_list.setItemDelegate(FindingDelegate(self.findings_list))
        self.findings_list.itemClicked.connect(self._finding_clicked)
        finds.body.addWidget(self.findings_list)
        lay.addWidget(finds, 1)

        note = Card("Hekim teşhis notu")
        trow = QHBoxLayout()
        self.note_target = QComboBox()
        self.note_target.addItem("Röntgenin geneli", GENERAL)
        for fdi in sorted(FDI_NAMES, key=int):
            self.note_target.addItem(f"Diş {fdi}", fdi)
        self.note_target.currentIndexChanged.connect(self._note_target_changed)
        trow.addWidget(QLabel("Not:"))
        trow.addWidget(self.note_target, 1)
        note.body.addLayout(trow)
        self.note_edit = QPlainTextEdit()
        self.note_edit.setPlaceholderText("Teşhis, tedavi planı veya bu dişe dair notunuz…")
        self.note_edit.setMinimumHeight(110)
        self.note_edit.textChanged.connect(self._note_changed)
        note.body.addWidget(self.note_edit)
        nrow = QHBoxLayout()
        self.note_status = QLabel("")
        self.note_status.setObjectName("muted")
        self.btn_note_save = AnimatedButton("Kaydet", "check")
        self.btn_note_save.clicked.connect(self._save_note)
        nrow.addWidget(self.note_status, 1)
        nrow.addWidget(self.btn_note_save)
        note.body.addLayout(nrow)
        self.noted_lbl = QLabel("")
        self.noted_lbl.setObjectName("muted")
        self.noted_lbl.setWordWrap(True)
        note.body.addWidget(self.noted_lbl)
        lay.addWidget(note)

        age = Card()
        arow = QHBoxLayout()
        acap = QLabel("DENTAL YAŞ TAHMİNİ")
        acap.setObjectName("h")
        self.age_label = QLabel("–")
        self.age_label.setObjectName("big")
        self.age_label.setStyleSheet("font-size: 18px;")
        arow.addWidget(acap)
        arow.addStretch(1)
        arow.addWidget(self.age_label)
        age.body.addLayout(arow)
        lay.addWidget(age)

        d = QLabel(DISCLAIMER)
        d.setObjectName("disclaimer")
        d.setWordWrap(True)
        lay.addWidget(d)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(side)
        scroll.setMinimumWidth(390)
        return scroll

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("Modeller")
        for key, title in [("teeth", "Diş numaralandırma modeli (.pt)…"),
                           ("findings", "Bulgu modeli (.pt)…"), ("age", "Yaş modeli (.pt)…")]:
            act = QAction(title, self)
            act.triggered.connect(lambda _=False, k=key: self.pick_weights(k))
            m.addAction(act)

    def closeEvent(self, e):
        self._flush_note()
        self.sens_popup.hide()
        # Çalışan QThread nesnesi yok edilirse Qt programı çökertir: önce pencereyi gizle, işler bitsin.
        self.hide()
        for t in [self.warmer, self.worker, *self._old_threads]:
            if t is not None:
                t.wait()
        super().closeEvent(e)

    # ---- model yönetimi ---------------------------------------------------------------
    def pick_weights(self, key: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Model seç", str(Path(self.weights[key]).parent), "PyTorch (*.pt)")
        if path:
            if self.worker is not None:
                QMessageBox.information(self, "DENT-AI", "Analiz sürerken model değiştirilemez.")
                return
            self.weights[key] = Path(path)
            self.analyzer = self._make_analyzer()
            self.raw = self.result = None
            self._update_model_status()
            self.refresh()
            self._start_warmup()

    def _update_model_status(self) -> None:
        a = self.analyzer.available
        mark = lambda ok: "✓" if ok else "✗"
        self.statusBar().showMessage(
            f"Modeller:  Diş {mark(a['teeth'])}   Bulgu {mark(a['findings'])}   Yaş {mark(a['age'])}"
            f"   |   ✗ olanlar ilgili çıktıyı üretmez (models/ klasörüne .pt koyun)")

    # ---- görüntü ---------------------------------------------------------------------
    def open_dialog(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        path, _ = QFileDialog.getOpenFileName(self, "Panoramik röntgen seç", "", f"Görüntü ({exts})")
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        path = Path(path)
        if self.worker is not None or self._pending_run:   # sonuç yanlış görüntüye yazılmasın
            self.statusBar().showMessage("Analiz sürüyor; bitince yeni görüntüyü yükleyin.")
            return
        try:
            image = imread(path)
        except ValueError as exc:
            QMessageBox.warning(self, "DENT-AI", str(exc))
            return
        self._flush_note()
        attached = ""
        p = self.patients.get(self.current_patient) if self.current_patient else None
        if p is not None:
            known = {image_key(x) for x in p.xrays}
            if image_key(path) not in known:
                # Hastanın zaten röntgeni varsa yanlışlıkla başka birinin röntgeni eklenmesin diye sor.
                ans = QMessageBox.StandardButton.Yes if not p.xrays else QMessageBox.question(
                    self, "Röntgeni hastaya ekle",
                    f"{path.name}\n\nBu röntgen {p.name} kaydına eklensin mi?\n"
                    "Hayır: hastaya eklenmeden yalnızca açılır.")
                if ans == QMessageBox.StandardButton.Yes:
                    self.patients.attach_xray(p.id, path)
                    attached = f"  ·  {p.name} kaydına eklendi"
                else:
                    self.current_patient = None
        self.image = image
        self.image_path = path
        self.raw = self.result = None
        self.fdi_bar.set_selected(None)
        self.viewer.set_image(adjust_image(self.image, self.sl_bright.value(), self.sl_contrast.value()),
                              reset_view=True)
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(False)
        self._fill_patients()
        self._update_title()
        self.note_target.setCurrentIndex(0)
        self._load_note_editor()
        self.refresh()
        self.statusBar().showMessage(f"{path.name}  ·  {self.image.shape[1]}×{self.image.shape[0]} px{attached}")

    def _update_title(self) -> None:
        p = self.patients.get(self.current_patient) if self.current_patient else None
        parts = [x for x in (p.name if p else None, self.image_path.name if self.image_path else None) if x]
        self.setWindowTitle("DENT-AI — " + (" · ".join(parts) if parts else "Panoramik Röntgen Karar Destek"))

    # Görüntüleyicinin dışındaki alanlara (yan panel, başlık) bırakılan dosyalar için.
    def dragEnterEvent(self, e):
        if dropped_image_path(e.mimeData()) is not None:
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if dropped_image_path(e.mimeData()) is not None:
            e.acceptProposedAction()

    def dropEvent(self, e):
        p = dropped_image_path(e.mimeData())
        if p is not None:
            e.acceptProposedAction()
            self.load_image(p)

    # ---- analiz ----------------------------------------------------------------------
    def run_analysis(self) -> None:
        a = self.analyzer.available
        if self.image is None or self.worker is not None:
            return
        if not (a["teeth"] or a["findings"]):
            QMessageBox.information(
                self, "Model bulunamadı",
                "models/ klasöründe teeth_best.pt veya findings_best.pt yok.\n"
                "Colab'da eğitip indirin ya da 'Modeller' menüsünden seçin.")
            return
        self.btn_run.setEnabled(False)
        self.progress.show()
        if self.warmer is not None:           # hazırlık bitince kendiliğinden başlar
            self._pending_run = True
            self.statusBar().showMessage("Modeller hazırlanıyor; hazır olunca analiz otomatik başlayacak…")
            return
        self.statusBar().showMessage("Analiz ediliyor…")
        self.worker = AnalysisWorker(self.analyzer, self.image)
        self.worker.done.connect(self._analysis_done)
        self.worker.failed.connect(self._analysis_failed)
        self.worker.start()

    def _finish_worker(self) -> None:
        self.progress.hide()
        self.btn_run.setEnabled(True)
        if self.worker:
            self.worker.wait()
            self.worker = None

    def _analysis_done(self, raw: RawResult) -> None:
        self._finish_worker()
        self.raw = raw
        self.btn_export.setEnabled(True)
        self.refresh()
        self.statusBar().showMessage(f"Analiz tamamlandı · {raw.elapsed_ms:.0f} ms · cihaz: {self.analyzer.device}")

    def _analysis_failed(self, msg: str) -> None:
        self._finish_worker()
        self.statusBar().showMessage("Analiz başarısız")
        QMessageBox.critical(self, "Analiz hatası", f"{msg}\n\nAyrıntılar kaydedildi:\n{ERROR_LOG}")

    # ---- görüntüleme ------------------------------------------------------------------
    def _conf_changed(self, v: int) -> None:
        self.conf_val.setText(f"%{v}")
        self.btn_sens.setText(f"Hassasiyet · %{v}")
        self.btn_sens.updateGeometry()
        self.refresh()

    def refresh(self) -> None:
        self.result = DentalAnalyzer.assemble(self.raw, self.slider.value() / 100) if self.raw else None
        r = self.result
        c = r.counts() if r else None
        self.card_caries.set_value("–" if not c else str(c["caries"]))
        self.card_resto.set_value("–" if not c else str(c["restoration"]))
        self.card_imp.set_value("–" if not c else str(c["impacted"]))
        if r and r.age is not None:
            self.age_label.setText(f"{r.age:.1f} yıl")
        else:
            self.age_label.setText("–" if not r else "model yok")
        self.fdi_bar.set_result(r)
        self._fill_findings()
        self._render_now()

    def _schedule_render(self, *_) -> None:
        self._render_timer.start()

    def _render_now(self) -> None:
        if self.image is None:
            return
        base = adjust_image(self.image, self.sl_bright.value(), self.sl_contrast.value())
        if self.result is None:
            self.viewer.set_image(base)
            return
        layers = {k for k, cb in self.layer_boxes.items() if cb.isChecked()}
        self.viewer.set_image(draw_analysis(base, self.result, layers, show_labels=self.cb_labels.isChecked(),
                                            show_conf=self.cb_conf.isChecked()))

    render = _render_now

    def _fill_findings(self) -> None:
        lst = self.findings_list
        lst.clear()
        r = self.result

        def info(text: str):
            it = QListWidgetItem(text)
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            it.setForeground(QColor(theme.MUTED))
            lst.addItem(it)

        if self.image is None:
            info("Bir hasta seçin ya da röntgen yükleyin.")
            return
        if r is None:
            info("Analiz sonrası, hangi dişte ne bulunduğu burada listelenir.")
            return
        if not r.findings:
            info(f"%{round(r.conf_threshold * 100)} güven eşiğinde bulgu yok.")
            return
        rank = {n: i for i, n in enumerate(FINDING_PRIORITY)}
        by_tooth = r.by_tooth()
        for fdi in sorted(by_tooth, key=int):
            dets = sorted(by_tooth[fdi], key=lambda d: (rank.get(d.name, 9), -d.conf))
            parts = [f"{self._label(d.name)} (%{round(d.conf * 100)})" for d in dets]
            it = QListWidgetItem(f"{fdi} numaralı dişte:  " + ", ".join(parts))
            it.setData(COLOR_ROLE, self._color(dets[0].name))
            it.setData(Qt.ItemDataRole.UserRole, fdi)
            it.setToolTip("Tıklayın: dişe yakınlaştır · tekrar tıklayın: seçimi kaldır")
            lst.addItem(it)
        loose = [d for d, f in zip(r.findings, r.tooth_of) if not f]
        for d in sorted(loose, key=lambda d: (rank.get(d.name, 9), -d.conf)):
            it = QListWidgetItem(f"Dişle eşleşmeyen bulgu:  {self._label(d.name)} (%{round(d.conf * 100)})")
            it.setData(COLOR_ROLE, self._color(d.name))
            it.setData(Qt.ItemDataRole.UserRole + 1, list(d.xyxy))
            lst.addItem(it)

    def _color(self, name: str) -> str:
        spec = self.classes.by_name(name)
        return "#%02X%02X%02X" % spec.color_rgb if spec else theme.DANGER

    def _label(self, name: str) -> str:
        spec = self.classes.by_name(name)
        return spec.label if spec else name

    def _finding_clicked(self, it: QListWidgetItem) -> None:
        fdi = it.data(Qt.ItemDataRole.UserRole)
        if fdi:
            self.toggle_tooth(fdi)
            if self.fdi_bar.selected is None:
                self.findings_list.clearSelection()
            return
        box = it.data(Qt.ItemDataRole.UserRole + 1)
        if box:
            self.viewer.zoom_to_box(box, context=0.8)

    def toggle_tooth(self, fdi: str) -> None:
        """Dişe tıklandı: seçili değilse yakınlaştır; zaten seçiliyse seçimi kaldır ve görünümü sıfırla."""
        if self.fdi_bar.selected == fdi:
            self.note_target.setCurrentIndex(0)
            self.fdi_bar.set_selected(None)
            self.viewer.reset_view()
            self.statusBar().showMessage(f"Diş {fdi} seçimi kaldırıldı")
            return
        self.focus_tooth(fdi)

    def focus_tooth(self, fdi: str) -> None:
        """FDI çubuğunda veya bulgu listesinde tıklanan dişe yakınlaştırır ve not hedefini o diş yapar."""
        self.fdi_bar.set_selected(fdi)
        i = self.note_target.findData(fdi)
        if i >= 0 and self.image is not None:
            self.note_target.setCurrentIndex(i)
        if self.image is None:
            self.statusBar().showMessage("Önce bir röntgen açın.")
            return
        if self.result is None:
            self.statusBar().showMessage(f"Diş {fdi}: dişlerin yerini bulmak için önce “Analiz Et”e basın.")
            return
        boxes = [t.xyxy for t in self.result.teeth if t.name == fdi]
        if not boxes:
            boxes = [d.xyxy for d, f in zip(self.result.findings, self.result.tooth_of) if f == fdi]
        if not boxes:
            self.statusBar().showMessage(f"Diş {fdi} bu röntgende algılanmadı (eksik olabilir).")
            return
        b = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        self.viewer.zoom_to_box(b)
        self.statusBar().showMessage(f"Diş {fdi}")

    # ---- hekim notları ----------------------------------------------------------------
    def _note_changed(self) -> None:
        if self._note_ctx is None:
            return
        self._note_dirty = True
        self.note_status.setText("Kaydediliyor…")
        self._note_timer.start()

    def _save_note(self) -> None:
        self._note_timer.stop()
        if self._note_ctx is None or not self._note_dirty:
            if self._note_ctx is not None:
                self.note_status.setText(f"Kaydedildi ✓  {datetime.now():%H:%M}")
            return
        img, target = self._note_ctx
        try:
            self.notes.set(img, target, self.note_edit.toPlainText())
        except OSError as exc:
            self.note_status.setText("Kaydedilemedi!")
            QMessageBox.warning(self, "DENT-AI", f"Not kaydedilemedi:\n{exc}")
            return
        self._note_dirty = False
        self.note_status.setText(f"Kaydedildi ✓  {datetime.now():%H:%M}")
        self._update_noted()

    def _flush_note(self) -> None:
        if self._note_dirty:
            self._save_note()

    def _note_target_changed(self, _i: int) -> None:
        self._flush_note()
        self._load_note_editor()
        fdi = self.note_target.currentData()
        if fdi != GENERAL:
            self.fdi_bar.set_selected(fdi)

    def _load_note_editor(self) -> None:
        target = self.note_target.currentData() or GENERAL
        self._note_ctx = None                 # setPlainText "değişti" sayılmasın
        has = self.image_path is not None
        self.note_edit.setEnabled(has)
        self.btn_note_save.setEnabled(has)
        self.note_target.setEnabled(has)
        self.note_edit.setPlainText(self.notes.get(self.image_path, target) if has else "")
        self.note_edit.setPlaceholderText(
            "Önce bir röntgen açın." if not has else
            "Bu röntgene dair genel teşhis / tedavi planı…" if target == GENERAL else
            f"Diş {target} için notunuz…")
        self._note_dirty = False
        self.note_status.setText("")
        if has:
            self._note_ctx = (self.image_path, target)
        self._update_noted()

    def _update_noted(self) -> None:
        if self.image_path is None:
            self.fdi_bar.set_noted(set())
            self.noted_lbl.setText("")
            return
        teeth = self.notes.noted_teeth(self.image_path)
        self.fdi_bar.set_noted(teeth)
        parts = []
        if self.notes.get(self.image_path, GENERAL):
            parts.append("genel not")
        if teeth:
            parts.append("diş " + ", ".join(sorted(teeth, key=int)))
        self.noted_lbl.setText("Bu röntgende kayıtlı: " + " · ".join(parts) if parts else "")

    # ---- dışa aktarım -----------------------------------------------------------------
    def export(self) -> None:
        if self.result is None or self.image is None:
            return
        default = str((self.image_path.with_suffix("") if self.image_path else Path("dentai")).with_name(
            (self.image_path.stem if self.image_path else "dentai") + "_dentai.png"))
        path, _ = QFileDialog.getSaveFileName(self, "Dışa aktar", default, "PNG (*.png)")
        if not path:
            return
        out = Path(path)
        layers = {k for k, cb in self.layer_boxes.items() if cb.isChecked()}
        imwrite(out, draw_analysis(self.image, self.result, layers, show_labels=self.cb_labels.isChecked(),
                                   show_conf=self.cb_conf.isChecked()))
        report = self.result.to_dict(self.image_path.name if self.image_path else "")
        if self.image_path is not None:
            report["doctor_notes"] = {t: self.notes.get(self.image_path, t)
                                      for t in [GENERAL, *sorted(self.notes.noted_teeth(self.image_path), key=int)]
                                      if self.notes.get(self.image_path, t)}
        p = self.patients.get(self.current_patient) if self.current_patient else None
        if p is not None:
            report["patient"] = {"name": p.name, "birth_year": p.birth_year}
        out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self.statusBar().showMessage(f"Kaydedildi: {out}  (+ .json rapor)")


def install_error_hook() -> None:
    """Arayüzdeki (çizim, tablo, kaydırıcı...) yakalanmamış hataları da günlüğe yazar ve gösterir.
    PySide6'da yakalanmayan hata sessizce konsola düşer; kullanıcı yalnızca bir şeyin çalışmadığını görür."""
    def hook(exc_type, exc, tb):
        detail = "".join(traceback.format_exception(exc_type, exc, tb))
        if sys.__stderr__:
            sys.__stderr__.write(detail)
        short = log_error("arayüz", detail)
        try:
            QMessageBox.critical(None, "DENT-AI hatası", f"{short}\n\nAyrıntılar kaydedildi:\n{ERROR_LOG}")
        except Exception:
            pass
    sys.excepthook = hook
