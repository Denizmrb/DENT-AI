"""Renk teması. Aydınlık tema: Turkuaz/nane ana renk, derin lacivert başlıklar, buz mavisi zemin.

Renkler modül değişkenidir; set_mode() hepsini günceller. Kendini çizen bileşenler renkleri
çizim anında buradan okur, bu yüzden tema değişince yeniden çizilmeleri yeterlidir.
"""

from __future__ import annotations

LIGHT = {
    "PRIMARY": "#2BB8A6",        # ana renk: butonlar, vurgu
    "PRIMARY_HOVER": "#229C8C",
    "PRIMARY_SOFT": "#D6F2EE",   # ikincil vurgu: hover, çizgiler
    "NAVY": "#12395B",           # başlıklar, menü
    "BG": "#F4FAFB",             # arka plan
    "PANEL": "#FFFFFF",          # kartlar / form alanı
    "PANEL_HI": "#EAF5F4",
    "BORDER": "#DCEAEC",
    "TEXT": "#2E3A46",
    "MUTED": "#6B7C8C",
    "DANGER": "#E5584F",         # uyarı / hata
    "SUCCESS": "#3DBE7A",        # başarı
    "STAGE": "#0F2A40",          # röntgen sahnesi (gri tonlar koyu zeminde daha net seçilir)
    "STAGE_TEXT": "#9DB4C6",
    "STAGE_LINE": "#3C5A72",
    "BAR_BG": "#12395B",         # FDI çubuğu zemini
    "CELL": "#E6F0F3",
    "CELL_TEXT": "#2E3A46",
    "SIDEBAR": "#FFFFFF",
}

DARK = {
    "PRIMARY": "#2BB8A6",
    "PRIMARY_HOVER": "#3ACBB8",
    "PRIMARY_SOFT": "#17423F",
    "NAVY": "#BFE6E0",           # koyu temada başlıklar açık nane
    "BG": "#0B1A28",
    "PANEL": "#112638",
    "PANEL_HI": "#163248",
    "BORDER": "#1E3D55",
    "TEXT": "#DCE8EE",
    "MUTED": "#8DA3B5",
    "DANGER": "#EF6B62",
    "SUCCESS": "#4CCB88",
    "STAGE": "#06121D",
    "STAGE_TEXT": "#7F97AA",
    "STAGE_LINE": "#2A4258",
    "BAR_BG": "#0E2133",
    "CELL": "#D5E2E8",
    "CELL_TEXT": "#1B2733",
    "SIDEBAR": "#0E2131",
}

DARK_MODE = False
PRIMARY = PRIMARY_HOVER = PRIMARY_SOFT = NAVY = BG = PANEL = PANEL_HI = BORDER = ""
TEXT = MUTED = DANGER = SUCCESS = STAGE = STAGE_TEXT = STAGE_LINE = BAR_BG = CELL = CELL_TEXT = SIDEBAR = ""


def set_mode(dark: bool) -> None:
    global DARK_MODE
    DARK_MODE = bool(dark)
    globals().update(DARK if dark else LIGHT)


def blend(c1: str, c2: str, t: float) -> str:
    """İki #RRGGBB rengi arasında doğrusal karışım (t=0 → c1, t=1 → c2)."""
    t = max(0.0, min(1.0, t))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def qss() -> str:
    on_primary = "#FFFFFF"
    disabled = blend(PANEL, MUTED, 0.45)
    return f"""
* {{ font-family: "Segoe UI", "Inter", sans-serif; font-size: 13px; color: {TEXT}; }}
QMainWindow, QWidget#root, QDialog {{ background: {BG}; }}
QScrollArea {{ border: none; background: transparent; }}
QWidget#side, QWidget#center {{ background: {BG}; }}
QFrame#sidebar {{ background: {SIDEBAR}; border-right: 1px solid {BORDER}; }}
QFrame#card, QFrame#toolbar {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; }}
QFrame#popup {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; }}
QLabel#title {{ font-size: 20px; font-weight: 800; letter-spacing: 0.5px; color: {NAVY}; }}
QLabel#brandAccent {{ font-size: 20px; font-weight: 800; color: {PRIMARY}; }}
QLabel#subtitle {{ color: {MUTED}; font-size: 12px; }}
QLabel#h {{ color: {NAVY}; font-size: 11px; font-weight: 700; letter-spacing: 1.2px; }}
QLabel#big {{ font-size: 26px; font-weight: 700; color: {NAVY}; }}
QLabel#muted {{ color: {MUTED}; font-size: 12px; }}
QLabel#ok {{ color: {SUCCESS}; font-size: 12px; }}
QLabel#error {{ color: {DANGER}; font-size: 12px; font-weight: 600; }}
QLabel#disclaimer {{ color: {MUTED}; font-size: 11px; }}
QLabel#formTitle {{ font-size: 24px; font-weight: 800; color: {NAVY}; }}
QPushButton {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 9px; padding: 8px 16px;
}}
QPushButton:hover {{ border-color: {PRIMARY}; background: {PRIMARY_SOFT}; }}
QPushButton:disabled {{ color: {disabled}; background: {PANEL_HI}; }}
QPushButton#primary {{ background: {PRIMARY}; color: {on_primary}; border: none; font-weight: 600; }}
QPushButton#primary:hover {{ background: {PRIMARY_HOVER}; color: {on_primary}; }}
QPushButton#primary:disabled {{ background: {blend(PRIMARY, PANEL, 0.55)}; color: {on_primary}; }}
QLineEdit, QPlainTextEdit, QComboBox {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 9px; padding: 7px 10px;
  selection-background-color: {PRIMARY}; selection-color: {on_primary};
}}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{ border: 1.5px solid {PRIMARY}; }}
QLineEdit#login {{ padding: 10px 12px; font-size: 14px; border-radius: 10px; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {PRIMARY_SOFT}; selection-color: {TEXT}; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ padding: 8px 10px; border-radius: 9px; margin: 1px 0; }}
QListWidget::item:hover {{ background: {PRIMARY_SOFT}; }}
QListWidget::item:selected {{ background: {PRIMARY}; color: {on_primary}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {blend(BORDER, MUTED, 0.35)}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {PRIMARY}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {blend(BORDER, MUTED, 0.35)}; border-radius: 4px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: transparent; }}
QWidget#vsep {{ background: {BORDER}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ color: {MUTED}; }}
QProgressBar {{ background: {BORDER}; border: none; border-radius: 3px; max-height: 6px; }}
QProgressBar::chunk {{ background: {PRIMARY}; border-radius: 3px; }}
QMenuBar {{ background: {SIDEBAR}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {PRIMARY_SOFT}; border-radius: 6px; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: {PRIMARY_SOFT}; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
QMessageBox {{ background: {BG}; }}
"""


set_mode(False)
