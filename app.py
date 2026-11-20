"""DENT-AI masaüstü uygulaması:  python app.py [görüntü]"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QDialog

from dentai.records import load_settings
from dentai.ui import theme
from dentai.ui.login import LoginDialog
from dentai.ui.logo import app_icon
from dentai.ui.main_window import MainWindow, install_error_hook


def main() -> int:
    install_error_hook()
    if sys.platform == "win32":
        # Görev çubuğunda python ikonu yerine DENT-AI logosu görünsün
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DENT-AI.app")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())
    theme.set_mode(load_settings().get("theme") == "dark")
    first_image = Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).exists() else None
    while True:
        app.setStyleSheet(theme.qss())
        login = LoginDialog()
        if login.exec() != QDialog.DialogCode.Accepted:
            return 0
        win = MainWindow(user=login.username)
        win.showMaximized()           # tüm ekran: FDI çubuğu dahil her şey görünür
        if first_image is not None:
            win.load_image(first_image)
            first_image = None
        app.exec()
        if not win.logout_requested:      # pencere kapatıldı → çık; "Oturumu kapat" → girişe dön
            return 0


if __name__ == "__main__":
    sys.exit(main())
