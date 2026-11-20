"""Arayüzü ekransız (offscreen) başlatır, analiz çalıştırır, ekran görüntüsü kaydeder.

    python scripts/ui_screenshot.py build/synth/age/pano000.jpg build/ui.png [models_dir]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")  # offscreen platformu font bulamaz
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from dentai.ui import theme  # noqa: E402
from dentai.ui.main_window import MainWindow  # noqa: E402


def main() -> None:
    img, out = Path(sys.argv[1]), Path(sys.argv[2])
    mdir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    app = QApplication([])
    app.setStyleSheet(theme.QSS)
    weights = {k: mdir / f"{k}_best.pt" for k in ("teeth", "findings", "age")} if mdir else None
    win = MainWindow(weights)
    win.slider.setValue(10)
    win.show()
    win.load_image(img)
    win.run_analysis()
    loop = QEventLoop()
    win.worker.finished.connect(loop.quit) if win.worker else None
    QTimer.singleShot(120_000, loop.quit)
    loop.exec()
    app.processEvents()
    assert win.raw is not None, "analiz sonucu gelmedi"
    win.refresh()
    app.processEvents()
    win.grab().save(str(out))
    print("kaydedildi:", out, "| bulgu:", len(win.result.findings), "diş:", len(win.result.teeth))


if __name__ == "__main__":
    main()
