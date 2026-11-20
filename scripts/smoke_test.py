"""Uçtan uca duman testi (CPU, birkaç dakika): sentetik veri -> hazırlama -> eğitim -> çıkarım -> arayüz.

    python scripts/smoke_test.py
Modellerin doğruluğunu DEĞİL, boru hattının çalıştığını doğrular.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# OneDrive eşzamanlama klasörleri kilitleyebildiği için geçici dizin kullanılır.
B = Path(tempfile.gettempdir()) / "dentai_smoke"


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def main() -> None:
    from dentai.age.train_age import train as train_age
    from dentai.data.prepare import build_dataset, print_report
    from dentai.imaging import imread
    from dentai.inference.draw import draw_analysis
    from dentai.inference.pipeline import DentalAnalyzer
    from dentai.train.train import train

    shutil.rmtree(B, ignore_errors=True)
    print("Çalışma dizini:", B)
    step("1) sentetik veri")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_dataset.py"),
                    "--out", str(B / "synth"), "--n", "30"], check=True)

    step("2) data.yaml + bölme")
    print_report(build_dataset(B / "synth" / "teeth", B / "ds_teeth", "teeth", keep_splits=True), B / "ds_teeth")
    print_report(build_dataset(B / "synth" / "findings", B / "ds_findings", "findings", clahe=True, keep_splits=True),
                 B / "ds_findings")

    models = B / "models"
    step("3) YOLOv8n eğitimi (2 epoch, CPU)")
    for task in ("teeth", "findings"):
        train(task, B / f"ds_{task}" / "data.yaml", epochs=2, imgsz=320, batch=4, patience=5,
              device="cpu", workers=0, cache=False, project=B / "runs", save_dir=models)

    step("4) yaş regresörü (2 epoch)")
    train_age(B / "synth" / "age", B / "synth" / "age" / "labels.csv", epochs=2, batch=8,
              device="cpu", workers=0, save_dir=models)

    step("5) çıkarım + çizim")
    an = DentalAnalyzer(models / "teeth_best.pt", models / "findings_best.pt", models / "age_best.pt",
                        device="cpu", raw_conf=0.001)
    img = imread(B / "synth" / "age" / "pano000.jpg")
    raw = an.infer(img)
    res = an.assemble(raw, 0.001)
    print(f"ham: diş={len(raw.teeth)} bulgu={len(raw.findings)} yaş={raw.age:.1f}  | eşik 0.001: {res.counts()}")
    assert raw.age is not None and 0 <= raw.age < 120
    out = draw_analysis(img, res)
    assert out.shape == img.shape
    print("OK: boru hattı çalıştı")


if __name__ == "__main__":
    main()
