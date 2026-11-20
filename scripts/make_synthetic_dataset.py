"""Sahte "panoramik" görüntülerden Roboflow-benzeri YOLOv8 dışa aktarımı üretir.

Yalnızca boru hattını (hazırlama, eğitim, çıkarım, arayüz) doğrulamak içindir; klinik değeri yoktur.

    python scripts/make_synthetic_dataset.py --out build/synth --n 24
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentai.fdi import LOWER_ROW, UPPER_ROW  # noqa: E402
from dentai.imaging import imwrite  # noqa: E402

FIND = ["caries", "filling", "crown", "implant", "root_canal_treatment", "impacted"]


def make_image(rng: random.Random, w=1200, h=600):
    img = np.full((h, w), 30, np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 2 - 40, h // 2 - 30), 0, 0, 360, 55, -1)
    teeth, finds = [], []
    for row, (names, cy) in enumerate(((UPPER_ROW, int(h * 0.38)), (LOWER_ROW, int(h * 0.64)))):
        step = (w - 200) / 16
        for i, fdi in enumerate(names):
            cx = int(100 + step * (i + 0.5))
            tw, th = int(step * 0.8), rng.randint(70, 95)
            x1, y1, x2, y2 = cx - tw // 2, cy - th // 2, cx + tw // 2, cy + th // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), rng.randint(170, 200), -1)
            teeth.append((fdi, (x1, y1, x2, y2)))
            if rng.random() < 0.22:
                kind = rng.choice(FIND)
                bw, bh = tw // 2, th // 3
                bx, by = cx - bw // 2, cy - bh // 2
                shade = 90 if kind == "caries" else 240
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), shade, -1)
                finds.append((kind, (bx, by, bx + bw, by + bh)))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, 4, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), teeth, finds


def yolo_line(cls, box, w, h):
    x1, y1, x2, y2 = box
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # Roboflow'un bölme + ".rf." artırılmış kopya düzenini taklit eder (sızıntı testi için).
    split_of = lambda i: {0: "valid", 1: "test"}.get(i % 5, "train")
    for task in ("teeth", "findings"):
        for s in ("train", "valid", "test"):
            (a.out / task / s / "images").mkdir(parents=True, exist_ok=True)
            (a.out / task / s / "labels").mkdir(parents=True, exist_ok=True)
    (a.out / "age").mkdir(parents=True, exist_ok=True)

    with open(a.out / "age" / "labels.csv", "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["filename", "age"])
        for i in range(a.n):
            img, teeth, finds = make_image(rng)
            h, w = img.shape[:2]
            s = split_of(i)
            for copy in range(2):  # her görüntü için 2 "artırılmış" kopya
                stem = f"pano{i:03d}_jpg.rf.{copy}abc"
                variant = img if copy == 0 else cv2.convertScaleAbs(img, alpha=1.1, beta=5)
                for task, rows in (("teeth", [(fdi, b) for fdi, b in teeth]), ("findings", finds)):
                    imwrite(a.out / task / s / "images" / f"{stem}.jpg", variant)
                    names = [n for n, _ in teeth] if task == "teeth" else FIND
                    # Roboflow sınıf adları: teeth için "tooth-36" gibi, findings için doğrudan ad
                    lines = [yolo_line(names.index(n) if task == "findings" else sorted(set(names)).index(n), b, w, h)
                             for n, b in rows]
                    (a.out / task / s / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            imwrite(a.out / "age" / f"pano{i:03d}.jpg", img)
            wr.writerow([f"pano{i:03d}.jpg", 8 + 40 * (len(finds) / 10) + rng.uniform(-2, 2)])

    import yaml
    teeth_names = sorted({n for n, _ in teeth})
    (a.out / "teeth" / "data.yaml").write_text(yaml.safe_dump({"names": [f"tooth-{n}" for n in teeth_names]}))
    (a.out / "findings" / "data.yaml").write_text(yaml.safe_dump({"names": FIND}))
    print("Sentetik veri:", a.out)


if __name__ == "__main__":
    main()
