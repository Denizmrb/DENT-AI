"""Dental yaş regresörünü eğitir.

    python -m dentai.age.train_age --images data/age/images --labels data/age/labels.csv

labels.csv: `filename,age` başlıklı, yıl cinsinden yaş. Hasta bazlı gruplama gerekiyorsa
`patient` sütunu eklenir (aynı hastanın görüntüleri aynı bölmeye gider).
Çıktı: models/age_best.pt  (en düşük val MAE).
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from dentai.age.model import INPUT_HW, build_age_model, preprocess
from dentai.config import MODELS_DIR
from dentai.imaging import imread


def read_labels(csv_path: Path) -> list[tuple[str, float, str]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["filename"], float(r["age"]), r.get("patient") or r["filename"]))
    return rows


def split_rows(rows, val_ratio: float, seed: int):
    patients = sorted({r[2] for r in rows})
    random.Random(seed).shuffle(patients)
    n_val = min(max(1, round(len(patients) * val_ratio)), len(patients) - 1)
    val_p = set(patients[:n_val])
    return [r for r in rows if r[2] not in val_p], [r for r in rows if r[2] in val_p]


def _augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    ang, sc = rng.uniform(-5, 5), rng.uniform(0.92, 1.08)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += (rng.uniform(-0.03, 0.03) * w, rng.uniform(-0.03, 0.03) * h)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    alpha, beta = rng.uniform(0.85, 1.15), rng.uniform(-15, 15)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def train(images: Path, labels: Path, epochs=60, batch=16, lr=3e-4, patience=12, val_ratio=0.15,
          device=None, workers=2, save_dir=MODELS_DIR, seed=0) -> Path:
    import torch
    from torch.utils.data import DataLoader, Dataset

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_labels(labels)
    tr, va = split_rows(rows, val_ratio, seed)
    ages = np.array([r[1] for r in tr], dtype=np.float32)
    mean, std = float(ages.mean()), float(ages.std() or 1.0)

    class DS(Dataset):
        def __init__(self, items, aug):
            self.items, self.aug, self.rng = items, aug, random.Random(seed)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            name, age, _ = self.items[i]
            img = imread(images / name)
            if self.aug:
                img = _augment(img, self.rng)
            x = torch.from_numpy(preprocess(img))
            return x, torch.tensor((age - mean) / std, dtype=torch.float32)

    dl_tr = DataLoader(DS(tr, True), batch_size=batch, shuffle=True, num_workers=workers, drop_last=len(tr) > batch)
    dl_va = DataLoader(DS(va, False), batch_size=batch, num_workers=workers)

    model = build_age_model(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best, bad = float("inf"), 0
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "age_best.pt"
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0], enabled=device == "cuda"):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        model.eval()
        err = []
        with torch.no_grad():
            for x, y in dl_va:
                pred = model(x.to(device)).squeeze(1).cpu()
                err.append(((pred - y).abs() * std))
        mae = torch.cat(err).mean().item()
        print(f"epoch {ep:3d}/{epochs}  val MAE = {mae:.2f} yıl")
        if mae < best:
            best, bad = mae, 0
            torch.save({"state_dict": model.state_dict(), "age_mean": mean, "age_std": std,
                        "input_hw": INPUT_HW, "val_mae": mae}, out)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (patience={patience})")
                break
    print(f"En iyi val MAE {best:.2f} yıl -> {out}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-dir", default=str(MODELS_DIR))
    a = ap.parse_args(argv)
    train(a.images, a.labels, a.epochs, a.batch, a.lr, a.patience, device=a.device,
          workers=a.workers, save_dir=a.save_dir)


if __name__ == "__main__":
    main()
