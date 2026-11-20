"""YOLOv8n eğitimi (Colab T4 için ayarlı).

    python -m dentai.train.train --task teeth    --data datasets/teeth/data.yaml
    python -m dentai.train.train --task findings --data datasets/findings/data.yaml --imgsz 1024

Çıktı: <save-dir>/<task>_best.pt  (+ <task>_best.json: imgsz / CLAHE bilgisi, çıkarımda otomatik okunur).
Colab'da --save-dir olarak Google Drive yolu verilirse oturum kapansa da model kaybolmaz.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from dentai.config import MODELS_DIR, ROOT, load_augment
from dentai.data.prepare import META_NAME


def pick_device(requested: str | None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        return "0"
    print("UYARI: CUDA yok, CPU'da eğitim çok yavaş olur. Colab: Çalışma zamanı > T4 GPU seçin.")
    return "cpu"


def make_portable_yaml(data: Path) -> dict:
    """data.yaml içindeki `path`'i bu dosyanın bulunduğu klasöre çevirir (Windows'ta üretilip Colab'da kullanılabilsin)."""
    cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    here = str(data.parent.resolve())
    if cfg.get("path") != here:
        cfg["path"] = here
        data.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return cfg


def evaluate_on_test(weights: Path, data: Path, imgsz: int, device: str, out_json: Path) -> None:
    """Eğitimde hiç kullanılmayan test kümesinde sınıf bazlı skor yazdırır ve JSON'a kaydeder."""
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data), split="test", imgsz=imgsz, device=device, plots=False, verbose=False)
    per = {m.names[int(c)]: round(float(a), 4) for c, a in zip(m.box.ap_class_index, m.box.ap50)}
    summary = {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
               "AP50_per_class": per}
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nTEST kümesi:", json.dumps(summary, ensure_ascii=False, indent=2))


def train(
    task: str,
    data: str | Path,
    epochs: int = 200,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 30,
    model: str = "yolov8n.pt",
    device: str | None = None,
    workers: int = 2,
    cache: str = "disk",
    project: str | Path = ROOT / "runs",
    save_dir: str | Path = MODELS_DIR,
    seed: int = 0,
    extra: dict | None = None,
    resume: str | Path | None = None,
) -> Path:
    from ultralytics import YOLO

    data = Path(data)
    cfg = make_portable_yaml(data)
    meta_path = data.parent / META_NAME
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    args = dict(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,          # Early stopping: val mAP bu kadar epoch iyileşmezse durur
        device=pick_device(device),
        workers=workers,
        cache=cache,
        project=str(Path(project).resolve()),
        name=task,
        exist_ok=True,
        amp=True,                   # T4'te karma hassasiyet: ~2x hız, daha az VRAM
        cos_lr=True,
        seed=seed,
        plots=True,
        **load_augment(task),
        **(extra or {}),
    )
    if resume:  # Colab oturumu kesilirse: <project>/<task>/weights/last.pt'den devam
        yolo = YOLO(str(resume))
        yolo.train(resume=True)
    else:
        yolo = YOLO(model)
        yolo.train(**args)

    run_dir = Path(yolo.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        best = run_dir / "weights" / "last.pt"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{task}_best.pt"
    shutil.copy2(best, out)
    (save_dir / f"{task}_best.json").write_text(
        json.dumps({"task": task, "imgsz": imgsz, "clahe": bool(meta.get("clahe", False)),
                    "names": cfg.get("names")}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nKaydedildi: {out}")
    if "test" in cfg:
        evaluate_on_test(out, data, imgsz, args["device"], save_dir / f"{task}_test_metrics.json")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["teeth", "findings"], required=True)
    ap.add_argument("--data", required=True, help="prepare.py'nin ürettiği data.yaml")
    ap.add_argument("--epochs", type=int, default=200, help="üst sınır; Early Stopping genelde daha erken durdurur")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="-1 = otomatik")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--model", default="yolov8n.pt", help="ör. yolo11n.pt de denenebilir")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cache", default="disk", choices=["ram", "disk", "False"])
    ap.add_argument("--project", default=str(ROOT / "runs"))
    ap.add_argument("--save-dir", default=str(MODELS_DIR))
    ap.add_argument("--resume", default=None, help="yarım kalan eğitimin last.pt dosyası")
    a = ap.parse_args(argv)
    train(a.task, a.data, a.epochs, a.imgsz, a.batch, a.patience, a.model, a.device,
          a.workers, a.cache if a.cache != "False" else False, a.project, a.save_dir, resume=a.resume)


if __name__ == "__main__":
    main()
