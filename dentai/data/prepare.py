"""Roboflow (YOLOv8 formatı) dışa aktarımından eğitime hazır veri seti üretir.

    python -m dentai.data.prepare download --workspace W --project P --version 1 --out raw/teeth
    python -m dentai.data.prepare build --src raw/teeth --out datasets/teeth --task teeth
    python -m dentai.data.prepare build --src raw/dentex --out datasets/findings --task findings --map configs/map.yaml
    python -m dentai.data.prepare build --src raw/a raw/b raw/c --out datasets/findings --task findings --map ...

Yaptıkları:
  * Bir ya da birden çok veri setinin train/valid/test bölmelerini birleştirir,
    %70 / %15 / %15 (train / val / test) yeniden böler; seyrek sınıflar her bölmeye orantılı düşer.
    Roboflow'un `.rf.` artırılmış kopyaları ve farklı setlerde tekrar eden aynı röntgen
    AYNI bölmeye gider (sızıntı olmaz).
  * Kaynak sınıf adlarını hedef sınıflara (FDI 32 sınıf ya da configs/classes.yaml) eşler.
  * Poligon (segmentasyon) etiketlerini sınırlayıcı kutuya çevirir, bozuk satırları atar.
  * Hedef sınıf listesinden data.yaml'ı dinamik üretir; istenirse CLAHE uygular.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from dentai.config import load_classes
from dentai.fdi import FDI_NAMES
from dentai.imaging import IMAGE_EXTS, enhance_contrast, imread, imwrite

META_NAME = "dentai_meta.json"
_FDI_RE = re.compile(r"(?<!\d)([1-4][1-8])(?!\d)")


def target_names(task: str) -> list[str]:
    if task == "teeth":
        return list(FDI_NAMES)
    if task == "findings":
        return load_classes().finding_names
    raise ValueError("task: teeth | findings")


def _norm(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def read_source_names(src: Path) -> list[str] | None:
    yml = src / "data.yaml"
    if not yml.exists():  # zip bir klasörün içine açılmış olabilir: en yakın data.yaml'ı ara
        found = sorted(src.rglob("data.yaml"), key=lambda p: len(p.parts))
        if not found:
            return None
        yml = found[0]
    names = yaml.safe_load(yml.read_text(encoding="utf-8")).get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names) if names else None


def build_class_map(
    src_names: list[str] | None, targets: list[str], task: str, mapping: dict | None
) -> dict[int, int]:
    """kaynak sınıf id -> hedef sınıf id. Eşlenemeyenler sözlükte yer almaz (atılır)."""
    if src_names is None:  # data.yaml yok: id'ler zaten hedef uzayında varsayılır
        return {i: i for i in range(len(targets))}
    tindex = {_norm(n): i for i, n in enumerate(targets)}
    mapping = {_norm(k): v for k, v in (mapping or {}).items()}
    out: dict[int, int] = {}
    for sid, sname in enumerate(src_names):
        key = _norm(sname)
        if key in mapping:
            tgt = mapping[key]
            if tgt is None:
                continue
            tgt = _norm(tgt)
        elif key in tindex:
            tgt = key
        elif task == "teeth" and (m := _FDI_RE.search(str(sname))):
            tgt = m.group(1)
        else:
            continue
        if tgt not in tindex:
            raise ValueError(f"Eşleme hedefi bilinmiyor: '{tgt}' (kaynak '{sname}')")
        out[sid] = tindex[tgt]
    return out


def parse_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    """YOLO kutu (5 değer) ya da poligon (>=7 değer) satırı -> (cls, xc, yc, w, h)."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        vals = [float(p) for p in parts[1:]]
    except ValueError:
        return None
    if len(vals) == 4:
        xc, yc, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    else:
        return None
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return None
    return cls, xc, yc, w, h


def _collect_pairs(src: Path) -> list[tuple[Path, Path]]:
    """İki yaygın düzeni tanır: <split>/images/x.jpg (Roboflow) ve images/<split>/x.jpg (Ultralytics)."""
    pairs = []
    for img in sorted(src.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        if img.parent.name == "images":
            lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        elif img.parent.parent.name == "images":
            lbl = img.parent.parent.parent / "labels" / img.parent.name / (img.stem + ".txt")
        else:
            continue
        pairs.append((img, lbl))
    return pairs


def _group_key(img: Path) -> str:
    return img.stem.split(".rf.")[0]


_HASH = 16                      # 16x16 = 256 bit
_DUP_MAX_BITS = 12              # bu kadar ya da daha az bit farkı -> aynı röntgen sayılır (~%5)
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def _dhash(img: Path) -> np.ndarray:
    """256 bit fark özeti (32 bayt). Yeniden boyutlanmış / sıkıştırılmış / kontrastı değişmiş
    aynı röntgen birkaç bit farkla aynı özeti verir; farklı hastalar çok daha uzak düşer."""
    g = cv2.cvtColor(imread(img), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(cv2.GaussianBlur(g, (0, 0), 1.5), (_HASH + 1, _HASH), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def _near_duplicate_pairs(hashes: np.ndarray, chunk: int = 128):
    """Hamming uzaklığı <= _DUP_MAX_BITS olan (i, j>i) çiftleri. Düz/boş görüntüler atlanır."""
    bits = _POPCOUNT[hashes].sum(1)
    ok = (bits > 8) & (bits < hashes.shape[1] * 8 - 8)
    for s in range(0, len(hashes), chunk):
        d = _POPCOUNT[hashes[s:s + chunk, None, :] ^ hashes[None, :, :]].sum(-1)
        for a, b in zip(*np.nonzero(d <= _DUP_MAX_BITS)):
            i = s + a
            if b > i and ok[i] and ok[b]:
                yield i, int(b)


_SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}


def _split_name(img: Path) -> str | None:
    folder = img.parent.parent if img.parent.name == "images" else img.parent
    return _SPLIT_ALIASES.get(folder.name.lower())


def _label_classes(lbl: Path, cmap: dict[int, int]) -> Counter:
    c: Counter = Counter()
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = parse_label_line(line)
            if p is not None and p[0] in cmap:
                c[cmap[p[0]]] += 1
    return c


def _group_pairs(pairs, stats: dict) -> dict[str, list]:
    """Aynı görüntünün kopyalarını tek grupta toplar (gruplar bölmeler arasında paylaştırılmaz).

    1) Roboflow'un `.rf.` artırılmış kopyaları (aynı kaynak içinde).
    2) Görüntü özeti aynı olanlar: farklı veri setlerinde tekrar yüklenmiş aynı röntgen.
    """
    parent: dict[str, str] = {}

    def find(k):
        while parent.setdefault(k, k) != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    keyed = [(f"{sidx}:{_group_key(img)}", img, lbl, sidx) for img, lbl, sidx in pairs]
    for k, *_ in keyed:
        find(k)
    print(f"Tekrar eden görüntüler aranıyor ({len(keyed)} görüntü)...", flush=True)
    hashes = np.stack([_dhash(img) for _, img, _, _ in keyed])
    for i, j in _near_duplicate_pairs(hashes):
        ri, rj = find(keyed[i][0]), find(keyed[j][0])
        if ri != rj:
            parent[rj] = ri
            stats["duplicate_images"] += 1
    groups: dict[str, list] = defaultdict(list)
    for key, img, lbl, sidx in keyed:
        groups[find(key)].append((img, lbl, sidx))
    return groups


def _assign_splits(pairs, cmaps, keep_splits: bool, val_ratio: float, test_ratio: float, seed: int,
                   stats: dict, tries: int = 300):
    """[(split, img, lbl, sidx)] üretir. keep_splits: kaynağın train/valid/test ayrımını olduğu gibi korur.

    Aksi halde gruplar train/val/test'e oranlarla dağıtılır. Seyrek sınıfların (çürük, kanal tedavisi)
    her bölmeye orantılı düşmesi için `tries` farklı karıştırma denenir, sınıf oranları hedefe
    en yakın olan seçilir (seed ile tekrarlanabilir).
    """
    if keep_splits:
        items = [(_split_name(i), i, l, s) for i, l, s in pairs]
        have = {it[0] for it in items}
        if None in have or not {"train", "val"} <= have:
            raise ValueError("--keep-splits için <src>/train|valid|test/images düzeni gerekir")
        return items
    if not (0 < val_ratio < 1 and 0 <= test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("Oranlar: 0 < val, 0 <= test, val + test < 1 olmalı")
    groups = _group_pairs(pairs, stats)
    keys = sorted(groups)
    n = len(keys)
    n_test = round(n * test_ratio)
    n_val = max(1, round(n * val_ratio))
    if n_val + n_test >= n:
        raise ValueError(f"{n} farklı görüntü bu oranlarla bölünemez (train boş kalır)")

    gcls = {k: sum((_label_classes(l, cmaps[s]) for _, l, s in groups[k]), Counter()) for k in keys}
    total = sum(gcls.values(), Counter())
    want = {"val": val_ratio, "test": test_ratio}

    def score(order):
        got = {"test": Counter(), "val": Counter()}
        for k in order[:n_test]:
            got["test"] += gcls[k]
        for k in order[n_test:n_test + n_val]:
            got["val"] += gcls[k]
        return sum(abs(got[s][c] / t - want[s]) for c, t in total.items() for s in want)

    rng = random.Random(seed)
    best, best_score = None, float("inf")
    for _ in range(max(1, tries) if total else 1):
        order = keys[:]
        rng.shuffle(order)
        sc = score(order)
        if sc < best_score:
            best, best_score = order, sc
    split_of = {k: "test" if i < n_test else "val" if i < n_test + n_val else "train" for i, k in enumerate(best)}
    return [(split_of[k], img, lbl, s) for k in keys for img, lbl, s in groups[k]]


def build_dataset(
    src: str | Path | list,
    out: str | Path,
    task: str,
    mapping: dict | None = None,
    val_ratio: float = 0.15,
    seed: int = 42,
    clahe: bool = False,
    keep_splits: bool = False,
    assume_target_ids: bool = False,
    test_ratio: float = 0.15,
) -> dict:
    """src tek klasör ya da klasör listesi olabilir: birden çok veri seti tek veri setinde birleştirilir.

    `mapping` tüm kaynaklara uygulanır (her kaynağın kendi data.yaml'ındaki sınıf adlarıyla).
    """
    srcs = [Path(s) for s in (src if isinstance(src, (list, tuple)) else [src])]
    out = Path(out)
    targets = target_names(task)
    cmaps, pairs = [], []
    for sidx, s in enumerate(srcs):
        src_names = read_source_names(s)
        if src_names is None and not assume_target_ids:
            # Sessizce "id'ler zaten hedef sırasında" varsaymak sınıfları yanlış eşler.
            raise FileNotFoundError(
                f"{s} altında data.yaml bulunamadı; sınıf adları bilinmeden eşleme yapılamaz. "
                "id'lerin zaten hedef sırasında olduğundan eminseniz --assume-target-ids verin.")
        cmaps.append(build_class_map(src_names, targets, task, mapping))
        found = _collect_pairs(s)
        if not found:
            raise FileNotFoundError(f"{s} altında 'images/' klasörlü görüntü bulunamadı")
        pairs += [(i, l, sidx) for i, l in found]

    stats = {"images": Counter(), "instances": defaultdict(Counter), "dropped_lines": 0,
             "invalid_lines": 0, "no_label_images": 0, "duplicate_images": 0}
    items = _assign_splits(pairs, cmaps, keep_splits, val_ratio, test_ratio, seed, stats)
    splits = sorted({it[0] for it in items})

    if out.exists():
        shutil.rmtree(out)
    for s in splits:
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)

    used: set[tuple[str, str]] = set()

    for split, img, lbl, sidx in items:
        cmap = cmaps[sidx]
        name = img.name
        n = 0
        while (split, name) in used:
            n += 1
            name = f"{img.stem}_{n}{img.suffix}"
        used.add((split, name))

        rows = []
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                parsed = parse_label_line(line)
                if parsed is None:
                    stats["invalid_lines"] += 1
                    continue
                cls, *box = parsed
                if cls not in cmap:
                    stats["dropped_lines"] += 1
                    continue
                rows.append(f"{cmap[cls]} " + " ".join(f"{v:.6f}" for v in box))
                stats["instances"][split][targets[cmap[cls]]] += 1
        else:
            stats["no_label_images"] += 1

        dst = out / "images" / split / name
        if clahe:
            imwrite(dst, enhance_contrast(imread(img)))
        else:
            shutil.copy2(img, dst)
        (out / "labels" / split / (Path(name).stem + ".txt")).write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )
        stats["images"][split] += 1

    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        **({"test": "images/test"} if "test" in splits else {}),
        "nc": len(targets),
        "names": {i: n for i, n in enumerate(targets)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / META_NAME).write_text(
        json.dumps({"task": task, "clahe": clahe, "seed": seed, "val_ratio": val_ratio,
                    "test_ratio": test_ratio, "keep_splits": keep_splits,
                    "sources": [str(s) for s in srcs]}, ensure_ascii=False), encoding="utf-8"
    )

    empty = [n for n in targets if stats["instances"]["train"][n] == 0]
    stats["classes_without_train_instances"] = empty
    return stats


def print_report(stats: dict, out: Path) -> None:
    print(f"\ndata.yaml -> {out / 'data.yaml'}")
    print(f"Görüntü: train={stats['images']['train']}  val={stats['images']['val']}  test={stats['images']['test']}"
          f"  | etiketsiz={stats['no_label_images']}  atılan satır={stats['dropped_lines']}"
          f"  bozuk satır={stats['invalid_lines']}"
          + (f"  | kaynaklar arası tekrar={stats['duplicate_images']} (aynı bölmede tutuldu)"
             if stats.get("duplicate_images") else ""))
    names = sorted({n for c in stats["instances"].values() for n in c})
    for n in names:
        print(f"  {n:<24} train={stats['instances']['train'][n]:>6}  val={stats['instances']['val'][n]:>6}  test={stats['instances']['test'][n]:>6}")
    if stats["classes_without_train_instances"]:
        shown = ", ".join(stats["classes_without_train_instances"][:12])
        print(f"UYARI: train'de hiç örneği olmayan sınıflar: {shown}")


def convert_coco(json_path: str | Path, images_dir: str | Path, out: str | Path) -> Path:
    """Kaggle'da sık görülen COCO json'unu Roboflow-benzeri YOLO düzenine çevirir (images/, labels/, data.yaml).

    Çıktı build_dataset'e --src olarak verilir. Sınıf adları json'daki kategorilerden gelir.
    """
    coco = json.loads(Path(json_path).read_text(encoding="utf-8"))
    images_dir, out = Path(images_dir), Path(out)
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_index = {c["id"]: i for i, c in enumerate(cats)}
    boxes: dict[int, list[str]] = defaultdict(list)
    meta = {im["id"]: im for im in coco["images"]}
    for a in coco["annotations"]:
        im = meta.get(a["image_id"])
        if im is None or a.get("category_id") not in cat_index:
            continue
        x, y, w, h = a["bbox"]
        W, H = im["width"], im["height"]
        xc, yc, nw, nh = (x + w / 2) / W, (y + h / 2) / H, w / W, h / H
        boxes[im["id"]].append(f"{cat_index[a['category_id']]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    missing = 0
    for im in meta.values():
        src = images_dir / Path(im["file_name"]).name
        if not src.exists():
            missing += 1
            continue
        shutil.copy2(src, out / "images" / src.name)
        (out / "labels" / (src.stem + ".txt")).write_text("\n".join(boxes[im["id"]]) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        yaml.safe_dump({"names": [c["name"] for c in cats]}, allow_unicode=True), encoding="utf-8")
    print(f"{len(meta) - missing} görüntü dönüştürüldü" + (f", {missing} dosya bulunamadı" if missing else ""))
    print("Sınıflar:", [c["name"] for c in cats])
    return out


def convert_dentex(json_path: str | Path, images_dir: str | Path, out: str | Path, task: str) -> Path:
    """DENTEX (MICCAI 2023) COCO json'unu YOLO düzenine çevirir (images/, labels/, data.yaml).

    DENTEX tek `category_id` yerine üç katman kullanır: category_id_1 = çeyrek (1-4),
    category_id_2 = diş no (1-8), category_id_3 = hastalık (yalnızca disease setinde).
      teeth    -> sınıf = FDI (çeyrek*10 + diş no). quadrant_enumeration setinde BÜTÜN dişler etiketli;
                  disease setinde yalnızca hastalıklı dişler var, diş modeli için o set kullanılmamalı.
      findings -> sınıf = hastalık adı (Impacted, Caries, Periapical Lesion, Deep Caries).
                  Not: DENTEX kutuları lezyonu değil hastalıklı DİŞİN TAMAMINI kapsar.
    Çıktı build_dataset'e --src olarak verilir. Görüntüler aynı diskteyse kopyalanmaz, sabit bağlantı kurulur.
    """
    from dentai.fdi import FDI_INDEX

    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    images_dir, out = Path(images_dir), Path(out)
    quad = {c["id"]: int(c["name"]) for c in d["categories_1"]}
    num = {c["id"]: int(c["name"]) for c in d["categories_2"]}
    if task == "teeth":
        names = list(FDI_NAMES)

        def cls_of(a):
            return FDI_INDEX.get(f"{quad[a['category_id_1']]}{num[a['category_id_2']]}")
    elif task == "findings":
        if "categories_3" not in d:
            raise ValueError(f"{json_path} hastalık etiketi içermiyor (categories_3 yok)")
        cats = sorted(d["categories_3"], key=lambda c: c["id"])
        names = [c["name"] for c in cats]
        index = {c["id"]: i for i, c in enumerate(cats)}

        def cls_of(a):
            return index.get(a.get("category_id_3"))
    else:
        raise ValueError("task: teeth | findings")

    meta = {im["id"]: im for im in d["images"]}
    rows: dict[int, list[str]] = defaultdict(list)
    skipped = 0
    for a in d["annotations"]:
        im, c = meta.get(a["image_id"]), cls_of(a)
        if im is None or c is None:
            skipped += 1
            continue
        W, H = im["width"], im["height"]
        x, y, w, h = a["bbox"]
        x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(float(W), x + w), min(float(H), y + h)
        if x1 <= x0 or y1 <= y0:
            skipped += 1
            continue
        rows[im["id"]].append(f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                              f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")

    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    missing = 0
    for im in meta.values():
        src = images_dir / Path(im["file_name"]).name
        if not src.exists():
            missing += 1
            continue
        dst = out / "images" / src.name
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        (out / "labels" / (src.stem + ".txt")).write_text(
            "\n".join(rows[im["id"]]) + ("\n" if rows[im["id"]] else ""), encoding="utf-8")
    (out / "data.yaml").write_text(yaml.safe_dump({"names": names}, allow_unicode=True), encoding="utf-8")
    counts = Counter(int(r.split()[0]) for rs in rows.values() for r in rs)
    print(f"{len(meta) - missing} görüntü, {sum(counts.values())} kutu dönüştürüldü"
          + (f", {missing} görüntü bulunamadı" if missing else "") + (f", {skipped} kutu atlandı" if skipped else ""))
    if task == "findings":
        print("Sınıflar:", {names[k]: v for k, v in sorted(counts.items())})
    else:
        print(f"Diş sınıfı sayısı: {len(counts)} / 32  (görüntü başına ort. {sum(counts.values()) / max(1, len(rows)):.1f} diş)")
    return out


def download_roboflow(workspace: str, project: str, version: int, out: str | Path, api_key: str | None = None) -> Path:
    from roboflow import Roboflow  # opsiyonel bağımlılık

    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("API anahtarı yok: --api-key ya da ROBOFLOW_API_KEY ortam değişkeni")
    ds = Roboflow(api_key=key).workspace(workspace).project(project).version(version).download("yolov8", location=str(out))
    return Path(ds.location)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Roboflow Universe'ten YOLOv8 formatında indir")
    d.add_argument("--workspace", required=True)
    d.add_argument("--project", required=True)
    d.add_argument("--version", type=int, required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--api-key")

    c = sub.add_parser("coco", help="COCO json -> YOLO düzeni (sonra 'build' ile kullanın)")
    c.add_argument("--json", required=True)
    c.add_argument("--images", required=True)
    c.add_argument("--out", required=True)

    x = sub.add_parser("dentex", help="DENTEX json -> YOLO düzeni (sonra 'build' ile kullanın)")
    x.add_argument("--json", required=True)
    x.add_argument("--images", required=True, help="xrays klasörü")
    x.add_argument("--out", required=True)
    x.add_argument("--task", choices=["teeth", "findings"], required=True)

    b = sub.add_parser("build", help="data.yaml + %70/%15/%15 (train/val/test) bölünmüş veri seti üret")
    b.add_argument("--src", required=True, nargs="+",
                   help="bir ya da daha çok veri seti klasörü (birden çoksa birleştirilir)")
    b.add_argument("--out", required=True)
    b.add_argument("--task", choices=["teeth", "findings"], required=True)
    b.add_argument("--map", help="kaynak_sınıf: hedef_sınıf (ya da null) içeren YAML")
    b.add_argument("--val-ratio", type=float, default=0.15)
    b.add_argument("--test-ratio", type=float, default=0.15)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--keep-splits", action="store_true",
                   help="kaynağın train/valid/test ayrımını koru (yeniden bölme yok; hazır bölünmüş veri için)")
    b.add_argument("--assume-target-ids", action="store_true",
                   help="data.yaml yoksa kaynak id'lerini hedef sırasıyla aynı kabul et (riskli)")
    b.add_argument("--clahe", action="store_true", help="görüntülere CLAHE uygula (çıkarımda da aynısı kullanılır)")

    a = ap.parse_args(argv)
    if a.cmd == "download":
        print("İndirildi:", download_roboflow(a.workspace, a.project, a.version, a.out, a.api_key))
        return
    if a.cmd == "coco":
        convert_coco(a.json, a.images, a.out)
        return
    if a.cmd == "dentex":
        convert_dentex(a.json, a.images, a.out, a.task)
        return
    mapping = yaml.safe_load(Path(a.map).read_text(encoding="utf-8")) if a.map else None
    stats = build_dataset(a.src, a.out, a.task, mapping, val_ratio=a.val_ratio, seed=a.seed, clahe=a.clahe,
                          keep_splits=a.keep_splits, assume_target_ids=a.assume_target_ids,
                          test_ratio=a.test_ratio)
    print_report(stats, Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
