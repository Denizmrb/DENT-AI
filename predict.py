"""Komut satırından panoramik röntgen analizi.

    python predict.py --source xray.jpg
    python predict.py --source klasor/ --out outputs --imgsz 1024 --conf 0.3 --show

Çıktı: <out>/<ad>_dentai.png (işaretli görüntü) ve <out>/<ad>_dentai.json (rapor).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from dentai.config import DEFAULT_WEIGHTS
from dentai.imaging import IMAGE_EXTS, fit_to_screen, imread, imwrite
from dentai.inference.draw import draw_analysis
from dentai.inference.pipeline import DentalAnalyzer


def collect_sources(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [source]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="görüntü ya da klasör")
    ap.add_argument("--teeth", default=DEFAULT_WEIGHTS["teeth"])
    ap.add_argument("--findings", default=DEFAULT_WEIGHTS["findings"])
    ap.add_argument("--age", default=DEFAULT_WEIGHTS["age"])
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=None, help="boş: modelin eğitim boyutu")
    ap.add_argument("--device", default=None)
    ap.add_argument("--show", action="store_true", help="oran korunarak ekrana sığdırılmış pencerede göster")
    ap.add_argument("--screen", default="1600x900", help="--show için azami pencere boyutu GENxYÜK")
    a = ap.parse_args(argv)

    analyzer = DentalAnalyzer(a.teeth, a.findings, a.age, device=a.device, imgsz=a.imgsz)
    if not (analyzer.available["teeth"] or analyzer.available["findings"]):
        raise SystemExit("Hiç model bulunamadı. Önce eğitin ya da --teeth/--findings verin (bkz. README).")
    print("Modeller:", {k: ("var" if v else "yok") for k, v in analyzer.available.items()})

    max_w, max_h = (int(v) for v in a.screen.lower().split("x"))
    for path in collect_sources(a.source):
        img = imread(path)
        res = analyzer.analyze(img, a.conf)
        annotated = draw_analysis(img, res)
        imwrite(a.out / f"{path.stem}_dentai.png", annotated)
        (a.out / f"{path.stem}_dentai.json").write_text(
            json.dumps(res.to_dict(path.name), ensure_ascii=False, indent=2), encoding="utf-8")
        c = res.counts()
        age = "-" if res.age is None else f"{res.age:.1f}"
        print(f"{path.name}: dis={len(res.teeth)} curuk={c['caries']} restorasyon={c['restoration']} "
              f"gomulu={c['impacted']} yas={age} ({res.elapsed_ms:.0f} ms)")
        if a.show:
            cv2.imshow("DENT-AI", fit_to_screen(annotated, max_w, max_h))
            if cv2.waitKey(0) in (27, ord("q")):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
