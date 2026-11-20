import yaml

from dentai.data.prepare import build_class_map, build_dataset, parse_label_line, target_names
from dentai.fdi import FDI_NAMES
from dentai.inference.pipeline import DentalAnalyzer, assign_findings, dedupe_teeth
from dentai.inference.results import Detection, RawResult


def _roboflow_export(root, n_images=10):
    (root).mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["tooth-11", "Tooth 36", "junk"]}))
    for i in range(n_images):
        for k in range(2):  # artırılmış kopyalar
            split = "train" if i % 2 else "valid"
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
            stem = f"img{i}_jpg.rf.h{k}"
            import cv2, numpy as np
            cv2.imwrite(str(root / split / "images" / f"{stem}.jpg"), np.zeros((40, 80, 3), np.uint8))
            (root / split / "labels" / f"{stem}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n2 0.5 0.5 0.1 0.1\n"
                "1 0.1 0.1 0.2 0.1 0.2 0.3 0.1 0.3\n"   # poligon
                "1 1.5 0.5 0.1 0.1\n")                  # sınır dışı -> bozuk


def test_class_map_fdi_heuristic():
    m = build_class_map(["tooth-11", "Tooth 36", "junk"], target_names("teeth"), "teeth", None)
    assert m == {0: FDI_NAMES.index("11"), 1: FDI_NAMES.index("36")}


def test_parse_polygon_to_bbox():
    cls, xc, yc, w, h = parse_label_line("3 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4")
    assert (cls, round(xc, 2), round(yc, 2), round(w, 2), round(h, 2)) == (3, 0.2, 0.25, 0.2, 0.3)
    assert parse_label_line("0 2 0.5 0.1 0.1") is None


def test_build_dataset_no_leakage_and_yaml(tmp_path):
    src = tmp_path / "rf"
    _roboflow_export(src)
    stats = build_dataset(src, tmp_path / "out", "teeth", val_ratio=0.2)
    cfg = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
    assert cfg["nc"] == 32 and cfg["names"][FDI_NAMES.index("36")] == "36"
    assert (tmp_path / "out" / "images" / "val").exists()

    def bases(split):
        return {p.stem.split(".rf.")[0] for p in (tmp_path / "out" / "images" / split).iterdir()}

    assert bases("train").isdisjoint(bases("val"))          # aynı görüntünün kopyaları bölünmedi
    assert stats["images"]["val"] == 2 * len(bases("val"))
    assert stats["dropped_lines"] > 0 and stats["invalid_lines"] > 0


def test_fusion_assignment_and_dedupe():
    teeth = [Detection((0, 0, 100, 200), "36", 0.9), Detection((110, 0, 210, 200), "37", 0.8),
             Detection((0, 0, 100, 200), "36", 0.4)]
    teeth = dedupe_teeth(teeth)
    assert [t.name for t in teeth] == ["36", "37"] and teeth[0].conf == 0.9
    finds = [Detection((20, 50, 60, 90), "caries_early", 0.7, "caries"),
             Detection((90, 50, 130, 90), "filling", 0.6, "restoration"),   # iki dişe yayılır
             Detection((900, 900, 950, 950), "crown", 0.6, "restoration")]  # hiçbir dişte değil
    got = assign_findings(finds, teeth)
    assert got[0] == "36" and got[2] is None and got[1] in ("36", "37")


def test_convert_coco_then_build(tmp_path):
    import json, cv2, numpy as np
    from dentai.data.prepare import convert_coco

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    images, anns = [], []
    for i in range(6):
        cv2.imwrite(str(imgs / f"a{i}.jpg"), np.zeros((100, 200, 3), np.uint8))
        images.append({"id": i, "file_name": f"sub/a{i}.jpg", "width": 200, "height": 100})
        anns.append({"image_id": i, "category_id": 7, "bbox": [20, 10, 40, 20]})
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 7, "name": "Filling"}, {"id": 9, "name": "Caries"}]}
    (tmp_path / "c.json").write_text(json.dumps(coco))
    convert_coco(tmp_path / "c.json", imgs, tmp_path / "yolo")
    line = (tmp_path / "yolo" / "labels" / "a0.txt").read_text().split()
    assert line[0] == "0" and float(line[1]) == 0.2 and float(line[3]) == 0.2
    stats = build_dataset(tmp_path / "yolo", tmp_path / "ds", "findings", mapping={"caries": "caries"})
    assert sum(stats["instances"][s]["filling"] for s in ("train", "val", "test")) == 6


def test_keep_splits_and_test_split(tmp_path):
    import cv2, numpy as np
    src = tmp_path / "ds"
    (src / "data.yaml").parent.mkdir(parents=True)
    (src / "data.yaml").write_text(yaml.safe_dump({"names": ["Caries", "Wire", "Impacted tooth", "Root Canal Treatment"]}))
    for split, n in (("train", 6), ("valid", 3), ("test", 2)):
        for i in range(n):
            (src / split / "images").mkdir(parents=True, exist_ok=True)
            (src / split / "labels").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(src / split / "images" / f"{split}{i}.jpg"), np.zeros((20, 20, 3), np.uint8))
            (src / split / "labels" / f"{split}{i}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n2 0.4 0.4 0.1 0.1\n3 0.6 0.6 0.1 0.1\n")
    import pathlib
    m = yaml.safe_load(pathlib.Path("configs/map_findings.yaml").read_text(encoding="utf-8"))
    stats = build_dataset(src, tmp_path / "out", "findings", mapping=m, keep_splits=True)
    assert (stats["images"]["train"], stats["images"]["val"], stats["images"]["test"]) == (6, 3, 2)
    assert stats["dropped_lines"] == 11                          # 11 görüntüde "Wire" atıldı
    assert stats["instances"]["train"]["impacted"] == 6 and stats["instances"]["val"]["root_canal_treatment"] == 3
    cfg = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
    assert cfg["test"] == "images/test"


def test_ultralytics_layout_images_split(tmp_path):
    import cv2, numpy as np
    src = tmp_path / "YOLO"
    src.mkdir()
    (src / "data.yaml").write_text(yaml.safe_dump({"names": ["Caries", "Filling"]}))
    for split in ("train", "valid", "test"):
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        for i in range(2):
            cv2.imwrite(str(src / "images" / split / f"{split}{i}.jpg"), np.zeros((20, 20, 3), np.uint8))
            (src / "labels" / split / f"{split}{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n")
    stats = build_dataset(src, tmp_path / "out", "findings", keep_splits=True)
    assert (stats["images"]["train"], stats["images"]["val"], stats["images"]["test"]) == (2, 2, 2)
    assert stats["instances"]["train"]["caries"] == 2 and stats["instances"]["test"]["filling"] == 2


def _noise_set(root, names, n, rng, label, reuse=None):
    """Rastgele dokulu görüntüler (özetleri farklı). reuse: başka setten aynen kopyalanacak görüntüler."""
    import cv2, numpy as np
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    imgs = []
    for i in range(n):
        if reuse and i < len(reuse):   # başka setteki görüntünün boyutu/kontrastı değişmiş kopyası
            img = cv2.convertScaleAbs(cv2.resize(reuse[i], (320, 160)), alpha=1.2, beta=-10)
        else:                          # röntgen gibi yumuşak geçişli rastgele görüntü
            low = rng.integers(0, 255, (6, 12, 3), np.uint8)
            img = cv2.resize(low, (256, 128), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(root / "train" / "images" / f"x{i}.png"), img)
        (root / "train" / "labels" / f"x{i}.txt").write_text(label(i))
        imgs.append(img)
    return imgs


def test_70_15_15_stratified_and_cross_source_dedupe(tmp_path):
    import numpy as np
    from dentai.imaging import imread
    rng = np.random.default_rng(0)
    # A: 60 görüntü, her 5.'inde çürük (12 çürüklü görüntü)
    a = _noise_set(tmp_path / "A", ["Caries", "Filling"], 60, rng,
                   lambda i: ("0 0.5 0.5 0.1 0.1\n" if i % 5 == 0 else "") + "1 0.3 0.3 0.1 0.1\n")
    # B: farklı sınıf adlandırması; ilk 5 görüntüsü A'nın ilk 5'inin yeniden boyutlanmış kopyası
    _noise_set(tmp_path / "B", ["Root Canal Treatment", "caries"], 40, rng,
               lambda i: "0 0.5 0.5 0.1 0.1\n1 0.4 0.4 0.1 0.1\n", reuse=a[:5])
    stats = build_dataset([tmp_path / "A", tmp_path / "B"], tmp_path / "out", "findings")
    assert stats["duplicate_images"] == 5
    imgs = stats["images"]
    assert sum(imgs.values()) == 100
    assert abs(imgs["val"] - 15) <= 2 and abs(imgs["test"] - 15) <= 2   # gruplar (tekrarlar) bölünmez
    car = stats["instances"]
    assert car["test"]["caries"] > 0 and car["val"]["caries"] > 0      # seyrek sınıf her bölmede
    assert car["train"]["root_canal_treatment"] > 0                     # B'nin eşlemesi kendi data.yaml'ından

    from dentai.data.prepare import _dhash, _near_duplicate_pairs
    files = [(s, p) for s in ("train", "val", "test") for p in (tmp_path / "out" / "images" / s).iterdir()]
    pairs = list(_near_duplicate_pairs(np.stack([_dhash(p) for _, p in files])))
    assert len(pairs) == 5                                              # 5 kopya çifti ...
    assert all(files[i][0] == files[j][0] for i, j in pairs)            # ... hep aynı bölmede (sızıntı yok)
    cfg = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
    assert cfg["test"] == "images/test"


def test_convert_dentex_teeth_and_findings(tmp_path):
    import json, cv2, numpy as np, pytest
    from dentai.data.prepare import convert_dentex
    xr = tmp_path / "xrays"
    xr.mkdir()
    cv2.imwrite(str(xr / "train_1.png"), np.zeros((100, 200, 3), np.uint8))
    base = {"images": [{"id": 7, "file_name": "train_1.png", "width": 200, "height": 100}],
            "categories_1": [{"id": i, "name": str(i + 1)} for i in range(4)],
            "categories_2": [{"id": i, "name": str(i + 1)} for i in range(8)]}
    anns = [{"image_id": 7, "bbox": [20, 10, 40, 20], "category_id_1": 2, "category_id_2": 5},     # 36
            {"image_id": 7, "bbox": [190, 90, 30, 30], "category_id_1": 0, "category_id_2": 7}]    # 18, taşan kutu
    (tmp_path / "e.json").write_text(json.dumps({**base, "annotations": anns}))
    convert_dentex(tmp_path / "e.json", xr, tmp_path / "t", "teeth")
    rows = [r.split() for r in (tmp_path / "t" / "labels" / "train_1.txt").read_text().splitlines()]
    assert [FDI_NAMES[int(r[0])] for r in rows] == ["36", "18"]
    assert rows[0][1:] == ["0.200000", "0.200000", "0.200000", "0.200000"]
    assert float(rows[1][3]) == pytest.approx(0.05) and float(rows[1][1]) == pytest.approx(0.975)  # görüntüye kırpıldı

    cats3 = [{"id": 0, "name": "Impacted"}, {"id": 1, "name": "Caries"}, {"id": 3, "name": "Deep Caries"}]
    anns3 = [dict(anns[0], category_id_3=3), dict(anns[0], category_id_3=1)]
    (tmp_path / "d.json").write_text(json.dumps({**base, "categories_3": cats3, "annotations": anns3}))
    convert_dentex(tmp_path / "d.json", xr, tmp_path / "f", "findings")
    m = yaml.safe_load(pathlib_read("configs/map_findings.yaml"))
    from dentai.data.prepare import read_source_names, build_class_map, target_names
    cmap = build_class_map(read_source_names(tmp_path / "f"), target_names("findings"), "findings", m)
    assert target_names("findings")[cmap[2]] == "caries"          # Deep Caries -> caries
    assert target_names("findings")[cmap[1]] == "caries"
    with pytest.raises(ValueError):                               # hastalık katmanı olmayan json
        convert_dentex(tmp_path / "e.json", xr, tmp_path / "x", "findings")


def pathlib_read(p):
    import pathlib
    return pathlib.Path(p).read_text(encoding="utf-8")


def test_nested_data_yaml_found_and_missing_yaml_fails(tmp_path):
    import cv2, numpy as np, pytest
    src = tmp_path / "outer"
    inner = src / "inner"
    for split in ("train", "valid"):
        (inner / split / "images").mkdir(parents=True)
        (inner / split / "labels").mkdir(parents=True)
        cv2.imwrite(str(inner / split / "images" / "a.jpg"), np.zeros((20, 20, 3), np.uint8))
        (inner / split / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n")
    with pytest.raises(FileNotFoundError):                      # yaml yok -> sessiz varsayım yok
        build_dataset(src, tmp_path / "o1", "findings", keep_splits=True)
    (inner / "data.yaml").write_text(yaml.safe_dump({"names": ["Crown", "Filling"]}))   # id0=Crown, id1=Filling
    stats = build_dataset(src, tmp_path / "o2", "findings", keep_splits=True)
    assert stats["instances"]["train"]["crown"] == 1 and stats["instances"]["train"]["filling"] == 1


def test_assemble_threshold():
    raw = RawResult([Detection((0, 0, 10, 10), "11", 0.9)],
                    [Detection((0, 0, 5, 5), "filling", 0.2, "restoration"),
                     Detection((0, 0, 5, 5), "caries_early", 0.8, "caries")], 12.3, (100, 100))
    res = DentalAnalyzer.assemble(raw, 0.5)
    assert res.counts() == {"caries": 1, "restoration": 0, "impacted": 0}
    assert res.to_dict("x.png")["findings"][0]["tooth"] == "11"
