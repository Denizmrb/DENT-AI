import numpy as np

from dentai.auth import check_login, password_strength
from dentai.inference.draw import draw_analysis
from dentai.inference.results import AnalysisResult, Detection
from dentai.records import GENERAL, NotesStore, PatientStore


def test_login():
    assert check_login("deniz123456", "1548Deniz?")
    assert check_login(" Deniz123456 ", "1548Deniz?")    # kullanıcı adı büyük/küçük harf duyarsız
    assert not check_login("deniz123456", "1548deniz?")  # şifre duyarlı
    assert not check_login("deniz123456", "1548Deniz")
    assert not check_login("deniz", "1548Deniz")         # eski bilgiler artık geçmez
    assert not check_login("baska", "1548Deniz?")
    assert not check_login("", "")


def test_password_strength_order():
    s = [password_strength(p) for p in ("", "abc", "1548Deniz", "Str0ng!Parola2026")]
    assert s[0] == 0 and s[1] == 1 and s[1] < s[2] < s[3] == 4


def test_patients_and_notes_persist(tmp_path):
    ps = PatientStore(tmp_path / "h.json")
    a = ps.add("  Ayşe   Yılmaz ", "1990", tmp_path / "a.png")
    b = ps.add("Mehmet")
    assert a.name == "Ayşe Yılmaz"
    assert ps.attach_xray(b.id, tmp_path / "b.png")
    assert not ps.attach_xray(b.id, tmp_path / "b.png")  # aynı röntgen iki kez eklenmez
    ps2 = PatientStore(tmp_path / "h.json")
    assert [p.name for p in ps2.patients] == ["Ayşe Yılmaz", "Mehmet"]
    assert ps2.get(b.id).latest_xray.name == "b.png"
    ps2.remove(a.id)
    assert len(PatientStore(tmp_path / "h.json").patients) == 1

    ns = NotesStore(tmp_path / "n.json")
    ns.set(tmp_path / "a.png", GENERAL, "genel not")
    ns.set(tmp_path / "a.png", "36", "derin çürük")
    ns2 = NotesStore(tmp_path / "n.json")
    assert ns2.get(tmp_path / "a.png", "36") == "derin çürük"
    assert ns2.noted_teeth(tmp_path / "a.png") == {"36"}
    ns2.set(tmp_path / "a.png", "36", "   ")              # boş not silinir
    assert NotesStore(tmp_path / "n.json").noted_teeth(tmp_path / "a.png") == set()


def test_draw_layers_per_class_and_label_conf_independent():
    img = np.zeros((300, 600, 3), np.uint8)
    r = AnalysisResult(teeth=[], findings=[Detection((50, 120, 120, 200), "filling", 0.9, "restoration"),
                                           Detection((300, 120, 380, 200), "crown", 0.8, "restoration")],
                       tooth_of=["36", "46"], age=None, size=(600, 300))
    only_filling = draw_analysis(img, r, {"filling"}, show_labels=False, show_conf=False)
    assert only_filling[120:200, 50:120].any() and not only_filling[120:200, 300:380].any()
    plain = draw_analysis(img, r, {"filling"}, show_labels=False, show_conf=False)
    conf_only = draw_analysis(img, r, {"filling"}, show_labels=False, show_conf=True)
    both = draw_analysis(img, r, {"filling"}, show_labels=True, show_conf=True)
    above = slice(80, 119)
    assert not plain[above].any() and conf_only[above].any()
    assert both[above].astype(bool).sum() > conf_only[above].astype(bool).sum()   # etiket yazısı daha uzun
