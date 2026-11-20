# DENT-AI — Mimari ve Strateji

## Karar: tek model değil, ardışık küçük uzmanlar (pipeline)

| Görev | Model | Neden |
|---|---|---|
| Diş tespiti + **FDI numaralandırma** | YOLOv8n, 32 sınıf (`teeth`) | Diş kutuları büyük ve düzenli; numara, çenedeki konum bağlamından çıkar. |
| Çürük, dolgu, kuron, implant, kanal tedavisi, **gömülü diş** | YOLOv8n, 6 sınıf (`findings`) | Küçük, nadir, düşük kontrastlı nesneler; dengesiz sınıflar. Ayrı model, ayrı çözünürlük/augmentation demek. |
| **Dental yaş** | ResNet18 regresyon (`age`) | YOLO'nun regresyon başlığı yok. Yaş, tek bir kutuya değil tüm gelişim görünümüne bağlı. |
| Birleştirme | Kural tabanlı (`pipeline.py`) | Her bulgu, kutusunun en çok içinde kaldığı dişe (IoA ≥ 0.25) atanır → "36 numaralı dişte ileri evre çürük". |

### Neden tek modelde 5 görev değil?
1. **Çelişen augmentation ihtiyacı.** Numaralandırma "çenenin neresindeyim" bağlamını kullanır: yatay çevirme 18↔28 etiketlerini bozar, mozaik bağlamı kırar → `teeth` için `fliplr=0, mosaic=0`. Çürük/dolgu tespiti ise mozaik ve çevirmeden çok fayda görür → `findings` için `mosaic=1, fliplr=0.5`. Tek modelde bu ikisi birlikte sağlanamaz.
2. **Veri uyuşmazlığı.** Hiçbir açık veri setinde beşi birden etiketli değil. Ayrı modellerle her veri seti kendi görevi için kullanılır; birleşik etiket üretmeye gerek kalmaz.
3. **Ölçek farkı.** Dişler büyük, erken çürük birkaç piksel. Bulgu modeli ileride 1024 px (ya da dilimli çıkarım) ile eğitilebilir, diş modeli 640'ta kalır.

### Neden "kırpılmış diş üzerinde çürük ara" (iki aşamalı kırpma) değil?
Fikir cazip (daha yüksek efektif çözünürlük), ama diş modeli hatasını (kaçırılan diş = kaçırılan çürük) doğrudan bulgu modeline taşır ve çürüklerin komşu dişlerle bağlamını keser. Bunun yerine bulgu modeli **tüm görüntüde** çalışır, diş modeli yalnızca **etiketleme** için kullanılır; böylece bir model çökerse diğeri çalışmaya devam eder (UI model eksikse ilgili çıktıyı gizler). Küçük çürükler yetersiz kalırsa sıradaki adım: `imgsz=1024` ya da SAHI tarzı dilimli çıkarım.

### Dental yaş için not
Bu bir **araştırma özelliğidir**. Doğru yöntem (Demirjian/Willems) her dişin kalsifikasyon evresini sınıflamaktır; bunun için evre etiketli veri gerekir. Şimdilik tüm görüntüden doğrudan regresyon yapılır (çocuk panoramik veri setleriyle); evre etiketli veri bulunursa `findings` benzeri bir "diş gelişim evresi" dedektörü + Willems tablosu daha açıklanabilir olur.

### Önerilen iyileştirmeler (bu repoda hazır)
- **CLAHE** ön işleme (`--clahe`): düşük kontrastlı röntgende erken çürük kenarlarını belirginleştirir. Eğitimde kullanıldıysa meta dosyası (`*_best.json`) sayesinde çıkarımda otomatik uygulanır.
- **Sızıntısız bölme:** Roboflow artırılmış kopyaları (`.rf.`) aynı tarafa gider; yoksa val skoru şişer.
- **Eşik kaydırıcı:** Model düşük eşikte bir kez çalışır, UI'daki güven kaydırıcısı yeniden çıkarım yapmadan filtreler.
- **Sonraki adımlar (veri varsa):** YOLO11n karşılaştırması (`--model yolo11n.pt`), `imgsz=1024`, test-time augmentation, bulgu modelinde sınıf ağırlıklı örnekleme.

### Mevcut veri setinin sınırları\nKullanılan 31 sınıflı veri setinde çürük için evre ve gömülü diş için pozisyon ayrımı yok (tek `Caries`, tek `Impacted tooth`). Bu yüzden erken/ileri evre ve vertikal/horizontal ayrımı şimdilik yapılmıyor; bunlar ayrı etiketleme gerektirir. Aynı veri setindeki `Permanent Teeth` / `Primary teeth` kutuları FDI numarası taşımaz, numaralandırma modeli için FDI etiketli ayrı veri gerekir.\n\n## Kayıp fonksiyonları
YOLOv8'in kutu kaybı zaten **CIoU + DFL**, sınıflandırma kaybı BCE'dir; `configs/augment.yaml` içindeki `box/cls/dfl` ağırlıkları görev bazlı ayarlanır (`findings` için nadir sınıflar adına `cls` 0.5 → 0.7). Gerçek "optimizasyon" ancak val mAP ile ölçülerek yapılabilir; sayılar tahmindir, eğitim sonrası doğrulayın.

## Veri kaynakları (lisansları ve etiket şemalarını kullanmadan önce kontrol edin)
- **Roboflow Universe** — "panoramic dental x-ray", "tooth numbering FDI", "impacted wisdom tooth", "dental caries panoramic" aramaları.
- **DENTEX** (MICCAI 2023, HuggingFace `ibrahimhamamci/DENTEX`) — panoramikte FDI numaralı çürük / derin çürük / periapikal lezyon / gömülü diş etiketleri; en kullanışlı tek kaynak.
- **Tufts Dental Database** — 1000 panoramik, diş/anomali etiketli (başvuru gerekebilir).
- **Children's Dental Panoramic Radiographs** (Figshare) — çocuk panoramikleri; yaş regresörü için uygun.
- **Kaggle** — "Panoramic Dental X-ray" çeşitli diş segmentasyon setleri (poligonlar `prepare.py` ile kutuya çevrilir).

> Bu liste hafızadan; erişilebilirlik ve lisans durumunu kendiniz doğrulayın.

## Klasör yapısı
```
DENT-AI/
├─ app.py                  masaüstü arayüzü (PySide6)
├─ predict.py              komut satırı çıkarımı
├─ configs/                classes.yaml (sınıf+renk), augment.yaml
├─ dentai/
│  ├─ imaging.py           unicode-güvenli G/Ç, CLAHE, letterbox
│  ├─ fdi.py, config.py
│  ├─ data/prepare.py      Roboflow indir, birleştir, %70/%15/%15 böl, data.yaml
│  ├─ train/train.py       YOLOv8n eğitimi (Colab/T4), best.pt kaydı
│  ├─ age/                 yaş regresörü (model + eğitim)
│  ├─ inference/           pipeline, çizim, sonuç modelleri
│  └─ ui/                  tema, görüntüleyici, ana pencere
├─ notebooks/colab_train.ipynb
├─ scripts/                sentetik veri, duman testi, ekran görüntüsü
└─ tests/
```
