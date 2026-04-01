# Yapay Zeka ile Obje Tespiti

Python ve OpenCV kullanarak YOLOv3 (You Only Look Once) nesne tespit modelini 
çalıştıran basit ve etkili bir görüntü analiz aracı.

## Özellikler

- **80 farklı nesne sınıfı** tespiti (COCO veri seti)
- **Tek resim işleme** veya **klasör toplu işleme** desteği
- **Görselleştirilmiş çıktı** ile sınırlayıcı kutular ve güven skorları
- **Hızlı kurulum** - sadece OpenCV ve NumPy gereksinimi
- **Özelleştirilebilir parametreler** (güven eşiği, NMS threshold)

## Tespit Edilebilen Nesneler

İnsanlar, araçlar (araba, bisiklet, otobüs, uçak), hayvanlar (kedi, köpek, kuş, at), 
elektronik eşyalar (dizüstü bilgisayar, telefon, klavye), günlük nesneler 
(sırt çantası, şemsiye, kitap, kupa) ve daha fazlası.

## Kurulum

### Gereksinimler

- Python 3.7+
- OpenCV 4.x
- NumPy

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/kullaniciadi/yolo-goruntu-analiz.git
cd yolo-goruntu-analiz

# 2. Sanal ortam oluştur (önerilir)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. YOLO model dosyalarını indir
# yolov3.weights: https://pjreddie.com/media/files/yolov3.weights
# yolov3.cfg: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true
# İndirilen dosyaları proje kök dizinine yerleştir
