# Yapay Zeka ile Obje Tespiti

Python ve OpenCV kullanarak YOLOv3 (You Only Look Once) nesne tespit modelini 
çalıştıran basit ve etkili bir görüntü analiz aracı.

## Özellikler

- **80 farklı nesne sınıfı** tespiti (COCO veri seti)
- **Resim, Video veya WebCam üzerinden işleme** veya **klasör toplu işleme** desteği
- **Görselleştirilmiş çıktı ve güven düzeyi**
- **Hızlı kurulum** - sadece OpenCV ve NumPy gereksinimi
- **Özelleştirilebilir nesne etiketleri**

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
git clone https://github.com/wolkansec/ai-object-detector
cd ai-object-detector

# 2. Gereksinimlerin Kurulması
pip install -r requirements.txt

# 4. Projenin Çalıştırılması
# python3 main.py
