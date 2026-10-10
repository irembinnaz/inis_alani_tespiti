# Güvenli İniş Alanı Tespiti

Bu proje, video akışında güvenli iniş alanı tespit etmek için tasarlanmış bir bilgisayarla görme sistemi geliştirmeyi amaçlar. Çalışma, hareket algılama, YOLO tabanlı engel tespiti, yeşil alan filtresi ve boşluk haritası çıkarımı kullanarak sahadaki en uygun iniş noktasını belirlemeye çalışır.

## Amaç

Otonom hava aracı için güvenli iniş karar mekanizmasının temelini oluşturan çevre algılama ve aday seçim sürecini simüle etmektir.

## Kullanılan yöntemler

- OpenCV ile video işleme
- Background Subtractor MOG2 ile hareket tespiti
- YOLO ile engel algılama
- HSV maskesi ile yeşil alan filtresi
- distanceTransform ile güvenli boşluk analizi
- Histerezis mantığı ile kararlı iniş noktası seçimi

## Gereksinimler

Python 3.10+ önerilir.

```bash
pip install -r requirements.txt
```

## Çalıştırma

Varsayılan video dosyasını kullanmak için:

```bash
python main.py
```

Özel video yolu belirtmek için:

```bash
python main.py --video .\videos\ornek_video.mp4
```

Çıktı video kaydetmek için:

```bash
python main.py --video .\videos\ornek_video.mp4 --output .\outputs\sonuc.mp4 --no-window
```

Not: Kod ilk frame üzerinde saha seçim penceresini açar. Fare ile alanı seçip ENTER tuşuna basarak onaylayın.

## Test akışı

Aşağıdaki sırayı takip ederek projeyi daha net bir şekilde test edebilirsiniz:

### 1) Mevcut videonuz ile ilk test
- Projeyi mevcut videonuzla çalıştırın.
- Saha alanını seçin.
- Sonuç ekranını izleyin.
- Aday nokta ve güvenlik skorunun doğru çalıştığını kontrol edin.
- Ekran görüntüsü alın.

### 2) İkinci video ile karşılaştırmalı test
- Başka bir video ile çalıştırın.
- Aynı hesaplama akışının tekrar çalıştığını kontrol edin.
- Farklı senaryolarda aday seçiminin nasıl değiştiğini izleyin.
- Ekran görüntüsü alın.

### 3) Sonuç değerlendirmesi
- Hangi videoda daha stabil aday seçimi yapıldı?
- Hangi senaryoda yanlış engel tespiti oldu?
- Hangi eşik değerlerinin değiştirilmesi gerektiğini not alın.

## Önemli notlar

- YOLO modeli `yolo11n.pt` dosyasını otomatik kullanır.
- `--output` ile kayıtlı video oluşturabilirsiniz.
- `--no-window` seçeneği, arayüz göstermeden çalıştırma için kullanılır; bunu özellikle sunum veya otomatik testlerde tercih edebilirsiniz.

## Geliştirme önerileri

- güvenlik skoru ve aday sıralaması daha ileri geliştirilebilir
- kararlı iniş noktası için zaman ortalamaları eklenebilir
- farklı araziler için eşik değerleri dinamik hale getirilebilir
- daha güvenli karar verme için çoklu sensör birleştirme düşünülebilir

## Proje çıktısı örneği

Yazdırılan terminal çıktısı aşağıya benzer bir yapı sunar:

```text
Toplam kare: 230 | Aday değişim sayısı: 3 | Uygun alan olmayan kare: 18
Video girdi: ...\video.mp4
Çıktı video: ...\outputs\sonuc.mp4
```

Bu çıktı, çalışma süresi, aday değişim sayısı ve uygun alan bulunamayan kare sayısı hakkında bilgi verir.
