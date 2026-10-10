# Güvenli İniş Alanı Tespiti

Sabit kameralı bir havadan videoda, **seçilen saha içinde** iniş için en güvenli noktayı bulan ve sahaya engel girince seçimi güncelleyen bir bilgisayarla görme prototipi.

> **Durum:** Aktif geliştirme (v0.1). Tek bir sahnede (aşağıdaki test videosu) denendi. Sınırlılıklar için aşağıya bakın.

## Problem

Bir İHA'nın otonom inişi için "şu an nereye inebilirim?" sorusuna sürekli ve kararlı bir cevap gerekir. Bu projede o kararın görüntü tarafı ele alınıyor: video üzerinde insan/araç gibi engelleri bul, iniş için uygun olmayan zeminleri (ağaç, çatı) ele, kalan boşlukta engellerden en uzak noktayı seç ve bu seçimi her karede zıplatmadan güncelle.

## Nasıl çalışıyor

1. **Saha seçimi:** İlk karede fareyle iniş sahası seçilir.
2. **YOLO (engel tespiti):** Saha ve çevresi kırpılıp YOLO'ya verilir (her 10 karede bir). Sınıf bilgisi kullanılmaz, bulunan her kutu engel sayılır; aşırı büyük kutular yanlış tespit olarak elenir.
3. **Hareket tespiti:** MOG2 arka plan çıkarma ile hareket eden her şey yakalanır (YOLO'nun göremediği küçük yayalar dahil). Kamera hafif kayarsa görüntü önce hizalanır.
4. **Zemin uygunluğu:** Renkten bulunan ağaç/çim (HSV) ve kırmızı/açık renkli çatılar (Lab) iniş için uygun olmayan alan sayılır.
5. **Boşluk haritası:** Engeller güvenlik payıyla maskeye işlenir, `distanceTransform` ile her pikselin en yakın engele uzaklığı hesaplanır.
6. **Aday seçimi:** Grid hücreleri arasında uzaklığı en büyük olan seçilir.
7. **Histerezis:** Seçili nokta tehlikeye girmedikçe ve yeni aday belirgin şekilde daha iyi olmadıkça değiştirilmez (aday zıplamaz).
8. **"Uygun alan yok":** En iyi aday bile minimum boşluğun altındaysa aday gösterilmez.

Ekranda: kırmızı kutu = YOLO, turuncu kutu = hareket, mor çizgi = ağaç/çim, sarı çizgi = çatı, yeşil daire = seçili iniş adayı ve boşluğu.

## Geliştirirken karşılaşılan sorunlar

| Sorun | Çözüm |
|---|---|
| Hazır YOLO modeli tepeden çekilmiş araçları "train", "parking meter" gibi yanlış sınıflara atıyor (`tani.py` çıktısı) | Sınıfa bakmadan her tespiti engel say, dev kutuları ele. `yolo11n` park halindeki bazı araçları kaçırdı, `yolo11m` ile bu videoda ekran görüntülerimde yakalandı (sistematik ölçüm yapılmadı). |
| Yayalar videoda yaklaşık 10x20 piksel, YOLO göremiyor | MOG2 ile hareket tespiti |
| Kamera yaklaşık 3 piksel kayınca kenarlar "hareket" sayıldı (bir karede 216 sahte kutu) | Faz korelasyonu ile kayma ölçüp görüntüyü hizalama (aynı karede 12'ye düştü) |
| Ağaçlar ve çatılar hiçbir tespitte engel görünmüyor, aday ağaç altına oturuyordu | Renk tabanlı ağaç/çim ve çatı maskeleri |
| Aday karelerde bir yerden başka yere zıplıyordu | Histerezis |
| Güvenli yer yokken yine de aday gösteriliyordu | "Uygun alan yok" kararı |

## Kurulum ve çalıştırma

```
pip install -r requirements.txt
python main.py --video video.mp4
```

Başlayınca ilk karede saha için fareyle bir dikdörtgen sürükleyip **ENTER**'a basın. Çıkmak için **Q**.

| Bayrak | Anlamı |
|---|---|
| `--video` | İşlenecek video (varsayılan: `video.mp4`) |
| `--model` | YOLO ağırlığı (varsayılan: `yolo11m.pt`, ilk çalıştırmada kendiliğinden iner) |
| `--conf` | YOLO güven eşiği (varsayılan: 0.10) |
| `--output` | Sonucu video olarak kaydet |
| `--no-window` | Pencere açmadan çalıştır |
| `--yasak` | Modelin göremediği engelleri elle yasak alan olarak işaretle |
| `--no-motion` | Hareket tespitini kapat (kamera sabit değilse) |
| `--max-kutu` | Bundan büyük YOLO kutularını yanlış tespit say (piksel) |

Program bitince terminalde kullanılan model, toplam kare, aday değişim sayısı, "uygun alan olmayan" kare sayısı ve ortalama FPS yazılır.

## Test videosu

Mixkit'ten [Busy intersection aerial view](https://mixkit.co/free-stock-video/busy-intersection-aerial-view-60/) klibi (`@marcolopez`, Mixkit Stock Video Free License) kullanıldı. Video bu depoya **dahil değildir**. Yukarıdaki sayfadan HD Ready sürümünü indirip proje klasörüne `video.mp4` adıyla koyun.

## Sınırlılıklar

- Yalnızca tek sahnede (yüksekten, sabit kameralı, tam tepeden) denendi. Ayarların çoğu **piksel cinsinden** ve bu videoya göre seçildi; farklı irtifa, çözünürlük veya açıda yeniden ayarlanması gerekir.
- Kamera hareketli videolarda çalışması beklenmiyor (yalnızca küçük kaymalar telafi ediliyor).
- Ağaç ve çatı tespiti renk eşiğine dayanıyor, gerçek bir segmentasyon değil. Koyu renkli çatılar asfalttan ayırt edilemiyor.
- Hareket tespiti duran insanı görmez. Park halindeki araçlar YOLO'nun tespit başarısına bağlı.
- Hareket kutuları araçların gölgesini de içine alıyor.
- Doğruluk etiketli veriyle **ölçülmedi**; sonuçlar gözle değerlendirildi.

## Yapılacaklar

- [ ] FPS, aday değişim sayısı ve "uygun alan yok" oranı için ölçüm tablosu (model ve parametre karşılaştırmalı)
- [ ] Elle etiketlenmiş birkaç kare ile YOLO ve YOLO + hareket tespiti karşılaştırması
- [ ] Farklı videolarda test ve parametrelerin çözünürlüğe göre ölçeklenmesi
- [ ] VisDrone ile fine-tune (tepeden araç/insan tespiti)
- [ ] Renk eşikleri yerine segmentasyon modeli
- [ ] Kamera hareketi için görüntü hizalama (homografi)

## Not

Bu projeyi geliştirirken yapay zeka asistanlarından (Claude, GitHub Copilot) yararlandım.