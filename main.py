import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------- AYARLAR
HUCRE = 40            # grid hücre boyutu (piksel)
PAY = 15              # her engelin çevresine eklenen güvenlik payı (piksel)
MIN_GUVENLI = 40      # seçili noktanın boşluğu bunun altına düşerse aday zorunlu değişir
IYILESME = 60         # yeni aday seçiliden en az bu kadar daha boşsa geçilir
MIN_YARICAP = 30      # en iyi aday bile bundan az boşluktaysa "UYGUN ALAN YOK"
YOK_CIKIS_PAYI = 10   # "yok" durumundan çıkmak için eşiğin bu kadar üstü gerekir
YOLO_ARALIK = 10      # YOLO kaç karede bir çalışsın (büyütürsen hızlanır)
YOLO_BOSLUK = 60      # YOLO'ya verilen kırpımda sahanın çevresindeki boşluk (piksel)
MAX_KUTU = 220        # bundan büyük YOLO kutuları yanlış tespit sayılır (piksel)
YESIL_ALT = (22, 35, 20)     # HSV alt sınır: ağaç/çim rengi
YESIL_UST = (95, 255, 255)   # HSV üst sınır
YESIL_MIN_ALAN = 800         # bundan küçük yeşil lekeler ağaç sayılmaz (piksel^2)
CATI_MIN_ALAN = 3000        # bundan küçük açık/kırmızı lekeler çatı sayılmaz (araçlar elenir) (piksel^2)
CATI_KIRMIZI_A = 152        # Lab 'a' kanalı bundan büyükse kırmızı (kırmızı çatı)
CATI_PARLAK_L = 140         # Lab 'L' bundan büyük ve renksizse açık gri/beyaz (açık çatı)
ISINMA = 30           # arka plan modelinin öğrenmesi için ilk kareler
MIN_ALAN = 30         # bundan küçük hareket lekeleri gürültü sayılır (piksel^2)
MAX_ALAN = 20000      # bundan büyük lekeler (ışık değişimi vb.) yok sayılır
MAX_ON_PLAN = 0.25    # karenin %25'inden fazlası "hareket" ise o karede hareketi güvenme


def parse_args():
    p = argparse.ArgumentParser(description="Video üzerinde güvenli iniş alanı adayı seçimi.")
    p.add_argument("--video", default=str(Path(__file__).parent / "video.mp4"), help="İşlenecek video")
    p.add_argument("--model", default="yolo11m.pt", help="YOLO ağırlığı (yolo11n.pt, yolo11s.pt, yolo11m.pt ...)")
    p.add_argument("--conf", type=float, default=0.10, help="YOLO güven eşiği (düşürürsen daha çok kutu çıkar)")
    p.add_argument("--output", default=None, help="İsteğe bağlı: sonucu bu videoya kaydet")
    p.add_argument("--no-window", action="store_true", help="Pencere göstermeden çalıştır")
    p.add_argument("--yasak", action="store_true", help="Modelin göremediği engelleri elle yasak alan olarak işaretle")
    p.add_argument("--no-motion", action="store_true", help="Hareket tespitini kapat (kamera sabit değilse kullan)")
    p.add_argument("--max-kutu", type=int, default=MAX_KUTU, help="Bundan büyük YOLO kutuları yanlış tespit sayılır (piksel); yakın çekimde büyüt")
    return p.parse_args()


args = parse_args()

# ---------------------------------------------------------------- MODEL
model = YOLO(args.model)
model_adi = Path(args.model).name
try:
    parametre_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    model_yazisi = f"{model_adi} ({parametre_m:.1f}M parametre)"
except Exception:
    model_yazisi = model_adi
print("Model:", model_yazisi)  # n ~2.6M, s ~9.4M, m ~20M parametre olmalı

# ---------------------------------------------------------------- VİDEO VE SAHA
video = cv2.VideoCapture(args.video)
if not video.isOpened():
    raise SystemExit(f"Video açılamadı: {args.video}")

basarili, ilk_kare = video.read()
if not basarili:
    raise SystemExit("Video okunamadı.")

# Sahayı fareyle seç, ENTER ile onayla
sx, sy, sw, sh = cv2.selectROI("Saha sec - ENTER ile onayla", ilk_kare, False)
cv2.destroyWindow("Saha sec - ENTER ile onayla")
if sw == 0 or sh == 0:
    raise SystemExit("Saha seçilmedi.")

# Yasak alanlar (isteğe bağlı, varsayılan KAPALI): modelin göremediği engelleri elle işaretle.
# Açmak için: python main.py --yasak   | Her alan için sürükle + ENTER, bitince ESC.
YASAK = ()
if args.yasak:
    YASAK = cv2.selectROIs("Yasak alanlar - ENTER: ekle, ESC: bitir", ilk_kare, False)
    cv2.destroyWindow("Yasak alanlar - ENTER: ekle, ESC: bitir")

# İsteğe bağlı çıkış videosu
writer = None
if args.output:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fps_video = video.get(cv2.CAP_PROP_FPS) or 25.0
    boy = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    en = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (en, boy))

# ---------------------------------------------------------------- HAREKET TESPİTİ HAZIRLIĞI
arka_plan = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=False)
cekirdek = np.ones((3, 3), np.uint8)
referans_gri = cv2.cvtColor(ilk_kare, cv2.COLOR_BGR2GRAY).astype(np.float32)


# ---------------------------------------------------------------- AĞAÇ / ÇİM MASKESİ
def yesil_maske(kare):
    """Ağaç/çim gibi yeşil alanları renkten bulur (YOLO ve hareket tespiti bunları görmez).
    Ağaçlar sabit olduğu için sadece ilk karede, bir kez hesaplanır."""
    hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, YESIL_ALT, YESIL_UST)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Yaprakların arasındaki küçük delikleri kapat
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    # Küçük yeşil lekeleri ele (yeşil araba, tabela vb.)
    n, etiket, ist, _ = cv2.connectedComponentsWithStats(m)
    sonuc = np.zeros_like(m)
    for i in range(1, n):
        if ist[i, cv2.CC_STAT_AREA] >= YESIL_MIN_ALAN:
            sonuc[etiket == i] = 255
    return cv2.dilate(sonuc, np.ones((5, 5), np.uint8))


AGAC = yesil_maske(ilk_kare)[sy:sy + sh, sx:sx + sw]  # sadece saha içi
AGAC_KONTUR, _ = cv2.findContours(AGAC, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def cati_maske(kare):
    """Kırmızı ve açık renkli (beyaz/gri) geniş yüzeyleri çatı sayar.
    Yol asfaltından renk olarak ayrılırlar. Yaya geçidi çizgileri ince olduğu için,
    araçlar da küçük olduğu için alan/şekil filtresiyle elenir.
    SINIR: koyu renkli çatılar asfalttan renkle ayırt edilemez, bu yöntem onları yakalamaz."""
    lab = cv2.cvtColor(cv2.GaussianBlur(kare, (0, 0), 2), cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    kirmizi = a > CATI_KIRMIZI_A
    acik = (L > CATI_PARLAK_L) & (abs(a.astype(int) - 128) < 14) & (abs(b.astype(int) - 128) < 16)
    m = ((kirmizi | acik) * 255).astype(np.uint8)
    elips = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, elips)   # ince çizgileri ve küçük leke/araçları sil
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, elips)  # çatıdaki küçük delikleri kapat
    n, etiket, ist, _ = cv2.connectedComponentsWithStats(m)
    sonuc = np.zeros_like(m)
    for i in range(1, n):
        if ist[i, cv2.CC_STAT_AREA] >= CATI_MIN_ALAN:
            sonuc[etiket == i] = 255
    return cv2.dilate(sonuc, np.ones((5, 5), np.uint8))


CATI = cati_maske(ilk_kare)[sy:sy + sh, sx:sx + sw]  # sadece saha içi
CATI_KONTUR, _ = cv2.findContours(CATI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ---------------------------------------------------------------- YOLO KIRPIMI
# YOLO'ya tüm kare yerine saha + çevresi verilir (daha hızlı)
KARE_Y, KARE_G = ilk_kare.shape[:2]
KX1 = max(sx - YOLO_BOSLUK, 0)
KY1 = max(sy - YOLO_BOSLUK, 0)
KX2 = min(sx + sw + YOLO_BOSLUK, KARE_G)
KY2 = min(sy + sh + YOLO_BOSLUK, KARE_Y)
YOLO_BOYUT = int(np.ceil(max(KY2 - KY1, KX2 - KX1) / 32) * 32)


def yolo_kutulari_bul(kare):
    """Sınıfa bakmadan her tespiti engel sayar (model tepeden araçları 'train',
    'parking meter' gibi yanlış sınıflara atıyor), ama dev kutuları eler."""
    parca = kare[KY1:KY2, KX1:KX2]
    sonuc = model.predict(parca, imgsz=YOLO_BOYUT, conf=args.conf,
                          agnostic_nms=True, verbose=False)
    kutular = sonuc[0].boxes.xyxy.cpu().numpy().astype(int).reshape(-1, 4)

    genislik = kutular[:, 2] - kutular[:, 0]
    yukseklik = kutular[:, 3] - kutular[:, 1]
    kutular = kutular[(genislik < args.max_kutu) & (yukseklik < args.max_kutu)]

    # Kırpım koordinatını tam kare koordinatına çevir
    return kutular + np.array([KX1, KY1, KX1, KY1])


def hareket_kutulari(kare, kare_no):
    """Hareket eden nesneleri (insan, araç, fark etmez) kutu olarak döner.
    Dronun hafif kaymasını önce telafi eder, yoksa kenarlar hareket sanılır."""
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(referans_gri, gri)

    h, w = kare.shape[:2]
    kaydirma = np.float32([[1, 0, -dx], [0, 1, -dy]])
    hizali = cv2.warpAffine(kare, kaydirma, (w, h), borderMode=cv2.BORDER_REPLICATE)

    maske = arka_plan.apply(hizali)
    if kare_no < ISINMA:
        return np.empty((0, 4), dtype=int)

    # Karenin büyük kısmı değiştiyse (ani ışık değişimi vb.) güvenme
    if (maske > 0).mean() > MAX_ON_PLAN:
        return np.empty((0, 4), dtype=int)

    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, cekirdek)
    maske = cv2.dilate(maske, cekirdek, iterations=2)
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kutular = []
    for k in konturlar:
        if MIN_ALAN <= cv2.contourArea(k) <= MAX_ALAN:
            x, y, kw, kh = cv2.boundingRect(k)
            # Hizalanmış koordinatı orijinal kareye geri çevir
            kutular.append([int(x + dx), int(y + dy), int(x + kw + dx), int(y + kh + dy)])
    return np.array(kutular, dtype=int).reshape(-1, 4)


# ---------------------------------------------------------------- ADAY SEÇİMİ
def en_guvenli_hucre(kutular):
    """Saha içinde engellere en uzak hücreyi bulur.
    Döner: (merkez, boşluk, mesafe_haritası). Merkez saha içi koordinattır (x, y)."""
    maske = np.full((sh, sw), 255, dtype=np.uint8)  # 255 = boş, 0 = engel

    for x1, y1, x2, y2 in kutular:
        a1 = max(x1 - sx - PAY, 0)
        b1 = max(y1 - sy - PAY, 0)
        a2 = min(x2 - sx + PAY, sw)
        b2 = min(y2 - sy + PAY, sh)
        if a2 > a1 and b2 > b1:
            maske[b1:b2, a1:a2] = 0

    # Ağaç/çim alanları iniş için uygun değil
    maske[AGAC > 0] = 0
    maske[CATI > 0] = 0  # çatılar da iniş için uygun değil

    # Operatörün işaretlediği yasak alanlar
    for yx, yy, yw, yh in YASAK:
        maske[max(yy - sy, 0):max(yy + yh - sy, 0), max(yx - sx, 0):max(yx + yw - sx, 0)] = 0

    # Saha kenarını da engel say, böylece boş sahada merkez tercih edilir
    kenarli = cv2.copyMakeBorder(maske, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    mesafe = cv2.distanceTransform(kenarli, cv2.DIST_L2, 5)[1:-1, 1:-1]

    yarim = HUCRE // 2
    hucre_mesafeleri = mesafe[yarim::HUCRE, yarim::HUCRE]
    if hucre_mesafeleri.size == 0:
        return None

    i, j = np.unravel_index(np.argmax(hucre_mesafeleri), hucre_mesafeleri.shape)
    merkez = (yarim + j * HUCRE, yarim + i * HUCRE)
    return merkez, float(hucre_mesafeleri[i, j]), mesafe


def yazi(kare, metin, konum, renk, olcek=0.8):
    """Önce kalın siyah, üstüne renkli yazı: her arka planda okunur."""
    cv2.putText(kare, metin, konum, cv2.FONT_HERSHEY_SIMPLEX, olcek, (0, 0, 0), 4)
    cv2.putText(kare, metin, konum, cv2.FONT_HERSHEY_SIMPLEX, olcek, renk, 2)


# ---------------------------------------------------------------- ANA DÖNGÜ
video.set(cv2.CAP_PROP_POS_FRAMES, 0)
kare_no = 0
yolo_kutulari = np.empty((0, 4), dtype=int)  # aradaki karelerde son sonuç kullanılır
secili = None         # şu an seçili iniş noktası (saha içi x, y)
degisim_sayisi = 0    # seçili nokta kaç kez değişti (ilk seçim sayılmaz)
yok_kare_sayisi = 0   # kaç karede uygun alan bulunamadı
baslangic = time.time()

while True:
    basarili, kare = video.read()
    if not basarili:
        break

    # 1) YOLO kutuları (her karede değil, belli aralıkla)
    if kare_no % YOLO_ARALIK == 0:
        yolo_kutulari = yolo_kutulari_bul(kare)

    # 2) Hareket kutuları (YOLO'nun kaçırdığı küçük insanları da yakalar)
    hareket = np.empty((0, 4), dtype=int) if args.no_motion else hareket_kutulari(kare, kare_no)

    # 3) İkisi de "engel"
    engeller = np.vstack([yolo_kutulari, hareket])

    # Çizim: kırmızı = YOLO, turuncu = hareket, mor = ağaç, sarı = çatı, kırmızı ince = yasak alan
    for x1, y1, x2, y2 in yolo_kutulari:
        cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for x1, y1, x2, y2 in hareket:
        cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 165, 255), 1)
    cv2.rectangle(kare, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)
    cv2.drawContours(kare, AGAC_KONTUR, -1, (255, 0, 255), 1, offset=(sx, sy))
    cv2.drawContours(kare, CATI_KONTUR, -1, (0, 255, 255), 1, offset=(sx, sy))
    for yx, yy, yw, yh in YASAK:
        cv2.rectangle(kare, (yx, yy), (yx + yw, yy + yh), (0, 0, 255), 1)

    # 4) Aday seçimi (histerezisli + "uygun alan yok" kararı)
    sonuc_aday = en_guvenli_hucre(engeller)

    if sonuc_aday is None:
        secili = None
    else:
        en_iyi, en_iyi_bosluk, mesafe = sonuc_aday

        # "Yok" durumundan çıkmak için biraz daha fazla boşluk iste (titremeyi önler)
        esik = MIN_YARICAP if secili is not None else MIN_YARICAP + YOK_CIKIS_PAYI

        if en_iyi_bosluk < esik:
            secili = None  # hiçbir yer yeterince güvenli değil: inme, havada kal
        elif secili is None:
            secili = en_iyi
        else:
            secili_bosluk = float(mesafe[secili[1], secili[0]])  # seçili noktanın BU karedeki boşluğu
            secili_tehlikede = secili_bosluk < MIN_GUVENLI
            yeni_belirgin_iyi = en_iyi_bosluk > secili_bosluk + IYILESME
            if (secili_tehlikede or yeni_belirgin_iyi) and en_iyi != secili:
                secili = en_iyi
                degisim_sayisi += 1

    if secili is not None:
        bosluk = float(mesafe[secili[1], secili[0]])
        merkez = (sx + secili[0], sy + secili[1])
        cv2.circle(kare, merkez, int(bosluk), (0, 255, 0), 2)
        cv2.circle(kare, merkez, 5, (0, 255, 0), -1)
        yazi(kare, f"Aday: {bosluk:.0f}px bosluk | degisim: {degisim_sayisi}", (15, 35), (0, 255, 0))
    else:
        yok_kare_sayisi += 1
        yazi(kare, "UYGUN ALAN YOK", (15, 40), (0, 0, 255), 1.0)

    yazi(kare, f"Model: {model_yazisi}", (15, 70), (255, 255, 255), 0.6)

    if writer is not None:
        writer.write(kare)

    kare_no += 1
    if not args.no_window:
        cv2.imshow("Inis alani - Q ile cikis", kare)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# ---------------------------------------------------------------- BİTİŞ
sure = time.time() - baslangic
video.release()
if writer is not None:
    writer.release()
if not args.no_window:
    cv2.destroyAllWindows()

print(f"Model: {model_yazisi}")
print(f"Toplam kare: {kare_no} | Aday değişim sayısı: {degisim_sayisi} | Uygun alan olmayan kare: {yok_kare_sayisi}")
print(f"Süre: {sure:.1f} sn | Ortalama FPS: {kare_no / sure:.1f}" if sure > 0 else "")