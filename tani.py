from pathlib import Path
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
video = cv2.VideoCapture(str(Path(__file__).parent / "video.mp4"))
video.set(cv2.CAP_PROP_POS_FRAMES, 300)
basarili, kare = video.read()

for imgsz in (640, 1280):
    sonuc = model.predict(kare, imgsz=imgsz, conf=0.05, verbose=False)
    print(f"--- imgsz={imgsz}: {len(sonuc[0].boxes)} kutu")
    for b in sonuc[0].boxes:
        print(model.names[int(b.cls)], round(float(b.conf), 2), b.xyxy[0].int().tolist())
    cv2.imwrite(f"tani_{imgsz}.jpg", sonuc[0].plot())