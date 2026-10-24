import os
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()
url = os.getenv("CAMERA_URL")
if not url:
    raise SystemExit("CAMERA_URL não encontrada no .env")

# YOLO nano: modelo pequeno, roda na CPU. Baixa os pesos na 1ª vez.
modelo = YOLO("models/yolo11n.pt")
CLASSE_PESSOA = 0         # no dataset COCO, a classe 0 é "person"
CONFIANCA_MINIMA = 0.5    # descarta detecções duvidosas

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit("Não consegui abrir o stream.")

frames = 0
inicio = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        print("Falha ao ler o quadro.")
        break

    resultado = modelo.predict(
        frame, classes=[CLASSE_PESSOA], conf=CONFIANCA_MINIMA, verbose=False
    )[0]
    for caixa in resultado.boxes:
        x1, y1, x2, y2 = map(int, caixa.xyxy[0])
        w, h = x2 - x1, y2 - y1
        conf = float(caixa.conf[0])
        # h/w > 1: em pé; h/w < 1: deitado (base da regra de queda)
        cor = (0, 255, 0) if h / w >= 1 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(frame, f"pessoa {conf:.2f}  h/w={h / w:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    frames += 1
    fps = frames / (time.time() - inicio)
    cv2.putText(frame, f"{fps:.1f} fps", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Detector de pessoa", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
