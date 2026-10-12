import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("CAMERA_URL")
if not url:
    raise SystemExit("CAMERA_URL não encontrada no .env")

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit("Não consegui abrir o stream.")

# Modelo do fundo: aprende a cena "vazia" e marca o que muda
fundo = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=True
)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
AREA_MINIMA = 1500  # ignora manchas pequenas (ruído, insetos)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Falha ao ler o quadro.")
        break

    mascara = fundo.apply(frame)
    # sombras vêm como cinza (127); só o branco (255) é movimento real
    _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)
    # limpa ruído e fecha buracos na silhueta
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.dilate(mascara, kernel, iterations=2)

    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contornos:
        if cv2.contourArea(c) < AREA_MINIMA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"h/w={h / w:.2f}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    cv2.imshow("Mascara", mascara)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()