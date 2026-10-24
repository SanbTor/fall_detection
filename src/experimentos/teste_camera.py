import os
import time

# Usa TCP no RTSP: mais estável que UDP em Wi-Fi
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
from dotenv import load_dotenv

load_dotenv()  # lê o arquivo .env
url = os.getenv("CAMERA_URL")
if not url:
    raise SystemExit("CAMERA_URL não encontrada no .env")

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)  # abre o stream
if not cap.isOpened():
    raise SystemExit("Não consegui abrir o stream. Confira URL, usuário e senha.")

frames = 0
inicio = time.time()
while True:
    ok, frame = cap.read()  # lê um quadro por vez
    if not ok:
        print("Falha ao ler o quadro.")
        break
    frames += 1
    fps = frames / (time.time() - inicio)
    cv2.putText(frame, f"{fps:.1f} fps", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # tecla q fecha
        break

cap.release()
cv2.destroyAllWindows()