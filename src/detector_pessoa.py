import csv
import os
import sys
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()
# Se passar um arquivo na linha de comando, usa ele; senão, a câmera do .env
if len(sys.argv) > 1:
    url = sys.argv[1]
else:
    url = os.getenv("CAMERA_URL")
if not url:
    raise SystemExit("CAMERA_URL não encontrada no .env")
eh_arquivo = os.path.isfile(url)
if not eh_arquivo and not url.startswith("rtsp://"):
    raise SystemExit(f"Arquivo não encontrado: {url}")

# YOLO nano: modelo pequeno, roda na CPU. Baixa os pesos na 1ª vez.
modelo = YOLO("models/yolo11n.pt")
CLASSE_PESSOA = 0         # no dataset COCO, a classe 0 é "person"
CONFIANCA_MINIMA = 0.35   # deitado, a confiança cai para ~0.5; 0.5 perdia quadros

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit("Não consegui abrir o stream.")
fps_video = cap.get(cv2.CAP_PROP_FPS)

# CSV com um registro por quadro, para analisar depois (pandas)
os.makedirs("data", exist_ok=True)
if eh_arquivo:
    nome = os.path.splitext(os.path.basename(url))[0]
else:
    nome = f"camera_{time.strftime('%Y%m%d_%H%M%S')}"
caminho_csv = f"data/{nome}.csv"
arquivo_csv = open(caminho_csv, "w", newline="")
escritor = csv.writer(arquivo_csv)
escritor.writerow(["quadro", "tempo_s", "pessoas", "conf",
                   "x1", "y1", "x2", "y2", "hw", "centro_x", "centro_y"])

frames = 0
inicio = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        print("Falha ao ler o quadro (ou o vídeo acabou).")
        break
    altura, largura = frame.shape[:2]
    # no arquivo, o tempo vem do vídeo; ao vivo, do relógio
    if eh_arquivo and fps_video > 0:
        tempo = frames / fps_video
    else:
        tempo = time.time() - inicio

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

    # Por enquanto registra só a pessoa mais confiável do quadro.
    # Quadro sem ninguém também vira linha (com campos vazios): os "buracos"
    # são informação importante para a regra de imobilidade.
    if len(resultado.boxes):
        melhor = max(resultado.boxes, key=lambda c: float(c.conf[0]))
        x1, y1, x2, y2 = map(int, melhor.xyxy[0])
        escritor.writerow([
            frames, round(tempo, 3), len(resultado.boxes),
            round(float(melhor.conf[0]), 3), x1, y1, x2, y2,
            round((y2 - y1) / (x2 - x1), 3),
            # centro normalizado: 0 = esquerda/topo, 1 = direita/base
            round((x1 + x2) / 2 / largura, 3), round((y1 + y2) / 2 / altura, 3),
        ])
    else:
        escritor.writerow([frames, round(tempo, 3), 0] + [""] * 8)

    frames += 1
    fps = frames / (time.time() - inicio)
    cv2.putText(frame, f"{fps:.1f} fps", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # reduz só a imagem exibida, para caber na tela (vídeos grandes)
    if largura > 960:
        frame = cv2.resize(frame, None, fx=960 / largura, fy=960 / largura)
    cv2.imshow("Detector de pessoa", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

arquivo_csv.close()
cap.release()
cv2.destroyAllWindows()
print(f"{frames} quadros registrados em {caminho_csv}")
