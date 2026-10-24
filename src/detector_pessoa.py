import argparse
import csv
import math
import os
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

from regra_queda import ANGULO_DEITADO, TEMPO_ALERTA, RegraQueda

# Uso: python src/detector_pessoa.py [arquivo_de_video] [--tempo 10]
parser = argparse.ArgumentParser()
parser.add_argument("video", nargs="?",
                    help="arquivo de vídeo; sem ele, usa a câmera do .env")
parser.add_argument("--tempo", type=float, default=TEMPO_ALERTA,
                    help="segundos no chão até alertar")
args = parser.parse_args()

load_dotenv()
url = args.video or os.getenv("CAMERA_URL")
if not url:
    raise SystemExit("CAMERA_URL não encontrada no .env")
eh_arquivo = os.path.isfile(url)
if not eh_arquivo and not url.startswith("rtsp://"):
    raise SystemExit(f"Arquivo não encontrado: {url}")

# YOLO-pose nano: além do retângulo, acha 17 pontos do corpo (keypoints)
modelo = YOLO("models/yolo11n-pose.pt")
CONFIANCA_MINIMA = 0.35   # deitado, a confiança cai para ~0.5; 0.5 perdia quadros
CONFIANCA_CONTAGEM = 0.6  # para contar "outra pessoa" (abajur chegou a 0.55)
CONFIANCA_PONTO = 0.5     # confiança mínima de cada ponto usado no ângulo

# Índices dos keypoints no padrão COCO
OMBRO_ESQ, OMBRO_DIR = 5, 6
QUADRIL_ESQ, QUADRIL_DIR = 11, 12


def angulo_tronco(pontos, confs):
    """Ângulo do tronco em relação à vertical: 0° = em pé, 90° = deitado.

    Usa o meio dos ombros e o meio dos quadris. Devolve None se algum
    desses 4 pontos não foi visto com confiança suficiente.
    """
    indices = [OMBRO_ESQ, OMBRO_DIR, QUADRIL_ESQ, QUADRIL_DIR]
    if any(confs[i] < CONFIANCA_PONTO for i in indices):
        return None
    ombro_x = (pontos[OMBRO_ESQ][0] + pontos[OMBRO_DIR][0]) / 2
    ombro_y = (pontos[OMBRO_ESQ][1] + pontos[OMBRO_DIR][1]) / 2
    quadril_x = (pontos[QUADRIL_ESQ][0] + pontos[QUADRIL_DIR][0]) / 2
    quadril_y = (pontos[QUADRIL_ESQ][1] + pontos[QUADRIL_DIR][1]) / 2
    dx = abs(ombro_x - quadril_x)
    dy = abs(ombro_y - quadril_y)
    return math.degrees(math.atan2(dx, dy))


def contar_pessoas(confs, principal):
    """Conta pessoas para a regra "2+ pessoas não alerta".

    A pessoa principal (a mais confiável) sempre conta, mesmo com confiança
    baixa: deitada no chão ela pode ter só ~0.4. As outras só contam com
    confiança alta, para um objeto parecido com gente (ex.: abajur) não
    virar uma "segunda pessoa" e bloquear um alerta real.
    """
    if not confs:
        return 0
    outras = [c for i, c in enumerate(confs) if i != principal]
    return 1 + sum(c >= CONFIANCA_CONTAGEM for c in outras)


cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit("Não consegui abrir o stream.")
fps_video = cap.get(cv2.CAP_PROP_FPS)

# CSV com um registro por quadro, para analisar depois (pandas)
os.makedirs("data/resultados", exist_ok=True)
if eh_arquivo:
    nome = os.path.splitext(os.path.basename(url))[0]
else:
    nome = f"camera_{time.strftime('%Y%m%d_%H%M%S')}"
caminho_csv = f"data/resultados/{nome}.csv"
arquivo_csv = open(caminho_csv, "w", newline="")
escritor = csv.writer(arquivo_csv)
escritor.writerow(["quadro", "tempo_s", "pessoas", "conf",
                   "x1", "y1", "x2", "y2", "hw", "centro_x", "centro_y",
                   "angulo_tronco", "deteccoes"])

regra = RegraQueda(tempo_alerta=args.tempo)
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

    # iou=0.4: junta caixas sobrepostas da mesma pessoa (padrão 0.7 duplicava)
    resultado = modelo.predict(
        frame, conf=CONFIANCA_MINIMA, iou=0.4, verbose=False
    )[0]
    # o modelo de pose só detecta pessoas, não precisa filtrar classe
    pontos_todos = resultado.keypoints.xy.tolist()
    confs_todos = resultado.keypoints.conf.tolist() if len(resultado.boxes) else []

    # desenha retângulo + esqueleto (função pronta do ultralytics)
    frame = resultado.plot(labels=False)
    angulos = []
    for i, caixa in enumerate(resultado.boxes):
        x1, y1, x2, y2 = map(int, caixa.xyxy[0])
        hw = (y2 - y1) / (x2 - x1)
        angulo = angulo_tronco(pontos_todos[i], confs_todos[i])
        angulos.append(angulo)
        texto_ang = f"{angulo:.0f} graus" if angulo is not None else "?"
        deitado = angulo is not None and angulo > ANGULO_DEITADO
        cor = (0, 0, 255) if deitado else (0, 255, 0)
        cv2.putText(frame, f"tronco={texto_ang}  h/w={hw:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    # Por enquanto registra só a pessoa mais confiável do quadro.
    # Quadro sem ninguém também vira linha (com campos vazios): os "buracos"
    # são informação importante para a regra de imobilidade.
    # "deteccoes" = tudo que o YOLO achou; "pessoas" = o que a regra usa
    deteccoes = len(resultado.boxes)
    if deteccoes:
        confs = resultado.boxes.conf.tolist()
        i = max(range(deteccoes), key=lambda k: confs[k])
        pessoas = contar_pessoas(confs, i)
        x1, y1, x2, y2 = map(int, resultado.boxes.xyxy[i])
        angulo = angulos[i]
        hw = (y2 - y1) / (x2 - x1)
        escritor.writerow([
            frames, round(tempo, 3), pessoas,
            round(confs[i], 3), x1, y1, x2, y2,
            round(hw, 3),
            # centro normalizado: 0 = esquerda/topo, 1 = direita/base
            round((x1 + x2) / 2 / largura, 3), round((y1 + y2) / 2 / altura, 3),
            round(angulo, 1) if angulo is not None else "",
            deteccoes,
        ])
    else:
        pessoas, angulo, hw = 0, None, None
        escritor.writerow([frames, round(tempo, 3), 0] + [""] * 9 + [0])

    # Regra de queda: mesma classe usada no simular_regra.py
    if regra.atualizar(tempo, angulo, hw, pessoas) == "alerta":
        print(f"ALERTA DE QUEDA em {tempo:.1f}s "
              f"({time.strftime('%H:%M:%S')})")
    no_chao = regra.tempo_no_chao(tempo)
    if regra.alertou:
        cv2.rectangle(frame, (0, 0), (largura - 1, altura - 1), (0, 0, 255), 12)
        cv2.putText(frame, "ALERTA: QUEDA", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    elif no_chao > 0:
        cv2.putText(frame, f"no chao: {no_chao:.0f}s / {args.tempo:.0f}s",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

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
