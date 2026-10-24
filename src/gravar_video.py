import os
import sys
import time

import cv2

from camera import Camera

# Uso: python src/gravar_video.py nome_do_teste 30
nome = sys.argv[1] if len(sys.argv) > 1 else "teste"
DURACAO = int(sys.argv[2]) if len(sys.argv) > 2 else 30  # segundos
ESPERA = 10  # segundos para você se posicionar antes de começar

camera = Camera()  # câmera do .env, com reconexão automática
frame = camera.ler()
altura, largura = frame.shape[:2]
fps = camera.fps
if not 1 <= fps <= 60:  # algumas câmeras informam 0 ou um valor absurdo
    fps = 15

os.makedirs("data/videos", exist_ok=True)
caminho = f"data/videos/{nome}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
gravador = cv2.VideoWriter(
    caminho, cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura)
)

# Durante a contagem, continua lendo e descartando quadros: se o stream
# ficar parado, acumula quadros atrasados e a gravação começaria "no passado"
for i in range(ESPERA, 0, -1):
    print(f"Começa em {i}...")
    fim = time.time() + 1
    while time.time() < fim:
        camera.ler()
print(f"Gravando {DURACAO}s em {caminho}")

inicio = time.time()
while time.time() - inicio < DURACAO:
    gravador.write(camera.ler())

gravador.release()
camera.fechar()
print("Pronto!")
