"""Abre a câmera (RTSP) ou um arquivo de vídeo, com reconexão automática.

Uso:
    camera = Camera(fonte)       # fonte = arquivo ou None (usa o .env)
    while (frame := camera.ler()) is not None:
        ...
    camera.fechar()

No arquivo, ler() devolve None quando o vídeo acaba. Na câmera, ler() nunca
desiste: se o stream cair, fecha, espera e tenta de novo (backoff).
"""
import os
import time

# Usa TCP no RTSP: mais estável que UDP em Wi-Fi
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
from dotenv import load_dotenv

TIMEOUT_MS = 5000      # sem isso, um read() na câmera desligada trava ~30 s
ESPERA_INICIAL = 1     # segundos antes da 1ª nova tentativa
ESPERA_MAXIMA = 30     # teto do backoff


def url_da_camera():
    load_dotenv()
    url = os.getenv("CAMERA_URL")
    if not url:
        raise SystemExit("CAMERA_URL não encontrada no .env")
    return url


class Camera:
    def __init__(self, fonte=None):
        self.url = fonte or url_da_camera()
        self.eh_arquivo = os.path.isfile(self.url)
        if not self.eh_arquivo and not self.url.startswith("rtsp://"):
            raise SystemExit(f"Arquivo não encontrado: {self.url}")
        self.cap = None
        self.espera = ESPERA_INICIAL
        self.ultimo_quadro = time.time()
        self.quedas = 0            # quantas vezes o stream caiu
        if not self._abrir():
            if self.eh_arquivo:
                raise SystemExit(f"Não consegui abrir o vídeo: {self.url}")
            print("Câmera indisponível; vou continuar tentando...")

    def _abrir(self):
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_MS,
        ])
        return self.cap.isOpened()

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0

    def segundos_sem_quadro(self):
        """Saúde da câmera: usado depois no heartbeat."""
        return time.time() - self.ultimo_quadro

    def ler(self):
        """Devolve o próximo quadro, ou None se o arquivo acabou."""
        while True:
            if self.cap is not None and self.cap.isOpened():
                ok, frame = self.cap.read()
                if ok:
                    if self.espera > ESPERA_INICIAL:
                        print("Câmera de volta.")
                    self.espera = ESPERA_INICIAL
                    self.ultimo_quadro = time.time()
                    return frame
                if self.eh_arquivo:
                    return None  # fim do vídeo
                self.quedas += 1

            # Câmera caiu (ou não abriu): fecha, espera e tenta de novo.
            # A espera dobra a cada falha (1, 2, 4... até 30 s) para não
            # martelar uma câmera que está reiniciando.
            if self.cap is not None:
                self.cap.release()
            print(f"Sem imagem há {self.segundos_sem_quadro():.0f}s; "
                  f"tentando de novo em {self.espera}s...")
            time.sleep(self.espera)
            self.espera = min(self.espera * 2, ESPERA_MAXIMA)
            self._abrir()

    def fechar(self):
        if self.cap is not None:
            self.cap.release()
