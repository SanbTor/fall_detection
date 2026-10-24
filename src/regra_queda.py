"""Regra de queda: detecta o ESTADO "pessoa no chão há X segundos".

Não depende de câmera nem de YOLO: recebe os números de cada quadro
(tempo, ângulo do tronco, h/w, nº de pessoas), um por vez. Assim a mesma
regra roda ao vivo e sobre os CSVs gravados.
"""

# Limites provisórios, medidos nos vídeos de teste (câmera na mesa).
# Dependem da posição da câmera: depois vão para um arquivo de config.
ANGULO_DEITADO = 60       # tronco > 60° da vertical = deitado
HW_DEITADO = 1.0          # sem ângulo, usa h/w < 1 como pista secundária
TEMPO_ALERTA = 20         # segundos no chão até alertar
TOLERANCIA = 3            # segundos sem ver "deitado" antes de zerar o timer


def esta_deitado(angulo, hw):
    """True/False se dá para dizer; None se não há informação."""
    if angulo is not None:
        return angulo > ANGULO_DEITADO
    if hw is not None:
        return hw < HW_DEITADO
    return None


class RegraQueda:
    def __init__(self, tempo_alerta=TEMPO_ALERTA, tolerancia=TOLERANCIA):
        self.tempo_alerta = tempo_alerta
        self.tolerancia = tolerancia
        self.inicio_chao = None       # quando começou a ficar deitado
        self.ultimo_deitado = None    # último quadro em que estava deitado
        self.alertou = False          # evita repetir o alerta a cada quadro

    def atualizar(self, tempo, angulo, hw, pessoas):
        """Processa um quadro. Devolve "alerta" uma única vez por episódio."""
        deitado = esta_deitado(angulo, hw) if pessoas > 0 else None

        if deitado:
            if self.inicio_chao is None:
                self.inicio_chao = tempo
            self.ultimo_deitado = tempo
        elif self.ultimo_deitado is not None:
            # Em pé ou sumiu por pouco tempo: pode ser só um quadro ruim
            # do YOLO. Só zera se passou da tolerância.
            if tempo - self.ultimo_deitado > self.tolerancia:
                self.zerar()

        if self.inicio_chao is None or self.alertou:
            return None
        # Com 2+ pessoas, alguém já está junto (cuidadora, visita)
        if pessoas >= 2:
            return None
        if tempo - self.inicio_chao >= self.tempo_alerta:
            self.alertou = True
            return "alerta"
        return None

    def tempo_no_chao(self, tempo):
        return 0 if self.inicio_chao is None else tempo - self.inicio_chao

    def zerar(self):
        self.inicio_chao = None
        self.ultimo_deitado = None
        self.alertou = False
