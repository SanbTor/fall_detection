"""Passa CSVs do detector pela regra de queda, como se fosse ao vivo.

Uso: python src/simular_regra.py data/resultados/*.csv
     python src/simular_regra.py data/resultados/*.csv --tempo 10
"""
import argparse

import pandas as pd

from regra_queda import TEMPO_ALERTA, RegraQueda

parser = argparse.ArgumentParser()
parser.add_argument("csvs", nargs="+")
parser.add_argument("--tempo", type=float, default=TEMPO_ALERTA,
                    help="segundos no chão até alertar")
args = parser.parse_args()


def valor(x):
    """Célula vazia do CSV vira NaN no pandas; a regra espera None."""
    return None if pd.isna(x) else float(x)


print(f"Regra: alerta após {args.tempo:.0f} s no chão\n")
for caminho in args.csvs:
    df = pd.read_csv(caminho)
    regra = RegraQueda(tempo_alerta=args.tempo)
    alertas = []
    maior_tempo = 0
    for linha in df.itertuples():
        if regra.atualizar(linha.tempo_s, valor(linha.angulo_tronco),
                           valor(linha.hw), linha.pessoas) == "alerta":
            alertas.append(linha.tempo_s)
        maior_tempo = max(maior_tempo, regra.tempo_no_chao(linha.tempo_s))

    nome = caminho.split("/")[-1]
    resultado = ", ".join(f"{t:.1f}s" for t in alertas) or "nenhum"
    print(f"{nome:30s} alertas: {resultado:12s} maior tempo no chão: {maior_tempo:.1f}s")
