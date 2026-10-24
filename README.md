# fall_detection

Sistema de visão para monitorar quedas de idosos com as câmeras Wi-Fi da casa.
Todo o processamento é local: só eventos ("queda", "entrou", "saiu") saem da
casa, nunca vídeo.

## Instalação

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # depois edite o .env com a URL da câmera
```

Os pesos do YOLO são baixados automaticamente para `models/` na primeira execução.

## Como usar

| Comando | O que faz |
| --- | --- |
| `python src/detector_pessoa.py` | Detector ao vivo na câmera do `.env` |
| `python src/detector_pessoa.py data/videos/ARQUIVO.mp4` | Detector num vídeo gravado |
| `python src/detector_pessoa.py ARQUIVO.mp4 --tempo 10` | Idem, alertando após 10 s no chão (padrão: 30 s) |
| `python src/gravar_video.py NOME 60` | Grava 60 s da câmera em `data/videos/` (10 s de contagem antes) |
| `python src/simular_regra.py data/resultados/*.csv` | Passa os CSVs pela regra de queda e mostra quando alertaria |

O detector salva um CSV por quadro (tempo, `h/w`, centro, ângulo do tronco) em
`data/resultados/`, para análise em pandas. Tecla `q` fecha a janela.

## Estrutura

```
src/                 código do sistema
src/experimentos/    testes iniciais (câmera, MOG2)
notebooks/           análises em pandas
docs/                pesquisa e decisões
tests/               testes automatizados
data/videos/         gravações (fora do Git)
data/resultados/     CSVs e gráficos gerados (fora do Git)
models/              pesos do YOLO (fora do Git)
```
