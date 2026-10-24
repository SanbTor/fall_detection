# Lições de projetos similares: guia de implementação para o fall_detection

Pesquisa feita em 2026-10-24 sobre projetos e estudos de detecção de quedas por câmera. Use este arquivo como contexto para as próximas etapas do projeto `fall_detection`. Ele complementa o `CONTEXTO.md` / `CLAUDE.md` já existente (objetivo, hardware, código atual).

> Como trabalhar comigo: a cada solução, explique brevemente o conceito envolvido (o que, por que, conceito). Português do Brasil.

## Resumo das decisões

1. **Detectar o estado "pessoa no chão há X segundos", não só o movimento da queda.** Em estudo com quedas reais de idosos, 86% ficaram muito tempo no chão. O estado dura minutos; o evento dura < 1 s e pode ser perdido se o Wi-Fi descartar quadros.
2. **MOG2 + razão h/w não serve como detector principal.** O mesmo método, em 7 casas reais, gerou ~179 falsos alarmes por dia e pegou só 62% das quedas. MOG2 pode, no máximo, ser um "porteiro" que acorda o YOLO.
3. **Usar YOLO-pose (`yolov8n-pose` ou mais novo, via `ultralytics`)** em vez de só o retângulo. Os keypoints dão a orientação do corpo; uma queda na direção da câmera quase não muda o h/w, mas muda o ângulo do tronco.
4. **Zonas permitidas (cama, sofá) e supressão com 2+ pessoas** para cortar falsos positivos.
5. **Monitorar o próprio sistema (heartbeat)**: se o script ou a câmera caírem, alguém precisa saber.
6. **Modo silencioso na casa por 1 a 2 semanas** antes de ligar alertas, registrando o que teria disparado.

## Lógica de detecção proposta

Features por pessoa, por quadro (a partir dos 17 keypoints COCO do YOLO-pose):
- Ângulo do tronco: vetor ombro-médio → quadril-médio em relação à vertical. Em pé ≈ 0°, deitado ≈ 90°.
- Ângulo cabeça-joelho-pé (projeto Raspberry Pi usou limite padrão de 45, dependente do ponto de vista da câmera).
- Altura do quadril/centro do tronco no quadro, e se está dentro de uma "zona de chão" desenhada.
- Velocidade vertical do centro do tronco (para detectar descida brusca).
- Razão h/w do retângulo (secundária).
- Confiança média dos keypoints (descartar detecções com confiança baixa, ex. < 0,5).

Regra inicial (regras ponderadas antes de treinar ML; um artigo de 2025 achou regras mais confiáveis que ML com poucos dados):

```
se pessoa_detectada e corpo_horizontal e centro_do_tronco_na_zona_do_chao:
    tempo_no_chao += dt
senão:
    tempo_no_chao = 0

se tempo_no_chao > 30 s e não está em zona_permitida (cama/sofá) e n_pessoas == 1:
    alerta (prioridade maior se houve descida brusca nos segundos anteriores)
```

- A descida brusca é reforço, não condição obrigatória.
- Tornar limiares configuráveis (ângulo, segundos, confiança) num arquivo de config, porque dependem da posição da câmera.
- Só treinar classificador (scikit-learn sobre as features) depois de ter vídeos próprios rotulados.

## Falsos positivos conhecidos

| Fonte | Mitigação |
| --- | --- |
| Deitar na cama/sofá | Polígonos de zona permitida; ignorar estado caído dentro deles |
| Ajoelhar, agachar, pegar algo no chão | Tempo mínimo no chão (ex. 30 s) |
| Mais de uma pessoa (cuidadora, visita) | Suspender alerta com 2+ pessoas |
| Luz mudando, IR à noite, lâmpada acendendo | Testar o modelo com imagem IR preto e branco das Intelbras; medir confiança à noite |
| "Fantasmas" do MOG2 (pessoa parada vira fundo) | Nunca decidir com MOG2; só usar para acordar o YOLO |
| Pessoa escondida por móvel após cair (67% dos casos reais) | Câmera alta, em canto, vendo o chão livre; aceitar torso parcial |

## Câmeras e RTSP (Intelbras linha iM)

- URL já confirmada: `rtsp://USUARIO:SENHA@IP:554/cam/realmonitor?channel=1&subtype=1` (subtype=1 para detecção, subtype=0 para clipes). Credenciais só no `.env`.
- **Reconexão automática**: no loop, se `cap.read()` devolver `ret == False` (ou não chegar quadro por N segundos), liberar o `VideoCapture`, esperar com backoff e reabrir, sem derrubar o processo.
- Manter `OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"`.
- Câmeras baratas aceitam poucas conexões simultâneas. Se precisar de visualização ao vivo junto com o detector, usar um restream (go2rtc, como faz o Frigate).
- Depois de estável, desligar atualização automática de firmware no app Mibo (relatos de firmware quebrando integrações).

## Banheiro e privacidade

- Não filmar dentro do banheiro. Manter o plano: câmera no quarto vendo a porta + timer de permanência.
- Camadas opcionais sem imagem: sensor magnético de porta; radar mmWave (Aqara FP2 tem modo queda e resiste a umidade, mas vai no teto, perde zonas/multi-pessoa no modo queda e não é dispositivo médico; C1001 com ESP32 é opção DIY ~US$ 35, alcance de queda ~3 m).
- Só eventos saem da casa, nunca vídeo.

## Alertas e confiabilidade

- Telegram com botão "Estou vendo"; sem resposta em X minutos, ligação (Twilio).
- Não depender de ação da idosa (relatos de pingente não acionado em queda real).
- **Heartbeat**: o processo faz ping a cada minuto num serviço como Healthchecks.io; se parar, o serviço avisa (dead man's switch). Incluir saúde de cada câmera (segundos desde o último quadro).
- Rodar como serviço `systemd` com reinício automático.

## Testes

- Datasets públicos para sensibilidade: UR Fall Detection, Le2i. Não medem falso alarme.
- Vídeos próprios em `data/` (fora do Git): quedas laterais, de frente, atrás de cadeira, de dia e com IR. Criar um script que roda o detector sobre uma pasta de vídeos e gera um relatório (alertas por vídeo), para comparar versões do código.
- Modo silencioso na casa: logar "alerta que teria enviado" com horário e snapshot local por 1 a 2 semanas; medir falsos alarmes por dia.

## Ordem de implementação sugerida

1. `src/detector_pessoa.py` com `yolov8n-pose`: desenhar keypoints, ângulo do tronco e h/w na tela; mostrar FPS.
2. Reconexão automática do RTSP.
3. Estado "no chão" com timer (regra acima) e config de limiares.
4. Ferramenta para desenhar polígonos: zonas permitidas e zona da porta do banheiro.
5. Script de avaliação offline sobre vídeos gravados.
6. Medir FPS no N100 com PyTorch e com export OpenVINO (`model.export(format="openvino")`); 3 a 5 fps bastam.
7. Alertas Telegram + heartbeat.
8. Instalação na casa em modo silencioso; depois ligar alertas e ligação.

## Fontes

- Debard et al., Camera-based fall detection using real-world versus simulated data (KU Leuven, 2016): https://homes.esat.kuleuven.be/~konijn/publications/2016/fallcam.pdf
- Vision-Based Fallen Person Detection for the Elderly: https://ar5iv.arxiv.org/html/1707.07608
- Pose-Based Fall Detection System (2025): https://arxiv.org/html/2503.19501v1
- Raspberry Pi AI Camera fall detection: https://www.hackster.io/540813/using-raspberry-pi-ai-camera-for-fall-detection-a772cb
- falldetection_openpifpaf: https://github.com/nbourre/falldetection_openpifpaf
- Action_Recognition_Surveillance (YOLO + AlphaPose + ST-GCN): https://github.com/guoqiang0148666/action_recognition_surveillance
- Fórum Home Assistant, Elderly person fall alarm: https://community.home-assistant.io/t/elderly-person-fall-alarm/659804
- Aqara FP2 review: https://staceyoniot.com/?p=12268
- Intelbras RTSP/ONVIF: https://www.smartrtsp.com/pt-br/guides/compatibilidade-intelbras-nvr-onvif
- Frigate restream (go2rtc): https://docs.frigate.video/configuration/restream
- OpenVINO YOLOv8 keypoints: https://docs.openvino.ai/2024/notebooks/yolov8-keypoint-detection-with-output.html
- Healthchecks.io: https://healthchecks.io/faq
