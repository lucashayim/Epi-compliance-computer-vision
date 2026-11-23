# EPI Compliance — Visão Computacional para Conformidade de EPIs

Sistema de **visão computacional** que usa uma câmera IP para verificar, em tempo real, se o colaborador está corretamente equipado antes de entrar em uma área produtiva (ex.: indústria de alimentos).

> **Status:** funcional e em evolução. Existem variáveis de campo (iluminação, ângulo, oclusão, variedade de EPIs) que ainda estão sendo refinadas para aumentar a precisão.

## O que ele faz
- Detecta **pessoa, touca e bota** com **YOLOv8** (Ultralytics).
- Valida a **passagem pelo tapete sanitizante** e o **tempo de higienização das mãos** na pia, usando zonas de interesse (polígonos) sobre a imagem.
- Mostra um painel em tempo real: `EPI: OK / SEM TOUCA / SEM BOTA`, `TAPETE`, `PIA`.
- Envia **alerta com foto** via Telegram (opcional), com intervalo mínimo entre alertas para evitar spam.

## Arquitetura
```
Câmera IP (RTSP) ──► leitor de frames (thread, só o frame mais recente)
                         │
                         ▼
                 YOLOv8 (detecção) ──► regras de negócio por zona (OpenCV)
                                            │
                          painel em tempo real  +  alerta Telegram (opcional)
```
Decisões técnicas:
- **RTSP sobre TCP** para estabilidade em VPN/Wi-Fi.
- **Leitura em thread separada** mantendo só o último frame, evitando acúmulo de atraso.
- **Memória curta de detecções (2 s)** para suavizar oscilações do modelo.
- **Zonas configuráveis** em `src/config.py`.

## Estrutura
```
src/
  config.py          # configuração (lê variáveis de ambiente do .env)
  monitor.py         # monitoramento em tempo real
  train.py           # treino do modelo a partir de dataset anotado (formato YOLO)
  collect_frames.py  # coleta de frames da câmera para montar o dataset
.env.example         # modelo de configuração (sem segredos)
```

## Como executar
> É necessário um modelo YOLO treinado com as classes `pessoa`, `touca` e `bota` (veja "Treinando o seu próprio modelo"). Os pesos do projeto original não são públicos.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env     # preencha RTSP_URL (0 = webcam), MODEL_PATH e (opcional) Telegram
python -m src.monitor    # pressione Q para sair
```

### Treinando o seu próprio modelo
1. Colete imagens (`python -m src.collect_frames`) ou use um dataset público de EPIs.
2. Anote no formato YOLO (ex.: LabelImg, Roboflow) e organize em `data/images` e `data/labels` com um `classes.txt`.
3. Rode `python -m src.train`. O melhor modelo fica em `runs_epi/run/weights/best.pt`.

Classes usadas neste projeto: `pessoa`, `touca`, `sem_touca`, `bota`, `celular`.

## Segurança e privacidade
- **Nenhuma credencial no código**: tudo vem de variáveis de ambiente (`.env` fora do Git).
- **Dataset e pesos treinados não são versionados** (contêm imagens de pessoas/ambiente real — LGPD).

## Tecnologias
Python · YOLOv8 (Ultralytics) · OpenCV · NumPy · Requests · RTSP · Telegram Bot API

## Próximos passos
- Ampliar o dataset (turnos, iluminação, EPIs adicionais: avental, luvas, máscara).
- Avaliar métricas por classe e reduzir falsos positivos.
- Registrar eventos em banco de dados e gerar dashboard de conformidade.
