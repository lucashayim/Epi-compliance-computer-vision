"""Monitoramento de conformidade de EPIs em tempo real (YOLOv8 + câmera IP).

Regras verificadas na área de entrada:
  - Touca presente
  - Bota presente (e passagem pelo tapete sanitizante)
  - Tempo de higienização das mãos na pia

Executar:  python -m src.monitor
"""
import os
import sys
import threading
import time

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from src import config as cfg

# Força TCP no RTSP (mais estável em VPN/Wi-Fi)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

GREEN, RED, YELLOW, GRAY, ORANGE, DARK = (0, 255, 0), (0, 0, 255), (0, 255, 255), (160, 160, 160), (0, 140, 255), (30, 30, 30)
WINDOW = "EPI Compliance - Computer Vision"


class LatestFrameReader:
    """Lê o stream em thread própria mantendo só o frame mais recente (evita atraso)."""

    def __init__(self, src):
        # "0", "1"... = webcam local; caso contrário, URL RTSP
        self.stream = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            _, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


def in_zone(box, polygon):
    """Verifica se o ponto inferior-central da caixa está dentro da zona."""
    x1, y1, x2, y2 = box
    return cv2.pointPolygonTest(polygon, (int((x1 + x2) / 2), int(y2)), False) >= 0


def send_telegram_async(image, message):
    """Envia alerta com foto ao Telegram (desativado se não houver token)."""
    if not (cfg.TELEGRAM_TOKEN and cfg.TELEGRAM_CHAT_ID):
        return

    def _send():
        try:
            ok, buf = cv2.imencode(".jpg", image)
            if not ok:
                return
            requests.post(
                f"https://api.telegram.org/bot{cfg.TELEGRAM_TOKEN}/sendPhoto",
                data={"chat_id": cfg.TELEGRAM_CHAT_ID, "caption": message},
                files={"photo": ("alert.jpg", buf.tobytes())},
                timeout=10,
            )
        except requests.RequestException as exc:
            print(f"[telegram] falha ao enviar alerta: {type(exc).__name__}")

    threading.Thread(target=_send, daemon=True).start()


def run():
    if not cfg.RTSP_URL:
        sys.exit("Defina RTSP_URL no arquivo .env (use 0 para a webcam).")
    if not os.path.exists(cfg.MODEL_PATH):
        sys.exit(f"Modelo não encontrado em {cfg.MODEL_PATH}. Treine com src/train.py.")

    model = YOLO(cfg.MODEL_PATH)
    cam = LatestFrameReader(cfg.RTSP_URL).start()
    time.sleep(1.0)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    seen = {"person": 0.0, "boot": 0.0, "cap": 0.0, "boot_on_mat": 0.0}
    person_since = 0.0
    sink_since = 0.0
    last_alert = 0.0
    wash_text, wash_color = "PIA: ---", GRAY
    mem = cfg.DETECTION_MEMORY_SECONDS

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.1)
            continue

        view = cv2.resize(frame, cfg.FRAME_SIZE)
        person_at_sink = False

        for result in model(view, stream=True, conf=cfg.CONFIDENCE, imgsz=640):
            view = cv2.addWeighted(view, 0.4, result.plot(), 0.6, 0)
            for box in result.boxes:
                name = model.names[int(box.cls[0])]
                coords = box.xyxy[0].tolist()
                if not in_zone(coords, cfg.ZONE_AUDIT):
                    continue
                now = time.time()
                if name == "pessoa":
                    seen["person"] = now
                    person_at_sink = person_at_sink or in_zone(coords, cfg.ZONE_SINK)
                elif name == "bota":
                    seen["boot"] = now
                    if in_zone(coords, cfg.ZONE_MAT):
                        seen["boot_on_mat"] = now
                elif name == "touca":
                    seen["cap"] = now

        now = time.time()
        has_person = now - seen["person"] < mem
        has_boot = now - seen["boot"] < mem
        has_cap = now - seen["cap"] < mem
        mat_ok = now - seen["boot_on_mat"] < mem

        epi_text, epi_color = "EPI: AGUARDANDO...", GRAY
        mat_text, mat_color = "TAPETE: ---", GRAY

        if has_person:
            person_since = person_since or now
            elapsed = now - person_since
            mat_text, mat_color = ("TAPETE: OK", GREEN) if mat_ok else ("TAPETE: PENDENTE", YELLOW)
            can_alert = elapsed > cfg.ALERT_AFTER_SECONDS and now - last_alert > cfg.ALERT_DELAY_SECONDS

            if not has_cap:
                epi_text, epi_color = "EPI: SEM TOUCA!", RED
                if can_alert:
                    send_telegram_async(view, "ALERTA: colaborador sem touca na entrada.")
                    last_alert = now
            elif not has_boot:
                epi_text, epi_color = "EPI: SEM BOTA!", ORANGE
                if can_alert:
                    send_telegram_async(view, "ATENÇÃO: colaborador sem botas na entrada.")
                    last_alert = now
            else:
                epi_text, epi_color = "EPI: OK", GREEN
        else:
            person_since = 0.0

        if person_at_sink:
            sink_since = sink_since or now
            washed = now - sink_since
            if washed > cfg.HANDWASH_MIN_SECONDS:
                wash_text, wash_color = "PIA: OK (LAVADO)", GREEN
            else:
                wash_text, wash_color = f"PIA: LAVANDO {int(washed)}s", YELLOW
        else:
            sink_since = 0.0
            if not has_person:
                wash_text, wash_color = "PIA: ---", GRAY
            elif "OK" not in wash_text:
                wash_text, wash_color = "PIA: AGUARDANDO", GRAY

        cv2.rectangle(view, (0, 0), (cfg.FRAME_SIZE[0], 80), DARK, -1)
        cv2.putText(view, epi_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, epi_color, 2)
        cv2.putText(view, mat_text, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mat_color, 2)
        cv2.putText(view, wash_text, (750, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, wash_color, 2)
        cv2.imshow(WINDOW, view)

        if cv2.waitKey(1) == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
