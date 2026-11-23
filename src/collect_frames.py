"""Coleta de frames da câmera para montar o dataset. 'S' salva um frame, 'Q' sai.

Atenção (LGPD): ao fotografar pessoas, tenha autorização e não publique as imagens.
"""
import os
from datetime import datetime

import cv2

from src import config as cfg

OUT_DIR = "data/images"


def run():
    if not cfg.RTSP_URL:
        raise SystemExit("Defina RTSP_URL no arquivo .env (veja .env.example).")
    os.makedirs(OUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(cfg.RTSP_URL)
    count = 0
    print("S = salvar frame | Q = sair")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Falha na leitura do frame.")
            break
        cv2.imshow("Coleta de frames (S salva, Q sai)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("s"), ord("S")):
            name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{count}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, name), frame)
            print(f"Salvo: {name}")
            count += 1
        elif key in (ord("q"), ord("Q")):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
