"""Configuração central. Todos os segredos vêm de variáveis de ambiente (.env)."""
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL", "")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.15"))
ALERT_DELAY_SECONDS = int(os.getenv("ALERT_DELAY_SECONDS", "45"))

# Zonas de auditoria (pixels, frame redimensionado para FRAME_SIZE).
# Ajuste conforme o enquadramento da sua câmera.
FRAME_SIZE = (1020, 600)
ZONE_AUDIT = np.array([[460, 0], [1020, 0], [1020, 600], [520, 600]], np.int32)
ZONE_MAT = np.array([[490, 330], [760, 330], [760, 600], [490, 600]], np.int32)   # tapete sanitizante
ZONE_SINK = np.array([[760, 200], [950, 200], [950, 600], [760, 600]], np.int32)  # pia

# Tempo mínimo (s) lavando as mãos para considerar higienização ok
HANDWASH_MIN_SECONDS = 5
# Tempo (s) com a pessoa em cena antes de disparar alerta
ALERT_AFTER_SECONDS = 3.0
# Memória curta (s) das detecções para evitar "piscadas"
DETECTION_MEMORY_SECONDS = 2.0
