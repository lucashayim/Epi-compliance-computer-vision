"""Treino do modelo YOLOv8 a partir de um dataset anotado (formato YOLO).

Estrutura esperada (NÃO versionada):
  data/images/*.jpg
  data/labels/*.txt   (+ classes.txt com um nome de classe por linha)

Executar:  python -m src.train
"""
import os
import random
import shutil
import sys

from ultralytics import YOLO

DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
OUT_DIR = os.path.join(DATA_DIR, "dataset_split")
PROJECT = "runs_epi"
VAL_RATIO = 0.2
EPOCHS = 50


def load_classes():
    for candidate in (os.path.join(LABELS_DIR, "classes.txt"), os.path.join(DATA_DIR, "classes.txt")):
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
    sys.exit("classes.txt não encontrado em data/labels/ ou data/.")


def build_split(classes):
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    for part in ("train", "val"):
        os.makedirs(f"{OUT_DIR}/{part}/images", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/{part}/labels", exist_ok=True)

    images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        sys.exit("Nenhuma imagem encontrada em data/images/.")
    random.shuffle(images)
    cut = int(len(images) * (1 - VAL_RATIO))

    for part, subset in (("train", images[:cut]), ("val", images[cut:])):
        for name in subset:
            shutil.copy(os.path.join(IMAGES_DIR, name), f"{OUT_DIR}/{part}/images/{name}")
            label = os.path.splitext(name)[0] + ".txt"
            src = os.path.join(LABELS_DIR, label)
            if os.path.exists(src):
                shutil.copy(src, f"{OUT_DIR}/{part}/labels/{label}")

    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(OUT_DIR)}\ntrain: train/images\nval: val/images\n")
        f.write(f"nc: {len(classes)}\nnames: {classes}\n")
    return yaml_path


if __name__ == "__main__":
    classes = load_classes()
    print(f"Classes: {classes}")
    data_yaml = build_split(classes)
    model = YOLO("yolov8n.pt")  # pesos pré-treinados oficiais (baixados automaticamente)
    model.train(data=data_yaml, epochs=EPOCHS, imgsz=640, project=PROJECT, name="run")
