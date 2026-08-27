#!/usr/bin/env python3
"""第一版训练：从YOLOv8n开始训练mouse和keyboard检测模型。"""

from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "dataset_final" / "data.yaml"
CLASS_NAMES = ["mouse", "keyboard"]


def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"找不到数据集配置文件: {DATA_FILE}")

    with DATA_FILE.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=int)]

    if names != CLASS_NAMES:
        raise ValueError(
            "类别顺序必须是0=mouse、1=keyboard，"
            f"当前data.yaml为: {names}"
        )

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_FILE),
        epochs=100,
        imgsz=640,
        batch=8,
        project=str(PROJECT_DIR / "training_runs"),
        name="train1_mouse_keyboard",
    )


if __name__ == "__main__":
    main()

