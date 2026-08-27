#!/usr/bin/env python3
"""第三阶段最终训练：使用387张版本的best.pt在580张数据集上继续训练。"""

from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "dataset_final" / "data.yaml"
BASE_MODEL = PROJECT_DIR / "training_runs" / "retrain_387" / "weights" / "best.pt"
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

    if not BASE_MODEL.is_file():
        raise FileNotFoundError(f"找不到第二轮模型: {BASE_MODEL}")

    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(DATA_FILE),
        epochs=100,
        imgsz=640,
        batch=8,
        project=str(PROJECT_DIR / "training_runs"),
        name="retrain_580",
    )


if __name__ == "__main__":
    main()

