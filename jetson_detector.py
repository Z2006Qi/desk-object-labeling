#!/usr/bin/env python3
"""Jetson实时检测：显示检测框、类别、置信度和FPS。"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("best.pt"))
    parser.add_argument("--source", type=int, default=0, help="USB摄像头编号")
    parser.add_argument("--conf", type=float, default=0.25, help="最低置信度")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"找不到模型文件: {args.model}")

    model = YOLO(str(args.model))
    camera = cv2.VideoCapture(args.source)
    if not camera.isOpened():
        raise RuntimeError(f"无法打开摄像头: {args.source}")

    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("无法读取摄像头画面")
                break

            result = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )[0]
            display_frame = result.plot()

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Mouse and Keyboard Detection", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

