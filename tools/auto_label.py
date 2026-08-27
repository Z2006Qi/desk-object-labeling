#!/usr/bin/env python3
"""使用YOLOv8的COCO预训练模型为mouse和keyboard生成初始标注。"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".heic"}
TARGET_CLASS_IDS = {"mouse": 0, "keyboard": 1}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="原始图片目录")
    parser.add_argument("--output", type=Path, required=True, help="预标注输出目录")
    parser.add_argument("--model", default="yolov8x.pt", help="COCO预训练YOLOv8模型")
    parser.add_argument("--conf", type=float, default=0.25, help="最低置信度")
    parser.add_argument("--device", default=None, help="例如cpu或0；不填时由YOLO自动选择")
    return parser.parse_args()


def find_images(directory):
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def xyxy_to_yolo(box, image_width, image_height):
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2.0) / image_width
    center_y = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return center_x, center_y, width, height


def unique_output_name(image_path, source_root):
    relative = image_path.relative_to(source_root)
    if len(relative.parts) == 1:
        return image_path.name
    prefix = "_".join(relative.parts[:-1])
    return f"{prefix}_{image_path.name}"


def main():
    args = parse_args()
    source_root = args.images.resolve()
    output_root = args.output.resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(f"找不到图片目录: {source_root}")
    if not 0.0 < args.conf <= 1.0:
        raise ValueError("--conf必须大于0且不超过1")

    image_paths = find_images(source_root)
    if not image_paths:
        raise ValueError("输入目录中没有找到图片")

    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    (output_root / "classes.txt").write_text("mouse\nkeyboard\n", encoding="utf-8")

    model = YOLO(args.model)
    predict_kwargs = {"conf": args.conf, "verbose": False}
    if args.device is not None:
        predict_kwargs["device"] = args.device

    total_boxes = 0
    empty_images = 0

    for index, image_path in enumerate(image_paths, start=1):
        result = model.predict(source=str(image_path), **predict_kwargs)[0]
        output_name = unique_output_name(image_path, source_root)
        output_image = output_images / output_name
        output_label = output_labels / f"{Path(output_name).stem}.txt"

        if output_image.exists() or output_label.exists():
            raise FileExistsError(f"输出文件名重复: {output_name}")

        shutil.copy2(image_path, output_image)
        label_lines = []

        if result.boxes is not None:
            for box, coco_class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                coco_name = result.names[int(coco_class_id)]
                if coco_name not in TARGET_CLASS_IDS:
                    continue
                center_x, center_y, width, height = xyxy_to_yolo(
                    box, result.orig_shape[1], result.orig_shape[0]
                )
                target_id = TARGET_CLASS_IDS[coco_name]
                label_lines.append(
                    f"{target_id} {center_x:.6f} {center_y:.6f} "
                    f"{width:.6f} {height:.6f}"
                )

        output_label.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        total_boxes += len(label_lines)
        empty_images += not label_lines
        print(f"[{index}/{len(image_paths)}] {image_path.name}: {len(label_lines)}个框")

    print(f"完成：{len(image_paths)}张图片，{total_boxes}个框，{empty_images}张空标注")
    print("请使用LabelImg逐张复核自动标注结果，修正漏检、误检和不准确的检测框。")


if __name__ == "__main__":
    main()

