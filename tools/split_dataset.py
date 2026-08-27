#!/usr/bin/env python3
"""将 mouse/keyboard 的 YOLO 标注数据划分为 train、val 和 test。"""

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".heic"}
CLASS_NAMES = ["mouse", "keyboard"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="labeled_final 目录")
    parser.add_argument("--output", type=Path, required=True, help="输出数据集目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--ratio",
        type=float,
        nargs=3,
        default=(0.7, 0.2, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    return parser.parse_args()


def find_input_directories(input_dir):
    image_dir = input_dir / "images"
    label_dir = input_dir / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("输入目录必须包含 images 和 labels 两个子目录")
    return image_dir, label_dir


def validate_label(label_path):
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number} 不是YOLO五字段格式")
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        if class_id not in (0, 1):
            raise ValueError(f"{label_path}:{line_number} 类别只能是0或1")
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{label_path}:{line_number} 坐标必须位于0到1之间")


def collect_samples(image_dir, label_dir):
    samples = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(
                f"{image_path.name} 没有同名标签；负样本也需要同名空txt文件"
            )
        validate_label(label_path)
        samples.append((image_path, label_path))
    if not samples:
        raise ValueError("输入目录中没有找到图片")
    return samples


def split_samples(samples, ratios, seed):
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train、val、test比例必须为非负数且总和等于1")
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * ratios[0])
    val_end = train_end + int(len(shuffled) * ratios[1])
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def prepare_output(output_dir):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录不是空目录，停止以避免覆盖: {output_dir}")
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_samples(output_dir, grouped_samples):
    for split, samples in grouped_samples.items():
        for image_path, label_path in samples:
            shutil.copy2(image_path, output_dir / "images" / split / image_path.name)
            shutil.copy2(label_path, output_dir / "labels" / split / label_path.name)


def write_data_yaml(output_dir):
    yaml_text = "\n".join(
        [
            f"path: {output_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "nc: 2",
            "names:",
            "  0: mouse",
            "  1: keyboard",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    args = parse_args()
    image_dir, label_dir = find_input_directories(args.input.resolve())
    samples = collect_samples(image_dir, label_dir)
    grouped = split_samples(samples, args.ratio, args.seed)
    prepare_output(args.output.resolve())
    copy_samples(args.output.resolve(), grouped)
    write_data_yaml(args.output.resolve())

    background_count = sum(label.stat().st_size == 0 for _, label in samples)
    print(f"总图片数: {len(samples)}，其中背景负样本: {background_count}")
    for split, split_samples_list in grouped.items():
        print(f"{split}: {len(split_samples_list)}")
    print(f"data.yaml: {args.output.resolve() / 'data.yaml'}")


if __name__ == "__main__":
    main()

