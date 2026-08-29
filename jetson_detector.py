#!/usr/bin/env python3
"""Jetson实时检测：显示结果、保存测试材料并可选发布ROS2消息。

检测窗口快捷键：
    S：保存正常检测截图
    E：保存典型错误截图
    Q或Esc：退出
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("best.pt"))
    parser.add_argument("--source", default="0", help="USB摄像头编号或视频路径")
    parser.add_argument("--csi", action="store_true", help="使用Jetson CSI摄像头")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default=None, help="例如0或cpu；不填时自动选择")
    parser.add_argument("--output", type=Path, default=Path("camera_results"))
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--ros2", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def parse_source(value):
    return int(value) if value.isdigit() else value


def csi_pipeline(width, height, fps):
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! nvvidconv flip-method=0 ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
    )


def open_camera(args):
    if args.csi:
        camera = cv2.VideoCapture(
            csi_pipeline(args.width, args.height, args.fps), cv2.CAP_GSTREAMER
        )
    else:
        camera = cv2.VideoCapture(parse_source(args.source))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        camera.set(cv2.CAP_PROP_FPS, args.fps)
    if not camera.isOpened():
        raise RuntimeError("无法打开摄像头或视频源")
    return camera


def create_ros2_publisher(enabled):
    if not enabled:
        return None, None
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError as error:
        raise RuntimeError("ROS2不可用，请先加载ROS2环境") from error

    rclpy.init()
    node = Node("desk_object_detector")
    publisher = node.create_publisher(String, "/desk_object_detections", 10)
    return node, publisher


def publish_ros2(node, publisher, detections, frame_number, fps):
    if publisher is None:
        return
    import rclpy
    from std_msgs.msg import String

    message = String()
    message.data = json.dumps(
        {
            "frame": frame_number,
            "fps": round(fps, 2),
            "detections": detections,
        },
        ensure_ascii=False,
    )
    publisher.publish(message)
    rclpy.spin_once(node, timeout_sec=0.0)


def extract_detections(result):
    detections = []
    if result.boxes is None:
        return detections

    boxes = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()

    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(value) for value in box]
        detections.append(
            {
                "class": result.names[int(class_id)],
                "confidence": round(float(score), 5),
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )
    return detections


def create_video_writer(path, fps, frame):
    height, width = frame.shape[:2]
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), codec, max(fps, 1.0), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建结果视频: {path}")
    return writer


def save_frame(directory, frame, prefix, frame_number):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_{frame_number:06d}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"无法保存截图: {path}")
    print(f"已保存: {path}")


def main():
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    if not 0.0 < args.conf <= 1.0:
        raise ValueError("--conf必须大于0且不超过1")

    session = args.output.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    csv_path = session / "detections.csv"
    video_path = session / "result.mp4"

    model = YOLO(str(model_path))
    camera = open_camera(args)
    ros_node, ros_publisher = create_ros2_publisher(args.ros2)
    video_writer = None
    frame_number = 0
    total_fps = 0.0

    csv_file = csv_path.open("w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["frame", "class", "confidence", "x1", "y1", "x2", "y2", "fps"]
    )

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("无法读取摄像头画面")
                break

            started = time.perf_counter()
            predict_options = {
                "source": frame,
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "verbose": False,
            }
            if args.device is not None:
                predict_options["device"] = args.device

            result = model.predict(**predict_options)[0]
            display_frame = result.plot()
            elapsed = time.perf_counter() - started
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            frame_number += 1
            total_fps += current_fps

            detections = extract_detections(result)
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox_xyxy"]
                csv_writer.writerow(
                    [
                        frame_number,
                        detection["class"],
                        detection["confidence"],
                        x1,
                        y1,
                        x2,
                        y2,
                        f"{current_fps:.2f}",
                    ]
                )
            csv_file.flush()

            publish_ros2(
                ros_node, ros_publisher, detections, frame_number, current_fps
            )

            cv2.putText(
                display_frame,
                f"FPS: {current_fps:.1f}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if args.save_video:
                if video_writer is None:
                    video_writer = create_video_writer(
                        video_path, camera.get(cv2.CAP_PROP_FPS), display_frame
                    )
                video_writer.write(display_frame)

            if not args.no_display:
                cv2.imshow("Mouse and Keyboard Detection", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    save_frame(
                        session / "screenshots", display_frame, "result", frame_number
                    )
                elif key == ord("e"):
                    save_frame(
                        session / "errors", display_frame, "error", frame_number
                    )
                if key in (ord("q"), 27):
                    break
    finally:
        csv_file.close()
        camera.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        if ros_node is not None:
            ros_node.destroy_node()
            import rclpy

            rclpy.shutdown()

    average_fps = total_fps / frame_number if frame_number else 0.0
    print(f"平均FPS: {average_fps:.2f}")
    print(f"检测记录: {csv_path}")
    if args.save_video:
        print(f"结果视频: {video_path}")


if __name__ == "__main__":
    main()
