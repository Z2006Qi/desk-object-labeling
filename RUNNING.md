# 桌面物体检测系统运行说明

## 1. 文件准备

运行前请确保以下文件位于同一项目目录：

```text
jetson_detector.py
best_1055.pt
requirements.txt
```

最终模型可从 GitHub Release `v1.0.0` 下载。

## 2. 环境要求

- Python 3；
- Ultralytics YOLOv8；
- OpenCV；
- 如需发布检测结果，Jetson 还需安装并加载 ROS2 环境。

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

使用 ROS2 前，先按 Jetson 上实际安装的 ROS2 版本加载环境，例如：

```bash
source /opt/ros/<ros2-distro>/setup.bash
```

其中 `<ros2-distro>` 应替换为设备实际安装的版本名称。

## 3. USB 摄像头检测

```bash
python3 jetson_detector.py \
  --model best_1055.pt \
  --source 0
```

如果摄像头编号不是 0，可把 `--source 0` 改为其他编号。

## 4. CSI 摄像头检测

```bash
python3 jetson_detector.py \
  --model best_1055.pt \
  --csi
```

## 5. 视频文件检测

```bash
python3 jetson_detector.py \
  --model best_1055.pt \
  --source input.mp4
```

## 6. 保存结果视频

在运行命令中加入 `--save-video`：

```bash
python3 jetson_detector.py \
  --model best_1055.pt \
  --source 0 \
  --save-video
```

程序会在 `camera_results/日期_时间/` 中保存：

- `detections.csv`：检测记录；
- `result.mp4`：带检测框、类别、置信度和 FPS 的结果视频。

## 7. ROS2 发布

在已加载 ROS2 环境的终端中运行：

```bash
python3 jetson_detector.py \
  --model best_1055.pt \
  --source 0 \
  --save-video \
  --ros2
```

检测结果发布到：

```text
/desk_object_detections
```

可在另一个已加载 ROS2 环境的终端中查看消息：

```bash
ros2 topic echo /desk_object_detections
```

每条消息包含帧号、FPS 和检测结果。每个检测结果包含类别、置信度和边界框坐标。

## 8. 运行快捷键

| 按键 | 功能 |
| --- | --- |
| `S` | 保存正常检测截图至 `screenshots/` |
| `E` | 保存典型错误截图至 `errors/` |
| `Q` 或 `Esc` | 退出程序 |

## 9. 常用参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--model` | `best.pt` | 模型权重路径 |
| `--source` | `0` | USB 摄像头编号或视频路径 |
| `--csi` | 关闭 | 使用 CSI 摄像头 |
| `--width` | 1280 | 输入宽度 |
| `--height` | 720 | 输入高度 |
| `--fps` | 30 | 请求的摄像头帧率 |
| `--imgsz` | 640 | YOLO 推理尺寸 |
| `--conf` | 0.25 | 置信度阈值 |
| `--iou` | 0.50 | IoU 阈值 |
| `--device` | 自动 | 例如 `0` 或 `cpu` |
| `--output` | `camera_results` | 输出目录 |
| `--save-video` | 关闭 | 保存结果视频 |
| `--ros2` | 关闭 | 发布 ROS2 消息 |
| `--no-display` | 关闭 | 不显示检测窗口 |

## 10. 常见问题

- 提示找不到模型：检查 `--model` 指向的文件是否存在；
- 无法打开摄像头：检查摄像头编号、连接状态及访问权限；
- ROS2 不可用：确认 ROS2 已安装并已执行环境加载命令；
- 无显示环境：加入 `--no-display`，并根据需要使用 `--save-video` 保存结果。

