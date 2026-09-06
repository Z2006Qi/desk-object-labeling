# desk-object-labeling

基于 YOLOv8 的桌面鼠标和键盘目标检测项目。项目完成了数据采集与标注、分阶段训练、Jetson 实时部署、检测结果保存以及 ROS2 消息发布。

## 检测类别

| 类别编号 | 类别名称 |
| ---: | --- |
| 0 | mouse |
| 1 | keyboard |

## 项目成果

- 最终数据集包含 1055 张图片及 1055 个同名 YOLO 标签文件；
- 数据集划分为 738 张训练图片、211 张验证图片和 106 张测试图片；
- 数据集中包含 54 张空标签负样本；
- 最终权重为 `best_1055.pt`；
- Jetson 测试记录包含 375 帧，按逐帧 ROS2 记录计算的平均速度为 28.08 FPS；
- 支持 USB 摄像头、CSI 摄像头、视频输入、检测视频保存、CSV 保存、正常截图、错误截图和 ROS2 发布。

最终权重、数据集、结果视频及测试记录通过 GitHub Release `v1.0.0` 提供。

## 仓库结构

```text
configs/data.example.yaml        数据集配置示例
docs/                            数据采集、标注、训练和测试记录
tools/auto_label.py              COCO 预标注工具
tools/split_dataset.py           train/val/test 数据集划分工具
train_detector.py                YOLOv8 训练程序
jetson_detector.py               Jetson 实时检测与 ROS2 发布程序
RUNNING.md                       单独运行说明
requirements.txt                 Python 依赖
```

## 数据集演进

1. 第一阶段：自行拍摄并标注 209 张图片；
2. 第二阶段：加入网络图片，扩充到 387 张；
3. 第三阶段：针对误检和漏检自行补拍，扩充到 580 张，并加入空标签负样本；
4. 最终阶段：继续整理和复核样本，形成 1055 张最终数据集。

各阶段的详细过程见 `docs/`。

## 快速使用

安装依赖：

```bash
pip install -r requirements.txt
```

使用 USB 摄像头运行检测：

```bash
python3 jetson_detector.py --model best_1055.pt --source 0 --save-video
```

启用 ROS2 发布：

```bash
python3 jetson_detector.py --model best_1055.pt --source 0 --save-video --ros2
```

完整操作步骤、参数和输出说明见 [RUNNING.md](RUNNING.md)。

## 测试材料

`20260831_184622` 测试目录包含：

- `result.mp4`：带检测框和类别信息的结果视频；
- `detections.csv`：逐目标检测记录；
- `ros2_messages_20260831_184616.txt`：ROS2 话题输出记录。

这些材料证明程序能够在 Jetson 上运行、保存检测结果并发布 ROS2 消息。

## Release

- `v1.0.0`：项目首个正式成果版本；
- `best_1055.pt`：最终 YOLOv8 模型权重；
- `dataset_1055.zip`：最终数据集；
- `result.mp4`：最终检测结果视频；
- `detections.csv` 与 ROS2 日志：测试记录。
