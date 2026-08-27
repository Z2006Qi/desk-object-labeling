# 使用LabelImg进行第一轮标注

## 安装和启动

安装LabelImg：

```powershell
pip install labelImg
```

启动程序：

```powershell
labelImg
```

## 配置标注目录

1. 点击 `Open Dir`，选择需要标注的图片目录。
2. 点击 `Change Save Dir`，选择标签保存目录。
3. 将保存格式切换为 `YOLO`。
4. 类别顺序固定为：

```text
mouse
keyboard
```

对应的YOLO类别编号为：

```text
0: mouse
1: keyboard
```

类别顺序确定后不能交换，否则已有标签中的0和1会被解释成错误类别。

## 标注规则

- 使用矩形框紧密包围可见的鼠标或键盘；
- 不在检测框中包含过多桌面背景；
- 一张图片中出现多个目标时分别绘制检测框；
- 鼠标和键盘同时出现时分别标注对应类别；
- 完成自动预标注后仍逐张人工复核，修正误检、漏检和不准确的框。

YOLO标签中的每个目标占一行：

```text
class_id x_center y_center width height
```

图片和标签文件必须同名，例如：

```text
images/mouse_001.jpg
labels/mouse_001.txt
```

## 第一轮标注结果

第一轮209张图片完成复核后，将图片和标签统一整理到 `labeled_final` 对应的 `images` 与 `labels` 目录，再使用 `split_dataset.py` 划分训练集、验证集和测试集。

