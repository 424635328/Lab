# AI教学实验箱：数字积木排序机械臂项目

## 1. 项目概述

本项目旨在利用人工智能教学实验箱，完整复现一个从零开始的AI应用开发流程。项目将通过摄像头识别“仓库一”中的数字积木（0-9），利用自行训练的深度学习模型进行识别，然后根据数字大小进行降序排序，并最终控制机械臂将积木按排序结果搬运至“仓库二”。

**核心特性:**
*   **端到端实现**: 涵盖数据采集、模型训练、模型优化、应用部署全流程。
*   **环境适应性**: 软件算法需考虑并处理真实环境中的光照、位置、角度等干扰因素。
*   **平台特定**: 教程针对 **Ubuntu 24.04 LTS PC** (用于训练) 和 **实验箱内置Linux系统** (用于部署) 进行了优化。

## 2. 技术栈

*   **PC平台**: Ubuntu 24.04 LTS
*   **实验箱平台**: 内置Linux系统 (ARMv8/aarch64)
*   **核心语言**: Python 3, C++11
*   **深度学习框架**: PaddlePaddle
*   **模型部署框架**: PaddleLite
*   **应用开发框架**: Qt 5
*   **计算机视觉库**: OpenCV

## 3. Phase 0: 环境搭建

在开始项目前，请分别配置好您的PC和实验箱环境。

### 3.1. PC环境搭建 (Ubuntu 24.04 LTS)

此环境用于数据处理、模型训练与转换。

#### 3.1.1. 安装系统及开发工具

在PC的终端中执行以下命令：

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装核心工具
sudo apt install -y build-essential git python3-pip python3-venv wget curl
```

#### 3.1.2. 创建Python虚拟环境

```bash
# 创建项目根目录并进入
mkdir ~/ai_arm_project
cd ~/ai_arm_project

# 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate
```
> **提示**: 之后所有Python相关的命令行操作，都应在激活的`(venv)`环境下进行。

#### 3.1.3. 安装Python依赖库

```bash
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install opencv-python numpy
```

#### 3.1.4. 配置PaddleLite `opt`工具

```bash
# 创建工具目录
mkdir -p ~/tools
cd ~/tools

# 下载独立的opt工具 (适用于x86 PC)
wget https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.12.0/opt_linux_x86

# 赋予权限并简化名称
chmod +x opt_linux_x86
mv opt_linux_x86 opt

# 将工具路径永久添加到系统PATH
echo 'export PATH=$PATH:~/tools' >> ~/.bashrc
source ~/.bashrc

# 验证安装
opt
# 如果输出帮助信息，则表示成功
```

### 3.2. 实验箱环境搭建

此环境用于运行最终的应用程序。

#### 3.2.1. 安装Qt及OpenCV

在实验箱的终端中执行：

```bash
sudo apt update
sudo apt install -y qtcreator qt5-default libopencv-dev
```

#### 3.2.2. 准备PaddleLite C++预测库

1.  在PC上，从[PaddleLite Release页面](https://github.com/PaddlePaddle/Paddle-Lite/releases)下载适用于**ARMv8 (aarch64)** 的C++预测库。文件通常命名为 `inference_lite_lib.linux.aarch64.*.tar.gz`。
2.  将下载的压缩包传输到实验箱，例如通过`scp`命令。
3.  在实验箱上解压该文件到指定目录，例如 `/home/linux/paddlelite_cpp/`。

---

## Phase 1: 数据集构建

### 1.1. 在实验箱上采集图像

1.  在实验箱上创建一个文件 `collect_data.py`，并复制以下内容：

    ```python
    # collect_data.py
    import cv2
    import os
    import time

    SAVE_ROOT = "/home/linux/Desktop/digital_images"
    CAMERA_INDEX = 0

    os.makedirs(SAVE_ROOT, exist_ok=True)
    for i in range(10):
        os.makedirs(os.path.join(SAVE_ROOT, str(i)), exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("错误: 无法打开摄像头!")
        exit()

    print("摄像头已启动。按数字'0'-'9'保存，按'q'退出。")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 0-9 to save. Press 'q' to quit.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Data Collection', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        
        if ord('0') <= key <= ord('9'):
            digit = chr(key)
            save_dir = os.path.join(SAVE_ROOT, digit)
            filename = f"{digit}_{int(time.time() * 1000)}.jpg"
            save_path = os.path.join(save_dir, filename)
            
            cv2.imwrite(save_path, frame)
            print(f"已保存 -> {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    ```

2.  在实验箱终端运行 `python3 collect_data.py`，并按提示进行多样化采集。

### 1.2. 数据传输与划分 (在PC上)

1.  **传输数据**: 获取实验箱IP后，在PC终端执行`scp`命令将`digital_images`文件夹拷贝到PC的`~/ai_arm_project/`下，并重命名为`digital_raw_data`。

2.  **划分数据**: 在PC的`~/ai_arm_project/`目录下创建 `split_dataset.py` 文件，内容如下：

    ```python
    # split_dataset.py
    import os
    import random
    import shutil

    SOURCE_DIR = "digital_raw_data"
    DEST_DIR = "data_split"
    SPLIT_RATIO = (0.8, 0.1, 0.1)

    if os.path.exists(DEST_DIR): shutil.rmtree(DEST_DIR)

    for sub_dir in ['train', 'val', 'test']:
        for i in range(10):
            os.makedirs(os.path.join(DEST_DIR, sub_dir, str(i)), exist_ok=True)

    for digit in os.listdir(SOURCE_DIR):
        digit_path = os.path.join(SOURCE_DIR, digit)
        if not os.path.isdir(digit_path): continue
        
        images = [f for f in os.listdir(digit_path) if f.endswith('.jpg')]
        random.shuffle(images)
        
        train_count = int(len(images) * SPLIT_RATIO[0])
        val_count = int(len(images) * SPLIT_RATIO[1])
        
        for i, img_name in enumerate(images):
            source_img_path = os.path.join(digit_path, img_name)
            if i < train_count: dest_folder = 'train'
            elif i < train_count + val_count: dest_folder = 'val'
            else: dest_folder = 'test'
            dest_img_path = os.path.join(DEST_DIR, dest_folder, digit, img_name)
            shutil.copy(source_img_path, dest_img_path)

    print(f"数据集划分完成，保存在 '{DEST_DIR}' 目录。")
    ```

3.  执行脚本: `python3 split_dataset.py`

---

## Phase 2: 模型训练与转换 (在PC上)

### 2.1. 训练模型

在`~/ai_arm_project/`目录下创建`train.py`文件，内容如下：

```python
# train.py
import paddle
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms

# 1. 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 2. 加载数据集
train_dataset = DatasetFolder('data_split/train', transform=transform)
val_dataset = DatasetFolder('data_split/val', transform=transform)

# 3. 定义LeNet模型
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.flatten = paddle.nn.Flatten()
        self.linear1 = paddle.nn.Linear(in_features=16*7*7, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=num_classes)
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# 4. 训练配置
model = paddle.Model(LeNet())
model.prepare(
    paddle.optimizer.Adam(parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy()
)

# 5. 开始训练
print("开始训练...")
model.fit(train_dataset, val_dataset, epochs=10, batch_size=64, verbose=1)

# 6. 保存模型用于部署
print("训练完成，保存推理模型...")
paddle.jit.save(model.network, './inference_model/lenet',
                input_spec=[paddle.static.InputSpec(shape=[None, 1, 32, 32], dtype='float32')])
print("推理模型已保存到 'inference_model' 目录。")
```

执行训练: `python3 train.py`

### 2.2. 转换模型

```bash
# 执行模型转换
opt --model_file=./inference_model/lenet.pdmodel \
    --param_file=./inference_model/lenet.pdiparams \
    --optimize_out_type=naive_buffer \
    --optimize_out=./lite_model \
    --valid_targets=arm
```

**关键产出**: `lite_model/lenet.nb`。

---

## Phase 3: 应用开发与部署 (在实验箱上)

### 3.1. 准备工作

1.  **传输模型**: 将PC上的`lite_model/lenet.nb`文件通过`scp`传输到实验箱的`~/`主目录。
2.  **创建项目**: 在实验箱上打开`Qt Creator`，创建一个名为`ArmSortApp`的Qt Widgets Application项目。
3.  **移动模型**: 将`lenet.nb`文件移动到新创建的`ArmSortApp`项目文件夹内。

### 3.2. 配置Qt项目

打开`ArmSortApp.pro`文件，用以下内容替换，**注意修改`PADDLE_LITE_CPP_DIR`为你自己的路径**。

```qmake
# ArmSortApp.pro
QT       += core gui serialport
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG   += c++11

TARGET = ArmSortApp
TEMPLATE = app

SOURCES += main.cpp\
        mainwindow.cpp
HEADERS  += mainwindow.h
FORMS    += mainwindow.ui

# 使用pkg-config自动链接OpenCV
CONFIG += link_pkgconfig
PKGCONFIG += opencv4

# 配置PaddleLite C++预测库
# !!! 修改为你在实验箱上解压的实际路径 !!!
PADDLE_LITE_CPP_DIR = /home/linux/paddlelite_cpp/inference_lite_lib.linux.aarch64

INCLUDEPATH += $$PADDLE_LITE_CPP_DIR/include
LIBS += -L$$PADDLE_LITE_CPP_DIR/lib -lpaddle_light_api_shared

# 自动将.so文件拷贝到编译输出目录，方便运行
PADDLE_LITE_SO = $$PADDLE_LITE_CPP_DIR/lib/libpaddle_light_api_shared.so
DESTDIR_SO = $$OUT_PWD
QMAKE_POST_LINK += $$QMAKE_COPY $$quote($$PADDLE_LITE_SO) $$quote($$DESTDIR_SO) $$escape_expand(\\n\\t)
```

### 3.3. 编写C++核心代码

请参考之前提供的C++代码框架和逻辑片段，在`Qt Creator`中完成`mainwindow.h`和`mainwindow.cpp`的编写，实现模型加载、图像捕获与处理、模型预测、结果排序和串口通信等功能。

### 3.4. 编译与运行

1.  在`Qt Creator`中，配置好构建套件，然后点击**构建**按钮。
2.  构建成功后，点击**运行**按钮。
3.  在实验箱上摆好积木，点击程序界面上的“开始排序”按钮，观察并调试。

---
