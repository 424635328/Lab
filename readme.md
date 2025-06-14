# 百度AI教学实验箱：数字积木排序机械臂项目

## 1. 项目概述

本项目旨在利用人工智能教学实验箱，完整复现一个从零开始的AI应用开发流程。项目将通过摄像头识别“仓库一”中的数字积木（0-9），利用自行训练的深度学习模型进行识别，然后根据数字大小进行降序排序，并最终控制机械臂将积木按排序结果搬运至“仓库二”。

**核心特性:**

* **端到端实现**: 涵盖数据采集、模型训练、模型优化、应用部署全流程。
* **环境适应性**: 软件算法需考虑并处理真实环境中的光照、位置、角度等干扰因素。
* **平台特定**: 教程针对 **Ubuntu 24.04 LTS PC** (用于训练) 和 **实验箱内置Linux系统** (用于部署) 进行了优化。

## 2. 技术栈

* **PC平台**: Ubuntu 24.04 LTS
* **实验箱平台**: 内置Linux系统 (ARMv8/aarch64)
* **核心语言**: Python 3, C++11
* **深度学习框架**: PaddlePaddle
* **模型部署框架**: PaddleLite
* **应用开发框架**: Qt 5
* **计算机视觉库**: OpenCV

## 3. Phase 0: 环境搭建

在开始项目前，请分别配置好您的PC和实验箱环境。

### 3.1. PC环境搭建 (Ubuntu 24.04 LTS)

此环境用于数据处理、模型训练与转换。

#### 3.1.1. 安装系统及开发工具

在PC的终端中执行以下命令：

```bash
# 更新系统，确保所有软件包都是最新的
sudo apt update && sudo apt upgrade -y

# 安装编译和开发所需的核心工具
sudo apt install -y build-essential git python3-pip python3-venv wget curl
```

#### 3.1.2. 配置Python环境 (使用pyenv)

> **说明**: Ubuntu 24.04默认的Python 3.12与当前PaddlePaddle不兼容。我们将使用`pyenv`安装并管理Python 3.11，这是解决版本冲突的最佳实践。

```bash
# 1. 安装pyenv及其编译依赖
sudo apt install -y make libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl
curl https://pyenv.run | bash

# 2. 配置shell以加载pyenv（使其在每次打开终端时都可用）
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec "$SHELL" # 重新加载shell以使配置生效

# 3. 安装Python 3.11.9
pyenv install 3.11.9
```

#### 3.1.3. 创建项目并配置虚拟环境

```bash
# 1. 创建项目目录并进入
mkdir ~/ai_arm_project
cd ~/ai_arm_project

# 2. 为此项目目录指定使用的Python版本
pyenv local 3.11.9

# 3. 创建并激活Python虚拟环境
python -m venv venv
source venv/bin/activate
```

> **提示**: 之后所有Python相关的命令行操作，都应在激活的`(venv)`环境下进行。您的命令行提示符前会显示`(venv)`字样。

#### 3.1.4. 安装Python依赖库

```bash
# 在激活的(venv)环境中执行
# 升级pip自身
pip install --upgrade pip setuptools wheel
# 安装PaddlePaddle和OpenCV，使用清华镜像源加速
pip install paddlepaddle==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python numpy
```

#### 3.1.5. 配置PaddleLite `opt`工具

`opt`工具用于将训练好的模型转换为适合嵌入式设备运行的轻量级格式。

```bash
# 创建一个专门存放工具的目录
mkdir -p ~/tools
cd ~/tools

# 下载适用于x86 PC的独立opt工具
wget https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.12.0/opt_linux_x86

# 赋予可执行权限并简化名称
chmod +x opt_linux_x86
mv opt_linux_x86 opt

# 将工具路径永久添加到系统PATH中
echo 'export PATH=$PATH:~/tools' >> ~/.bashrc
source ~/.bashrc

# 验证安装是否成功
opt
# 如果屏幕输出了opt工具的帮助信息，则表示成功
```

### 3.2. 实验箱环境搭建

此环境用于运行最终的应用程序。

#### 3.2.1. 安装Qt及OpenCV

在**实验箱的终端**中执行（通过`Ctrl+Alt+T`打开）：

```bash
# 更新软件源
sudo apt update
# 安装Qt Creator, Qt5默认库, 以及OpenCV开发库
sudo apt install -y qtcreator qt5-default libopencv-dev
```

#### 3.2.2. 准备PaddleLite C++预测库

1. 在**PC**上，从[PaddleLite Release页面](https://github.com/PaddlePaddle/Paddle-Lite/releases)下载适用于**ARMv8 (aarch64)** 的C++预测库。文件通常命名为 `inference_lite_lib.linux.aarch64.*.tar.gz`。

2. **将下载的压缩包传输到实验箱**。首先在实验箱终端用`ifconfig`查看IP地址。假设实验箱IP为`192.168.1.101`，用户名为`linux`，在**PC终端**的下载目录中执行：

    ```bash
    # 语法: scp <本地文件路径> <用户名>@<IP地址>:<远程目标路径>
    # 将下载的库文件复制到实验箱用户的家目录下
    scp ./inference_lite_lib.linux.aarch64.gcc.tar.gz linux@192.168.1.101:~/
    ```

3. 在**实验箱终端**上解压该文件到指定目录：

    ```bash
    # 创建一个存放库文件的目录
    mkdir -p ~/paddlelite_cpp
    # 解压到刚刚创建的目录中
    tar -zxvf ~/inference_lite_lib.linux.aarch64.gcc.tar.gz -C ~/paddlelite_cpp
    ```

    最终，C++预测库的路径应为 `/home/linux/paddlelite_cpp/inference_lite_lib.linux.aarch64/`。

---

## Phase 1: 数据集构建

### 1.1. 在实验箱上采集图像

1. 在实验箱上创建一个文件 `collect_data.py`，并复制以下内容：

    ```python
    # 文件名: collect_data.py
    # 功能: 通过摄像头采集数字积木图像，并按数字分类保存
    import cv2
    import os
    import time

    # 保存图像的根目录，这里设置为桌面上的一个文件夹
    SAVE_ROOT = "/home/linux/Desktop/digital_images"
    # 摄像头设备索引，0通常代表默认的USB摄像头
    CAMERA_INDEX = 0

    # 确保根目录存在，并为0-9每个数字创建子目录
    os.makedirs(SAVE_ROOT, exist_ok=True)
    for i in range(10):
        os.makedirs(os.path.join(SAVE_ROOT, str(i)), exist_ok=True)

    # 初始化摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("错误: 无法打开摄像头!")
        exit()

    print("摄像头已启动。")
    print("操作提示: 将数字积木置于镜头前，按下对应的数字键('0'-'9')进行拍照保存。")
    print("按 'q' 键退出程序。")

    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        if not ret: 
            print("错误: 无法读取视频帧。")
            break
        
        # 创建一个用于显示的副本，并在上面添加提示文字
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 0-9 to save. Press 'q' to quit.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Data Collection', display_frame)

        # 等待按键，1毫秒超时
        key = cv2.waitKey(1) & 0xFF
        
        # 如果按下 'q'，则退出循环
        if key == ord('q'):
            break
        
        # 如果按下的是数字键
        if ord('0') <= key <= ord('9'):
            digit = chr(key)
            save_dir = os.path.join(SAVE_ROOT, digit)
            # 使用时间戳生成唯一文件名，避免覆盖
            filename = f"{digit}_{int(time.time() * 1000)}.jpg"
            save_path = os.path.join(save_dir, filename)
            
            # 保存图像
            cv2.imwrite(save_path, frame)
            print(f"图像已保存 -> {save_path}")

    # 释放摄像头资源并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
    ```

2. 在实验箱终端运行 `python3 collect_data.py`。根据提示，在不同光照、角度、位置下采集每个数字的图像，确保数据集的多样性，每个数字建议采集50-100张。

### 1.2. 数据传输与划分 (在PC上)

1. **传输数据**: 采集完成后，在**PC终端**执行`scp`命令，将整个图像文件夹下载到PC。

    ```bash
    # 回到PC的项目目录
    cd ~/ai_arm_project
    
    # 将实验箱上采集的整个文件夹递归(-r)拷贝过来，并重命名为digital_raw_data
    scp -r linux@192.168.1.101:~/Desktop/digital_images ./digital_raw_data
    ```

2. **划分数据**: 在PC的`~/ai_arm_project/`目录下创建 `split_dataset.py` 文件，用于将数据集划分为训练集、验证集和测试集。

    ```python
    # 文件名: split_dataset.py
    # 功能: 将原始数据集按比例划分为训练集、验证集和测试集
    import os
    import random
    import shutil

    # 源数据目录
    SOURCE_DIR = "digital_raw_data"
    # 目标目录
    DEST_DIR = "data_split"
    # 划分比例 (训练集, 验证集, 测试集)
    SPLIT_RATIO = (0.8, 0.1, 0.1)

    # 如果目标目录已存在，先删除，确保每次运行都是全新的划分
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)

    # 创建目标目录结构
    for sub_dir in ['train', 'val', 'test']:
        for i in range(10):
            os.makedirs(os.path.join(DEST_DIR, sub_dir, str(i)), exist_ok=True)

    # 遍历每个数字类别
    for digit in os.listdir(SOURCE_DIR):
        digit_path = os.path.join(SOURCE_DIR, digit)
        if not os.path.isdir(digit_path):
            continue
        
        # 获取该数字下的所有图片
        images = [f for f in os.listdir(digit_path) if f.endswith('.jpg')]
        random.shuffle(images) # 随机打乱顺序
        
        # 计算划分数量
        train_count = int(len(images) * SPLIT_RATIO[0])
        val_count = int(len(images) * SPLIT_RATIO[1])
        
        # 遍历图片并复制到对应目录
        for i, img_name in enumerate(images):
            source_img_path = os.path.join(digit_path, img_name)
            if i < train_count:
                dest_folder = 'train'
            elif i < train_count + val_count:
                dest_folder = 'val'
            else:
                dest_folder = 'test'
            dest_img_path = os.path.join(DEST_DIR, dest_folder, digit, img_name)
            shutil.copy(source_img_path, dest_img_path)

    print(f"数据集划分完成，保存在 '{DEST_DIR}' 目录。")
    ```

3. 执行脚本: `python3 split_dataset.py`

---

## Phase 2: 模型训练与转换 (在PC上)

### 2.1. 训练模型

在`~/ai_arm_project/`目录下创建`train.py`文件。我们将使用经典的LeNet模型，它结构简单、效果好，非常适合此任务。

```python
# 文件名: train.py
# 功能: 使用划分好的数据集训练一个LeNet数字分类模型
import paddle
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms

# 1. 定义数据预处理流程
transform = transforms.Compose([
    transforms.Resize((32, 32)),    # 统一图像大小为32x32
    transforms.Grayscale(),         # 转换为灰度图
    transforms.ToTensor(),          # 转换为Tensor格式
    transforms.Normalize(mean=[0.5], std=[0.5]) # 归一化到[-1, 1]范围
])

# 2. 加载数据集
train_dataset = DatasetFolder('data_split/train', transform=transform)
val_dataset = DatasetFolder('data_split/val', transform=transform)

# 3. 定义LeNet模型结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 第一个卷积-池化层
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        # 第二个卷积-池化层
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        # 展平层，用于连接全连接层
        self.flatten = paddle.nn.Flatten()
        # 三个全连接层
        self.linear1 = paddle.nn.Linear(in_features=16*6*6, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=num_classes)
        # 激活函数
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

# 4. 实例化模型，并配置优化器、损失函数和评估指标
model = paddle.Model(LeNet())
model.prepare(
    paddle.optimizer.Adam(parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy()
)

# 5. 开始训练
print("开始训练...")
model.fit(train_dataset, val_dataset, epochs=10, batch_size=64, verbose=1)

# 6. 保存模型用于部署（保存为静态图格式）
print("训练完成，保存推理模型...")
paddle.jit.save(model.network, './inference_model/lenet',
                input_spec=[paddle.static.InputSpec(shape=[None, 1, 32, 32], dtype='float32')])
print("推理模型已保存到 'inference_model' 目录。")
```

执行训练: `python3 train.py`

### 2.2. 转换模型

使用之前配置好的`opt`工具，将训练好的模型转换为`.nb`格式。

在`~/ai_arm_project/`目录下执行：

```bash
opt --model_file=./inference_model/lenet.pdmodel \
    --param_file=./inference_model/lenet.pdiparams \
    --optimize_out_type=naive_buffer \
    --optimize_out=./lite_model \
    --valid_targets=arm
```

**关键产出**: `lite_model/lenet.nb`。这个文件就是我们要在实验箱上使用的最终模型。

---

## Phase 3: 应用开发与部署 (在实验箱上)

### 3.1. 准备工作

1. **创建项目**: 在实验箱上打开`Qt Creator`，新建一个`Qt Widgets Application`项目，命名为`ArmSortApp`。
2. **传输模型文件**: 在**PC终端**执行，将最终模型传输到实验箱上刚创建的项目文件夹内。

    ```bash
    # 语法: scp <本地文件> <用户名>@<IP>:<远程项目路径>
    scp ~/ai_arm_project/lite_model/lenet.nb linux@192.168.1.101:~/ArmSortApp/
    ```

### 3.2. 配置Qt项目 (`.pro`文件)

双击Qt Creator左侧项目树中的`ArmSortApp.pro`文件，用以下内容**完全替换**原有内容。
**请务必根据您自己的实际情况修改 `PADDLE_LITE_CPP_DIR` 的路径！**

```qmake
# 文件名: ArmSortApp.pro
# 功能: 配置项目的依赖库和编译选项

# 引入Qt模块：核心、图形界面、串口
QT       += core gui serialport
# 对于Qt5及以上版本，引入widgets模块
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

# 启用C++11标准
CONFIG   += c++11
# 在macOS上不创建app包
CONFIG   -= app_bundle

# 禁用一些不推荐使用的警告
DEFINES += QT_DEPRECATED_WARNINGS

# 项目源文件
SOURCES += \
    main.cpp \
    mainwindow.cpp
# 项目头文件
HEADERS  += \
    mainwindow.h
# UI设计文件
FORMS    += \
    mainwindow.ui

# 使用pkg-config工具自动链接OpenCV库
CONFIG += link_pkgconfig
PKGCONFIG += opencv4 # 假设系统安装的是OpenCV 4.x

# 配置PaddleLite C++预测库
# !!! 关键：请修改为你在实验箱上解压PaddleLite库的实际路径 !!!
PADDLE_LITE_CPP_DIR = /home/linux/paddlelite_cpp/inference_lite_lib.linux.aarch64

# 添加头文件搜索路径
INCLUDEPATH += $$PADDLE_LITE_CPP_DIR/include
# 添加库文件链接路径和要链接的库
LIBS += -L$$PADDLE_LITE_CPP_DIR/lib -lpaddle_light_api_shared

# 自动将依赖的动态库(libpaddle_light_api_shared.so)拷贝到编译输出目录
# 这样运行时程序才能找到它
PADDLE_LITE_SO = $$PADDLE_LITE_CPP_DIR/lib/libpaddle_light_api_shared.so
DESTDIR_SO = $$OUT_PWD
COPY_CMD = $$QMAKE_COPY $$quote($$PADDLE_LITE_SO) $$quote($$DESTDIR_SO)
QMAKE_POST_LINK += $$COPY_CMD $$escape_expand(\\n\\t)
```

> **重要**: 修改完 `.pro` 文件后，在Qt Creator中右键点击项目名称，选择 `Run qmake`，让配置生效。

### 3.3. 设计UI界面 (`mainwindow.ui`)

1. 双击`mainwindow.ui`文件，进入**设计模式**。
2. 从左侧的**控件盒子 (Widget Box)** 中拖拽以下控件到主窗口上：
    * 一个 `QLabel`，用于显示摄像头视频。将其`objectName`属性修改为 `videoLabel`。
    * 一个 `QPushButton`，用于启动任务。将其`text`属性修改为 `开始排序`，`objectName`属性修改为 `startButton`。
    * 一个 `QLabel`，用于显示状态信息。将其`text`属性留空，`objectName`属性修改为 `statusLabel`。
3. 使用顶部的布局工具对控件进行美观的排列。
4. 按下 `Ctrl + S` 保存UI设计。

### 3.4. 编写C++核心代码

#### `main.cpp` (保持默认即可)

```cpp
#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
```

#### `mainwindow.h`

用以下内容**完全替换** `mainwindow.h`：

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QSerialPort>
#include <vector>
#include <memory>
#include <algorithm>

// OpenCV 头文件
#include <opencv2/opencv.hpp>

// PaddleLite C++ API 头文件
#include "paddle_api.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

// 用于保存识别结果的数据结构
struct DetectionResult {
    int digit;                // 识别出的数字
    int original_pos_index;   // 在仓库一中的原始位置索引 (0-3)

    // 重载小于操作符，方便后续使用std::sort进行排序
    bool operator<(const DetectionResult& other) const {
        return this->digit < other.digit;
    }
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_startButton_clicked(); // “开始排序”按钮的槽函数
    void update_camera_frame();    // 定时器触发，用于更新摄像头视频帧

private:
    // 初始化系列函数
    void initCamera();
    void initModel();
    void initSerialPort();
    void initUI();

    // 核心功能函数
    std::vector<DetectionResult> processFrame(const cv::Mat& frame); // 处理单帧图像，返回所有识别结果
    int predict(const cv::Mat& roi); // 对单个ROI进行预测，返回数字
    void sendArmCommand(int from_pos, int to_pos); // 发送机械臂移动指令

    Ui::MainWindow *ui;

    // Qt组件
    QTimer *camera_timer;       // 摄像头定时器
    QSerialPort *serial_port;   // 串口通信对象

    // OpenCV组件
    cv::VideoCapture cap;       // 摄像头捕获对象

    // PaddleLite组件
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor; // AI预测器

    // 业务逻辑变量
    bool is_task_running = false;          // 任务是否正在运行的标志
    std::vector<cv::Rect> warehouse1_rois; // 存储仓库一4个格子的ROI（感兴趣区域）
};
#endif // MAINWINDOW_H
```

#### `mainwindow.cpp`

用以下内容**完全替换** `mainwindow.cpp`：

```cpp
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QApplication>
#include <QThread>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 依次执行初始化
    initUI();
    initModel();
    initCamera();
    initSerialPort();
}

MainWindow::~MainWindow()
{
    // 清理资源
    if (camera_timer && camera_timer->isActive()) {
        camera_timer->stop();
    }
    if (cap.isOpened()) {
        cap.release();
    }
    delete ui;
}

void MainWindow::initUI()
{
    this->setWindowTitle("AI机械臂数字排序系统");
    ui->statusLabel->setText("系统准备就绪。");

    // !!! 关键：根据你的摄像头和布局，手动标定这4个ROI矩形框 !!!
    // cv::Rect(矩形左上角x坐标, 矩形左上角y坐标, 宽度, 高度)
    // 这个步骤需要您运行程序后，对着摄像头画面反复调整，直到框体精确
    warehouse1_rois.push_back(cv::Rect(45, 110, 80, 80));    // 仓库一：左上角格子
    warehouse1_rois.push_back(cv::Rect(145, 110, 80, 80));   // 仓库一：右上角格子
    warehouse1_rois.push_back(cv::Rect(45, 210, 80, 80));    // 仓库一：左下角格子
    warehouse1_rois.push_back(cv::Rect(145, 210, 80, 80));   // 仓库一：右下角格子
}

void MainWindow::initModel()
{
    // 模型文件应与编译生成的可执行文件在同一目录
    QString model_path = QApplication::applicationDirPath() + "/lenet.nb";
    
    ui->statusLabel->setText("正在加载AI模型...");
    QApplication::processEvents(); // 刷新UI以显示加载信息

    paddle::lite_api::MobileConfig config;
    config.set_model_from_file(model_path.toStdString());

    try {
        predictor = paddle::lite_api::CreatePaddlePredictor(config);
        qDebug() << "PaddleLite模型已从以下路径加载:" << model_path;
        ui->statusLabel->setText("AI模型加载成功。");
    } catch (const std::exception& e) {
        qCritical() << "模型加载失败: " << e.what();
        ui->statusLabel->setText("错误：AI模型加载失败！");
        ui->startButton->setEnabled(false); // 禁用开始按钮
    }
}

void MainWindow::initCamera()
{
    cap.open(0); // 0代表系统默认摄像头
    if (!cap.isOpened()) {
        qCritical() << "无法打开摄像头!";
        ui->statusLabel->setText("错误：无法打开摄像头！");
        ui->startButton->setEnabled(false);
        return;
    }

    // 创建定时器，用于周期性地获取并显示视频帧
    camera_timer = new QTimer(this);
    connect(camera_timer, &QTimer::timeout, this, &MainWindow::update_camera_frame);
    camera_timer->start(40); // 每40毫秒触发一次，约等于25 FPS
}

void MainWindow::initSerialPort()
{
    serial_port = new QSerialPort(this);
    // !!! 关键：修改为你的实验箱机械臂的实际串口设备名 !!!
    // 可以通过 `ls /dev/ttyS*` 或 `ls /dev/ttyUSB*` 查找
    serial_port->setPortName("/dev/ttyS1"); 
    serial_port->setBaudRate(QSerialPort::Baud115200); // 波特率，需与下位机一致
    serial_port->setDataBits(QSerialPort::Data8);
    serial_port->setParity(QSerialPort::NoParity);
    serial_port->setStopBits(QSerialPort::OneStop);
    serial_port->setFlowControl(QSerialPort::NoFlowControl);
}

void MainWindow::update_camera_frame()
{
    if (!cap.isOpened()) return;

    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) return;

    // 在视频帧上绘制ROI框，方便调试时观察位置是否正确
    if (!is_task_running) {
        for (const auto& roi : warehouse1_rois) {
            cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2); // 绘制绿色矩形
        }
    }

    // 将OpenCV的Mat格式(BGR)转换为Qt的QImage格式(RGB)以在QLabel中显示
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    QImage qimg(frame.data, frame.cols, frame.rows, static_cast<int>(frame.step), QImage::Format_RGB888);
    ui->videoLabel->setPixmap(QPixmap::fromImage(qimg).scaled(ui->videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::on_startButton_clicked()
{
    if (is_task_running) return; // 防止重复点击
    
    is_task_running = true;
    ui->startButton->setEnabled(false);
    ui->statusLabel->setText("任务开始：正在识别积木...");
    QApplication::processEvents();

    cv::Mat current_frame;
    cap >> current_frame; // 捕获当前帧用于识别
    if (current_frame.empty()) {
        ui->statusLabel->setText("错误：无法捕获图像！");
        is_task_running = false;
        ui->startButton->setEnabled(true);
        return;
    }

    // 处理图像，获取识别结果
    std::vector<DetectionResult> results = processFrame(current_frame);

    if (results.empty()) {
        ui->statusLabel->setText("任务完成：未识别到任何积木。");
    } else {
        // 使用C++ STL的sort，配合重载的<操作符进行降序排序
        std::sort(results.rbegin(), results.rend());

        QString log = "识别并排序完成: ";
        for(const auto& res : results) {
            log += QString::number(res.digit) + " ";
        }
        ui->statusLabel->setText(log);
        QApplication::processEvents();
        QThread::sleep(2); // 显示结果2秒，让用户看清楚

        // 遍历排序后的结果，依次发送搬运指令
        for (size_t i = 0; i < results.size(); ++i) {
            int from = results[i].original_pos_index; // 积木的原始位置
            int to = i;                               // 积木的目标位置（按降序）
            
            ui->statusLabel->setText(QString("正在搬运积木 %1: 从位置 %2 到 %3...")
                                     .arg(results[i].digit).arg(from + 1).arg(to + 1));
            QApplication::processEvents();
            
            sendArmCommand(from, to);
            // 简化处理：假设机械臂完成一次抓取-放置动作需要5秒
            // 在实际项目中，应通过串口接收下位机的完成信号
            QThread::sleep(5); 
        }
        ui->statusLabel->setText("所有积木搬运完毕！任务全部完成！");
    }
    
    is_task_running = false;
    ui->startButton->setEnabled(true);
}

std::vector<DetectionResult> MainWindow::processFrame(const cv::Mat& frame) {
    std::vector<DetectionResult> detections;
    for (size_t i = 0; i < warehouse1_rois.size(); ++i) {
        // 确保ROI在图像范围内
        if(warehouse1_rois[i].x + warehouse1_rois[i].width < frame.cols &&
           warehouse1_rois[i].y + warehouse1_rois[i].height < frame.rows)
        {
            cv::Mat roi_img = frame(warehouse1_rois[i]); // 提取ROI
            int digit = predict(roi_img);
            if (digit != -1) { // -1代表未识别或置信度低
                detections.push_back({digit, static_cast<int>(i)});
            }
        }
    }
    return detections;
}

int MainWindow::predict(const cv::Mat& roi) {
    if (!predictor || roi.empty()) return -1;

    // 图像预处理，与训练时保持一致
    cv::Mat resized, gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(32, 32));

    // 获取输入张量并设置其形状
    std::unique_ptr<paddle::lite_api::Tensor> input_tensor(predictor->GetInput(0));
    input_tensor->Resize({1, 1, 32, 32});
    auto* input_data = input_tensor->mutable_data<float>();
    
    // 填充输入数据并进行归一化
    for (int i = 0; i < 32 * 32; ++i) {
        input_data[i] = (static_cast<float>(resized.data[i]) / 255.0f - 0.5f) / 0.5f;
    }

    // 执行预测
    predictor->Run();

    // 获取输出张量
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(predictor->GetOutput(0));
    const float* output_data = output_tensor->data<float>();
    int num_classes = output_tensor->shape()[1];
    
    // 找到得分最高的类别索引
    int max_idx = std::distance(output_data, std::max_element(output_data, output_data + num_classes));
    
    // 增加置信度阈值判断，过滤掉不确定的结果，提高鲁棒性
    if (output_data[max_idx] < 0.8) return -1; // 如果最高分都小于0.8，则认为没识别出来

    return max_idx;
}

void MainWindow::sendArmCommand(int from_pos, int to_pos)
{
    // 确保串口在发送前是关闭的，然后以写模式打开
    if (serial_port->isOpen()) {
        serial_port->close();
    }
    if (!serial_port->open(QIODevice::WriteOnly)) {
        qCritical() << "无法打开串口 " << serial_port->portName() << " : " << serial_port->errorString();
        return;
    }
    
    // !!! 关键：定义你和下位机（机械臂控制器）的通信协议 !!!
    // 示例协议：帧头(0xAA) + 指令类型(0x01代表搬运) + 起始位置(from_pos) + 目标位置(to_pos) + 帧尾(0xBB)
    QByteArray command;
    command.append(static_cast<char>(0xAA)); // 帧头
    command.append(static_cast<char>(0x01)); // 搬运指令
    command.append(static_cast<char>(from_pos)); // 数据1：起始位置 (0-3)
    command.append(static_cast<char>(to_pos));   // 数据2：目标位置 (0-3)
    command.append(static_cast<char>(0xBB)); // 帧尾

    qDebug() << "正在发送串口指令:" << command.toHex();
    serial_port->write(command);
    serial_port->waitForBytesWritten(100); // 等待数据发送完毕
    serial_port->close(); // 发送完即关闭
}
```

### 3.5. 编译、运行与调试

1. **关键标定 (必做！)**:
    * **ROI标定**: 打开`mainwindow.cpp`，找到`initUI()`函数。点击Qt Creator的运行按钮，程序会启动。观察`videoLabel`中显示的绿色框的位置，**反复修改`warehouse1_rois`的四个`cv::Rect`的坐标和大小**，然后重新编译运行，直至它们精确地框住四个仓库格子。
    * **串口配置**: 在`initSerialPort()`函数中，**修改`setPortName("/dev/ttyS1")`为您实验箱上机械臂控制器实际使用的串口设备名**。您可以通过在实验箱终端输入`ls /dev/tty*`命令来查找可能的串口设备（通常是`/dev/ttyS1`或`/dev/ttyUSB0`等）。

2. **编译与运行**: 在`Qt Creator`中，确保左下角选择了正确的构建套件（通常是Desktop），然后点击绿色的**运行**按钮。

3. **最终测试**: 在实验箱的“仓库一”中摆好2-4块数字积木，点击程序界面上的“开始排序”按钮，观察`statusLabel`的状态信息变化，并查看机械臂是否按照识别和排序的结果正确执行搬运动作。
