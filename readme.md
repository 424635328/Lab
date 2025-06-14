# AI教学实验箱：数字积木排序机械臂项目【完整代码与步骤终极版】

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
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装核心工具
sudo apt install -y build-essential git python3-pip python3-venv wget curl
```

#### 3.1.2. 配置Python环境 (使用pyenv)

> **说明**: Ubuntu 24.04默认的Python 3.12与当前PaddlePaddle不兼容。我们将使用`pyenv`安装并管理Python 3.11，这是解决版本冲突的最佳实践。

```bash
# 1. 安装pyenv及其编译依赖
sudo apt install -y make libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl
curl https://pyenv.run | bash

# 2. 配置shell以加载pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec "$SHELL"

# 3. 安装Python 3.11.9
pyenv install 3.11.9
```

#### 3.1.3. 创建项目并配置虚拟环境

```bash
# 1. 创建项目目录并指定Python版本
mkdir ~/ai_arm_project
cd ~/ai_arm_project
pyenv local 3.11.9

# 2. 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate
```

> **提示**: 之后所有Python相关的命令行操作，都应在激活的`(venv)`环境下进行。

#### 3.1.4. 安装Python依赖库

```bash
# 在激活的(venv)环境中执行
pip install --upgrade pip setuptools wheel
pip install paddlepaddle==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python numpy
```

#### 3.1.5. 配置PaddleLite `opt`工具

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

1. 在PC上，从[PaddleLite Release页面](https://github.com/PaddlePaddle/Paddle-Lite/releases)下载适用于**ARMv8 (aarch64)** 的C++预测库。文件通常命名为 `inference_lite_lib.linux.aarch64.*.tar.gz`。
2. **将下载的压缩包传输到实验箱**。假设实验箱IP为`192.168.1.101`，用户名为`linux`，在**PC终端**的下载目录中执行：

    ```bash
    # 语法: scp <本地文件路径> <用户名>@<IP地址>:<远程目标路径>
    scp ./inference_lite_lib.linux.aarch64.gcc.tar.gz linux@192.168.1.101:~/
    ```

3. 在**实验箱终端**上解压该文件到指定目录：

    ```bash
    mkdir -p ~/paddlelite_cpp
    tar -zxvf ~/inference_lite_lib.linux.aarch64.gcc.tar.gz -C ~/paddlelite_cpp
    ```

    最终路径应为 `/home/linux/paddlelite_cpp/inference_lite_lib.linux.aarch64/`。

---

## Phase 1: 数据集构建

### 1.1. 在实验箱上采集图像

1. 在实验箱上创建一个文件 `collect_data.py`，并复制以下内容：

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

2. 在实验箱终端运行 `python3 collect_data.py`，并按提示进行多样化采集，确保每个数字类别图像丰富。

### 1.2. 数据传输与划分 (在PC上)

1. **传输数据**: 获取实验箱IP后，在**PC终端**执行`scp`命令。

    ```bash
    # 回到PC的项目目录
    cd ~/ai_arm_project
    
    # 将实验箱上采集的整个文件夹递归(-r)拷贝过来
    scp -r linux@192.168.1.101:~/Desktop/digital_images ./digital_raw_data
    ```

2. **划分数据**: 在PC的`~/ai_arm_project/`目录下创建 `split_dataset.py` 文件，内容如下：

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

3. 执行脚本: `python3 split_dataset.py`

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

在`~/ai_arm_project/`目录下执行：

```bash
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

1. **创建项目**: 在实验箱上打开`Qt Creator`，按之前的步骤创建`ArmSortApp`项目。
2. **传输模型文件**: 在**PC终端**执行，将最终模型传输到实验箱上的项目文件夹内。

    ```bash
    # 语法: scp <本地文件> <用户名>@<IP>:<远程项目路径>
    scp ~/ai_arm_project/lite_model/lenet.nb linux@192.168.1.101:~/ArmSortApp/
    ```

### 3.2. 配置Qt项目 (`.pro`文件)

双击Qt Creator左侧项目树中的`ArmSortApp.pro`文件，用以下内容**完全替换**原有内容。
**请务必根据您自己的实际情况修改 `PADDLE_LITE_CPP_DIR` 的路径！**

```qmake
# ArmSortApp.pro
QT       += core gui serialport
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG   += c++11
CONFIG   -= app_bundle

DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
    main.cpp \
    mainwindow.cpp
HEADERS  += \
    mainwindow.h
FORMS    += \
    mainwindow.ui

# 使用pkg-config自动链接OpenCV
CONFIG += link_pkgconfig
PKGCONFIG += opencv4

# 配置PaddleLite C++预测库
# !!! 关键：请修改为你在实验箱上解压PaddleLite库的实际路径 !!!
PADDLE_LITE_CPP_DIR = /home/linux/paddlelite_cpp/inference_lite_lib.linux.aarch64

INCLUDEPATH += $$PADDLE_LITE_CPP_DIR/include
LIBS += -L$$PADDLE_LITE_CPP_DIR/lib -lpaddle_light_api_shared

# 自动将依赖的动态库拷贝到编译输出目录
PADDLE_LITE_SO = $$PADDLE_LITE_CPP_DIR/lib/libpaddle_light_api_shared.so
DESTDIR_SO = $$OUT_PWD
COPY_CMD = $$QMAKE_COPY $$quote($$PADDLE_LITE_SO) $$quote($$DESTDIR_SO)
QMAKE_POST_LINK += $$COPY_CMD $$escape_expand(\\n\\t)
```

> **重要**: 修改完 `.pro` 文件后，右键点击项目名称，选择 `Run qmake`，让配置生效。

### 3.3. 设计UI界面 (`mainwindow.ui`)

1. 双击`mainwindow.ui`文件，进入**设计模式**。
2. 从左侧的**控件盒子 (Widget Box)** 中拖拽以下控件到主窗口上：
    * 一个 `QLabel`，用于显示摄像头视频。将其`objectName`属性修改为 `videoLabel`。
    * 一个 `QPushButton`，用于启动任务。将其`text`属性修改为 `开始排序`，`objectName`属性修改为 `startButton`。
    * 一个 `QLabel`，用于显示状态信息。将其`text`属性留空，`objectName`属性修改为 `statusLabel`。
3. 使用顶部的布局工具对控件进行排列。
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

// OpenCV Headers
#include <opencv2/opencv.hpp>

// PaddleLite Headers
#include "paddle_api.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

// 用于保存识别结果的数据结构
struct DetectionResult {
    int digit;
    int original_pos_index; // 在仓库一中的原始位置索引 (0-3)

    // 重载小于操作符，方便排序
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
    void update_camera_frame();    // 定时器触发，用于更新视频帧

private:
    // 初始化函数
    void initCamera();
    void initModel();
    void initSerialPort();
    void initUI();

    // 核心功能函数
    std::vector<DetectionResult> processFrame(const cv::Mat& frame);
    int predict(const cv::Mat& roi);
    void sendArmCommand(int from_pos, int to_pos);

    Ui::MainWindow *ui;

    // Qt组件
    QTimer *camera_timer;
    QSerialPort *serial_port;

    // OpenCV组件
    cv::VideoCapture cap;

    // PaddleLite组件
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor;

    // 业务逻辑变量
    bool is_task_running = false;
    std::vector<cv::Rect> warehouse1_rois; // 仓库一4个格子的ROI
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

    initUI();
    initModel();
    initCamera();
    initSerialPort();
}

MainWindow::~MainWindow()
{
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
    this->setWindowTitle("AI机械臂排序系统");
    ui->statusLabel->setText("系统准备就绪。");

    // !!! 关键：根据你的摄像头和布局，手动标定这4个ROI矩形框 !!!
    // cv::Rect(x, y, width, height)
    warehouse1_rois.push_back(cv::Rect(50, 50, 100, 100));    // 左上角格子
    warehouse1_rois.push_back(cv::Rect(200, 50, 100, 100));   // 右上角格子
    warehouse1_rois.push_back(cv::Rect(50, 200, 100, 100));   // 左下角格子
    warehouse1_rois.push_back(cv::Rect(200, 200, 100, 100));  // 右下角格子
}

void MainWindow::initModel()
{
    // 模型文件应与可执行文件在同一目录
    QString model_path = QApplication::applicationDirPath() + "/lenet.nb";
    
    ui->statusLabel->setText("正在加载AI模型...");
    QApplication::processEvents();

    paddle::lite_api::MobileConfig config;
    config.set_model_from_file(model_path.toStdString());

    try {
        predictor = paddle::lite_api::CreatePaddlePredictor(config);
        qDebug() << "PaddleLite model loaded from:" << model_path;
        ui->statusLabel->setText("AI模型加载成功。");
    } catch (const std::exception& e) {
        qCritical() << "Failed to load model: " << e.what();
        ui->statusLabel->setText("错误：AI模型加载失败！");
        ui->startButton->setEnabled(false);
    }
}

void MainWindow::initCamera()
{
    cap.open(0); // 0代表默认摄像头
    if (!cap.isOpened()) {
        qCritical() << "Cannot open camera!";
        ui->statusLabel->setText("错误：无法打开摄像头！");
        ui->startButton->setEnabled(false);
        return;
    }

    camera_timer = new QTimer(this);
    connect(camera_timer, &QTimer::timeout, this, &MainWindow::update_camera_frame);
    camera_timer->start(40); // 25 FPS
}

void MainWindow::initSerialPort()
{
    serial_port = new QSerialPort(this);
    serial_port->setPortName("/dev/ttyS1"); // !!! 修改为你的实验箱机械臂的实际串口 !!!
    serial_port->setBaudRate(QSerialPort::Baud115200);
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

    // 在视频帧上绘制ROI框，方便调试
    if (!is_task_running) {
        for (const auto& roi : warehouse1_rois) {
            cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);
        }
    }

    // 将OpenCV的Mat格式转换为Qt的QImage格式以在QLabel中显示
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    QImage qimg(frame.data, frame.cols, frame.rows, static_cast<int>(frame.step), QImage::Format_RGB888);
    ui->videoLabel->setPixmap(QPixmap::fromImage(qimg).scaled(ui->videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::on_startButton_clicked()
{
    if (is_task_running) return;
    
    is_task_running = true;
    ui->startButton->setEnabled(false);
    ui->statusLabel->setText("任务开始：正在识别积木...");
    QApplication::processEvents();

    cv::Mat current_frame;
    cap >> current_frame;
    if (current_frame.empty()) {
        ui->statusLabel->setText("错误：无法捕获图像！");
        is_task_running = false;
        ui->startButton->setEnabled(true);
        return;
    }

    std::vector<DetectionResult> results = processFrame(current_frame);

    if (results.empty()) {
        ui->statusLabel->setText("任务完成：未识别到任何积木。");
    } else {
        std::sort(results.rbegin(), results.rend()); // 降序排序

        QString log = "识别并排序完成: ";
        for(const auto& res : results) {
            log += QString::number(res.digit) + " ";
        }
        ui->statusLabel->setText(log);
        QApplication::processEvents();
        QThread::sleep(2); // 显示结果2秒

        for (size_t i = 0; i < results.size(); ++i) {
            int from = results[i].original_pos_index;
            int to = i;
            
            ui->statusLabel->setText(QString("正在搬运积木 %1: 从位置 %2 到 %3...")
                                     .arg(results[i].digit).arg(from + 1).arg(to + 1));
            QApplication::processEvents();
            
            sendArmCommand(from, to);
            QThread::sleep(5); // 简化处理：假设机械臂动作需要5秒
        }
        ui->statusLabel->setText("任务全部完成！");
    }
    
    is_task_running = false;
    ui->startButton->setEnabled(true);
}

std::vector<DetectionResult> MainWindow::processFrame(const cv::Mat& frame) {
    std::vector<DetectionResult> detections;
    for (size_t i = 0; i < warehouse1_rois.size(); ++i) {
        if(warehouse1_rois[i].x + warehouse1_rois[i].width < frame.cols &&
           warehouse1_rois[i].y + warehouse1_rois[i].height < frame.rows)
        {
            cv::Mat roi_img = frame(warehouse1_rois[i]);
            int digit = predict(roi_img);
            if (digit != -1) {
                detections.push_back({digit, static_cast<int>(i)});
            }
        }
    }
    return detections;
}

int MainWindow::predict(const cv::Mat& roi) {
    if (!predictor || roi.empty()) return -1;

    cv::Mat resized, gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(32, 32));

    std::unique_ptr<paddle::lite_api::Tensor> input_tensor(predictor->GetInput(0));
    input_tensor->Resize({1, 1, 32, 32});
    auto* input_data = input_tensor->mutable_data<float>();
    
    for (int i = 0; i < 32 * 32; ++i) {
        input_data[i] = (static_cast<float>(resized.data[i]) / 255.0f - 0.5f) / 0.5f;
    }

    predictor->Run();

    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(predictor->GetOutput(0));
    const float* output_data = output_tensor->data<float>();
    int num_classes = output_tensor->shape()[1];
    
    int max_idx = std::distance(output_data, std::max_element(output_data, output_data + num_classes));
    
    // 增加置信度阈值判断，过滤掉不确定的结果
    if (output_data[max_idx] < 0.8) return -1;

    return max_idx;
}

void MainWindow::sendArmCommand(int from_pos, int to_pos)
{
    if (serial_port->isOpen()) {
        serial_port->close();
    }
    if (!serial_port->open(QIODevice::WriteOnly)) {
        qCritical() << "无法打开串口 " << serial_port->portName() << " : " << serial_port->errorString();
        return;
    }
    
    // !!! 关键：定义你和下位机（机械臂控制器）的通信协议 !!!
    // 示例协议：帧头(0xAA) 指令(0x01) 数据1(from_pos) 数据2(to_pos) 帧尾(0xBB)
    QByteArray command;
    command.append(static_cast<char>(0xAA));
    command.append(static_cast<char>(0x01));
    command.append(static_cast<char>(from_pos));
    command.append(static_cast<char>(to_pos));
    command.append(static_cast<char>(0xBB));

    qDebug() << "Sending command to serial port:" << command.toHex();
    serial_port->write(command);
    serial_port->waitForBytesWritten(100);
    serial_port->close();
}
```

### 3.5. 编译、运行与调试

1. **关键标定 (必做！)**:
    * **ROI标定**: 打开`mainwindow.cpp`，找到`initUI()`函数。运行一次程序，根据视频画面中绿色框的位置，**反复修改`warehouse1_rois`的四个`cv::Rect`的坐标和大小**，直至它们精确地框住四个仓库格子。
    * **串口配置**: 在`initSerialPort()`函数中，**修改`setPortName("/dev/ttyS1")`为您实验箱上机械臂控制器实际使用的串口号**。您可以通过`ls /dev/tty*`命令查找可能的串口设备。

2. **编译与运行**: 在`Qt Creator`中，确保选择了正确的构建套件，点击**运行**按钮。
3. **最终测试**: 在实验箱上摆好积木，点击程序界面上的“开始排序”按钮，观察`statusLabel`的状态信息和机械臂的实际动作。

>scp

---
