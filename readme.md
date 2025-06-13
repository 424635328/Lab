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

此阶段将在实验箱上使用`Qt Creator`完成整个应用程序的开发。

### 3.1. 准备工作

1.  **传输模型**: 确认已将PC上生成的`lite_model/lenet.nb`文件通过`scp`传输到实验箱的`~/`主目录。
2.  **创建项目**:
    *   在实验箱上打开`Qt Creator`。
    *   选择 `File -> New Project`。
    *   选择 `Application (Qt) -> Qt Widgets Application -> Choose...`。
    *   项目名称填写 `ArmSortApp`，路径选择 `~/` (主目录)。
    *   构建系统选择 `qmake`。
    *   类信息保持默认 (`MainWindow`, base class `QMainWindow`)。
    *   构建套件(Kit)选择默认的Desktop kit。
    *   完成创建。
3.  **移动模型**: 在实验箱的文件管理器中，将`~/lenet.nb`文件移动到刚刚创建的`~/ArmSortApp/`项目文件夹内。

### 3.2. 配置Qt项目 (`.pro`文件)

双击Qt Creator左侧项目树中的`ArmSortApp.pro`文件，用以下内容**完全替换**原有内容。

**请务必根据您自己的实际情况修改 `PADDLE_LITE_CPP_DIR` 的路径！**

```qmake
# ArmSortApp.pro

QT       += core gui serialport
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG   += c++11
CONFIG   -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated.
DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

# === 配置OpenCV ===
# 使用pkg-config自动链接OpenCV，这是在Linux上最稳健的方式
CONFIG += link_pkgconfig
PKGCONFIG += opencv4

# === 配置PaddleLite C++预测库 ===
# !!! 关键：请修改为你在实验箱上解压PaddleLite库的实际路径 !!!
PADDLE_LITE_CPP_DIR = /home/linux/paddlelite_cpp/inference_lite_lib.linux.aarch64

INCLUDEPATH += $$PADDLE_LITE_CPP_DIR/include

# 链接PaddleLite动态库
LIBS += -L$$PADDLE_LITE_CPP_DIR/lib -lpaddle_light_api_shared

# === 部署步骤：自动将依赖的动态库拷贝到编译输出目录 ===
# 这样，程序运行时就能直接找到它，无需配置LD_LIBRARY_PATH
PADDLE_LITE_SO = $$PADDLE_LITE_CPP_DIR/lib/libpaddle_light_api_shared.so

# 获取编译输出目录 (e.g., build-ArmSortApp-Desktop-Debug/)
DESTDIR_SO = $$OUT_PWD

# 定义一个拷贝命令，在链接步骤完成后执行
# $$quote确保路径中有空格也能正确处理
# $$escape_expand确保换行符被正确解析
COPY_CMD = $$QMAKE_COPY $$quote($$PADDLE_LITE_SO) $$quote($$DESTDIR_SO)
QMAKE_POST_LINK += $$COPY_CMD $$escape_expand(\\n\\t)
```
> **重要**: 修改完 `.pro` 文件后，右键点击项目名称，选择 `Run qmake`，让配置生效。

### 3.3. 设计UI界面 (`mainwindow.ui`)

1.  双击`mainwindow.ui`文件，进入**设计模式**。
2.  从左侧的**控件盒子 (Widget Box)** 中拖拽以下控件到主窗口上：
    *   一个 `QLabel`，用于显示摄像头视频。将其`objectName`属性修改为 `videoLabel`。
    *   一个 `QPushButton`，用于启动任务。将其`text`属性修改为 `开始排序`，`objectName`属性修改为 `startButton`。
    *   一个 `QLabel`，用于显示状态信息。将其`text`属性留空，`objectName`属性修改为 `statusLabel`。
3.  使用顶部的布局工具（如垂直布局、水平布局、栅格布局）对控件进行简单的排列，使其美观。
4.  按下 `Ctrl + S` 保存UI设计。

### 3.4. 编写C++核心代码

现在，我们将填充头文件和源文件的内容。

#### 3.4.1. 头文件 (`mainwindow.h`)

用以下内容**完全替换** `mainwindow.h`：

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QSerialPort>
#include <vector>
#include <memory>

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

#### 3.4.2. 源文件 (`mainwindow.cpp`)

用以下内容**完全替换** `mainwindow.cpp`：

```cpp
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QApplication>

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
    // 获取可执行文件所在目录，模型文件应与可执行文件在同一目录
    QString model_path = QApplication::applicationDirPath() + "/lenet.nb";
    
    ui->statusLabel->setText("正在加载AI模型...");
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
    camera_timer->start(40); // 每秒25帧
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
    for (const auto& roi : warehouse1_rois) {
        cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);
    }

    // 将OpenCV的Mat格式转换为Qt的QImage格式以在QLabel中显示
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    QImage qimg(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
    ui->videoLabel->setPixmap(QPixmap::fromImage(qimg).scaled(ui->videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::on_startButton_clicked()
{
    if (is_task_running) return;
    
    is_task_running = true;
    ui->startButton->setEnabled(false);
    ui->statusLabel->setText("任务开始：正在识别积木...");
    QApplication::processEvents(); // 强制UI刷新

    cv::Mat current_frame;
    cap >> current_frame;
    if (current_frame.empty()) {
        ui->statusLabel->setText("错误：无法捕获图像！");
        is_task_running = false;
        ui->startButton->setEnabled(true);
        return;
    }

    // 核心处理逻辑
    std::vector<DetectionResult> results = processFrame(current_frame);

    if (results.empty()) {
        ui->statusLabel->setText("任务完成：未识别到任何积木。");
    } else {
        // 降序排序
        std::sort(results.rbegin(), results.rend());

        QString log = "识别并排序完成: ";
        for(const auto& res : results) {
            log += QString::number(res.digit) + " ";
        }
        ui->statusLabel->setText(log);
        QApplication::processEvents();

        // 执行机械臂搬运
        for (size_t i = 0; i < results.size(); ++i) {
            int from = results[i].original_pos_index;
            int to = i; // 目标位置按排序后的顺序
            
            ui->statusLabel->setText(QString("正在搬运积木 %1: 从位置 %2 到 %3...")
                                     .arg(results[i].digit).arg(from + 1).arg(to + 1));
            QApplication::processEvents();
            
            sendArmCommand(from, to);
            // 这里应该等待机械臂完成动作，可以通过接收串口返回信号
            // 为简化，我们使用固定延时
            QThread::sleep(5); // 等待5秒
        }
        ui->statusLabel->setText("任务全部完成！");
    }
    
    is_task_running = false;
    ui->startButton->setEnabled(true);
}

std::vector<DetectionResult> MainWindow::processFrame(const cv::Mat& frame) {
    std::vector<DetectionResult> detections;
    for (size_t i = 0; i < warehouse1_rois.size(); ++i) {
        cv::Mat roi_img = frame(warehouse1_rois[i]);
        int digit = predict(roi_img);

        // 假设模型输出-1表示没有识别到，或者置信度低
        if (digit != -1) {
            detections.push_back({digit, static_cast<int>(i)});
        }
    }
    return detections;
}

int MainWindow::predict(const cv::Mat& roi) {
    if (!predictor || roi.empty()) return -1;

    // 1. 预处理
    cv::Mat resized, gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(32, 32));

    // 2. 准备输入数据
    std::unique_ptr<paddle::lite_api::Tensor> input_tensor(predictor->GetInput(0));
    input_tensor->Resize({1, 1, 32, 32});
    auto* input_data = input_tensor->mutable_data<float>();
    
    // 3. 填充Tensor并归一化
    for (int i = 0; i < 32 * 32; ++i) {
        input_data[i] = (static_cast<float>(resized.data[i]) / 255.0f - 0.5f) / 0.5f;
    }

    // 4. 执行推理
    predictor->Run();

    // 5. 获取输出
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(predictor->GetOutput(0));
    const float* output_data = output_tensor->data<float>();
    auto output_shape = output_tensor->shape();
    int num_classes = output_shape[1];
    
    // 6. 找到概率最高的类别
    int max_idx = std::distance(output_data, std::max_element(output_data, output_data + num_classes));
    
    // 可以增加一个置信度阈值判断
    // if (output_data[max_idx] < 0.5) return -1;

    return max_idx;
}

void MainWindow::sendArmCommand(int from_pos, int to_pos)
{
    if (!serial_port->isOpen()) {
        if (!serial_port->open(QIODevice::WriteOnly)) {
            qCritical() << "无法打开串口 " << serial_port->portName() << " : " << serial_port->errorString();
            return;
        }
    }
    
    // !!! 关键：定义你和下位机（机械臂控制器）的通信协议 !!!
    // 示例协议：一个字节帧头，一个字节指令，两个字节数据，一个字节帧尾
    // 0xAA 0x01 <from_pos> <to_pos> 0xBB
    QByteArray command;
    command.append(static_cast<char>(0xAA)); // 帧头
    command.append(static_cast<char>(0x01)); // 0x01代表“搬运”指令
    command.append(static_cast<char>(from_pos));
    command.append(static_cast<char>(to_pos));
    command.append(static_cast<char>(0xBB)); // 帧尾

    qDebug() << "Sending command to serial port:" << command.toHex();
    serial_port->write(command);
    // 最好等待写入完成
    serial_port->waitForBytesWritten(100);

    // 在实际项目中，最好在发送后关闭，或保持长连接
    if (serial_port->isOpen()) {
        serial_port->close();
    }
}
```

### 3.5. 编译、运行与调试

1.  **关键标定**: 打开`mainwindow.cpp`，找到`initUI()`函数，**手动修改`warehouse1_rois`的四个`cv::Rect`的坐标和大小**，使其精确框住摄像头视野中的四个仓库格子。您可以通过运行程序，观察绿色框的位置来进行调试。
2.  **串口配置**: 在`initSerialPort()`函数中，**修改`setPortName`为您实验箱上机械臂控制器实际使用的串口号**（如`/dev/ttyS0`, `/dev/ttyUSB0`等）。
3.  **编译与运行**: 在`Qt Creator`中点击**运行**按钮。程序会自动编译、链接，并将`libpaddle_light_api_shared.so`和`lenet.nb`都置于正确的运行目录下。
4.  **最终测试**: 在实验箱上摆好积木，点击程序界面上的“开始排序”按钮，观察`statusLabel`的状态变化和机械臂的实际动作。