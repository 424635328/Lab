# AI教学实验箱：数字积木排序项目 (PyQt5终极版)

## 1. 项目概述

本项目采用**Python + PyQt5**技术栈，在实验箱官方的Ubuntu 16.04 & Python 3.5环境下，完整复现一个从AI模型训练到硬件控制的端到端应用。项目通过摄像头识别积木上的数字，利用自行训练的模型进行识别，按数字大小降序排序，并最终控制机械臂完成搬运任务。

**核心特性:**

* **统一技术栈**: 从数据采集到最终应用，全程使用Python，开发高效。
* **现代化GUI**: 使用PyQt5构建功能丰富、响应灵敏的图形用户界面。
* **官方环境适配**: 所有步骤均基于实验箱内置的 **Ubuntu 16.04** 和 **Python 3.5** 环境。

## 2. 技术栈

* **PC/实验箱平台**: Ubuntu 16.04
* **核心语言**: **Python 3.5**
* **深度学习框架**: PaddlePaddle (<=2.4.2)
* **模型部署框架**: PaddleLite
* **GUI框架**: **PyQt5**
* **计算机视觉库**: OpenCV-Python
* **串口通信库**: PySerial

## 3. Phase 0: 环境搭建 (PyQt5版)

### 3.1. PC端环境搭建

在您的PC上（推荐Ubuntu 16.04/18.04/20.04虚拟机，以保证兼容性），配置好Python 3.5+的环境，并安装好`paddlepaddle`, `opencv-python`, `numpy`。

### 3.2. 实验箱环境搭建 (关键)

此环境用于运行最终的PyQt5应用程序。

1. **连接到实验箱** (通过直接操作或VNC/SSH)。

2. **创建隔离的Python 3.5虚拟环境**:

    ```bash
    # 在实验箱终端
    pip3 install --user virtualenv
    mkdir ~/my_arm_project_pyqt
    cd ~/my_arm_project_pyqt
    virtualenv -p python3.5 venv
    source venv/bin/activate
    ```

3. **在虚拟环境中安装所有Python依赖**:

    ```bash
    # (venv)环境下执行
    # 1. 升级pip
    pip install --upgrade pip

    # 2. 安装PyQt5及相关工具
    # Ubuntu 16.04的源中PyQt5版本较旧，但可用
    pip install PyQt5==5.15.2
    pip install PyQt5-sip==12.8.1

    # 3. 安装其他库
    pip install numpy==1.21.0
    pip install opencv-python==4.5.5.64
    pip install pyserial

    # 4. 安装PaddleLite Python API
    # 从PC下载并scp传输与Python 3.5 (cp35)匹配的.whl文件到当前目录
    # 然后安装
    # pip install paddlelite-*-cp35-*-linux_aarch64.whl
    ```

    > **注意**: 安装PyQt5时可能会从源码编译，耗时较长，请耐心等待。如果pip安装失败，可尝试系统级安装：`sudo apt-get install python3-pyqt5`，但这不推荐。

---

## Phase 1 & 2: 数据集构建与模型训练

这部分与之前的指南**完全相同**。请严格遵循之前的步骤在您的PC（或实验箱）上完成：

1. **采集数据** (`collect_data.py`)。
2. **划分数据集** (`split_dataset.py`)。
3. **训练模型** (`train.py`)。
4. **转换模型** (使用`opt`工具)，得到最终的`lite_model/lenet.nb`。

---

## Phase 3: PyQt5应用开发与部署 (在实验箱上)

### 3.1. 准备工作

1. **创建项目目录**: 如果尚未创建，`mkdir ~/my_arm_project_pyqt`。
2. **传输模型**: 将PC上生成的`lite_model/lenet.nb`文件通过`scp`传输到实验箱的`~/my_arm_project_pyqt/`目录下。

### 3.2. 编写PyQt5应用程序

在实验箱的`~/my_arm_project_pyqt/`目录下，创建一个名为 `main_app_pyqt.py` 的文件，并将以下完整代码复制进去。

```python
# main_app_pyqt.py
import sys
import threading
import time
import cv2
import numpy as np
import serial
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# 导入PaddleLite Python API
from paddlelite.lite import CxxConfig, create_paddle_predictor

# --- 全局配置 (请务必修改以匹配您的硬件) ---
CAMERA_INDEX = 0
MODEL_PATH = "lenet.nb"
SERIAL_PORT = '/dev/ttyS1'  # !!! 重要：修改为你的实际串口号 !!!
BAUD_RATE = 115200

# !!! 重要：根据你的摄像头视野，手动标定这4个ROI矩形框 !!!
# 格式: (x_左上角, y_左上角, 宽度, 高度)
WAREHOUSE1_ROIS = [
    (50, 50, 100, 100), (200, 50, 100, 100),
    (50, 200, 100, 100), (200, 200, 100, 100)
]
# ---

# Step 1: 创建一个工作线程类来处理耗时任务
class WorkerThread(QThread):
    """
    一个专用的工作线程，用于执行AI识别和机械臂控制，避免UI线程阻塞。
    """
    # 定义信号，用于向主线程发送状态更新
    status_update = pyqtSignal(str)
    task_finished = pyqtSignal()

    def __init__(self, app_instance):
        super().__init__()
        self.app = app_instance

    def run(self):
        """线程启动时执行的核心逻辑"""
        self.status_update.emit("任务开始：正在识别积木...")

        ret, frame = self.app.cap.read()
        if not ret:
            self.status_update.emit("错误：无法从摄像头捕获图像！")
            self.task_finished.emit()
            return

        # 1. 识别所有积木
        results = []
        for i, (x, y, w, h) in enumerate(WAREHOUSE1_ROIS):
            if x + w < frame.shape[1] and y + h < frame.shape[0]:
                roi = frame[y:y+h, x:x+w]
                digit = self.app.predict(roi)
                if digit != -1:
                    results.append({'digit': digit, 'from_pos': i})

        # 2. 排序并执行搬运
        if results:
            results.sort(key=lambda x: x['digit'], reverse=True) # 降序排序
            log_msg = "识别排序完成: " + " ".join([str(r['digit']) for r in results])
            self.status_update.emit(log_msg)
            time.sleep(2)

            for i, res in enumerate(results):
                from_p, to_p = res['from_pos'], i
                self.status_update.emit(f"正在搬运 {res['digit']}: 从位置{from_p+1}到{to_p+1}...")
                self.app.send_arm_command(from_p, to_p)
                time.sleep(5) # 简化处理：假设机械臂动作需要5秒

            self.status_update.emit("任务全部完成！")
        else:
            self.status_update.emit("任务完成：未识别到任何积木。")

        self.task_finished.emit()


# Step 2: 创建主窗口类
class ArmSortApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI机械臂排序系统 (PyQt5版)")
        self.setGeometry(100, 100, 800, 600) # x, y, width, height

        # 初始化核心组件
        self.predictor = self._init_model()
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise IOError(f"无法打开摄像头 {CAMERA_INDEX}")

        # 设置UI
        self._create_widgets()

        # 设置定时器用于更新视频
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_feed)
        self.timer.start(40) # 25 FPS

    def _init_model(self):
        """加载PaddleLite模型"""
        try:
            config = CxxConfig()
            config.set_model_file(MODEL_PATH)
            predictor = create_paddle_predictor(config)
            print("PaddleLite模型加载成功。")
            return predictor
        except Exception as e:
            print(f"错误：模型加载失败！ {e}")
            return None

    def _create_widgets(self):
        """创建所有PyQt5界面组件"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # 视频显示标签
        self.video_label = QLabel("正在加载摄像头...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.video_label, 1) # 占据大部分空间

        # 开始按钮
        self.start_button = QPushButton("开始排序")
        self.start_button.clicked.connect(self.start_task)
        if self.predictor is None: self.start_button.setEnabled(False)
        layout.addWidget(self.start_button)

        # 状态显示标签
        self.status_label = QLabel("系统准备就绪。")
        self.status_label.setAlignment(Qt.AlignCenter)
        font = self.status_label.font()
        font.setPointSize(14)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)

    def update_video_feed(self):
        """定时更新摄像头画面"""
        ret, frame = self.cap.read()
        if ret:
            # 在视频帧上绘制ROI框
            for (x, y, w, h) in WAREHOUSE1_ROIS:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 转换图像格式以在PyQt中显示
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def predict(self, roi_img):
        """对单个ROI图像进行预处理和AI推理"""
        if not self.predictor or roi_img is None: return -1

        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        img_data = (resized.astype('float32') / 255.0 - 0.5) / 0.5
        img_data = img_data[np.newaxis, np.newaxis, :, :]

        input_tensor = self.predictor.get_input(0)
        input_tensor.from_numpy(img_data)
        self.predictor.run()
        output_tensor = self.predictor.get_output(0)
        output_data = output_tensor.numpy().flatten()
        
        prediction = np.argmax(output_data)
        confidence = output_data[prediction]
        
        return prediction if confidence > 0.8 else -1

    def send_arm_command(self, from_pos, to_pos):
        """通过串口发送机械臂控制指令"""
        try:
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
                command = bytearray([0xAA, 0x01, from_pos, to_pos, 0xBB])
                ser.write(command)
                print(f"发送串口指令: {command.hex()}")
        except serial.SerialException as e:
            self.update_status_label(f"串口错误: {e}")

    def start_task(self):
        """启动后台工作线程"""
        self.start_button.setEnabled(False)
        self.worker = WorkerThread(self)
        self.worker.status_update.connect(self.update_status_label)
        self.worker.task_finished.connect(lambda: self.start_button.setEnabled(True))
        self.worker.start()

    def update_status_label(self, message):
        """线程安全地更新状态标签"""
        self.status_label.setText(message)

    def closeEvent(self, event):
        """处理窗口关闭事件，释放资源"""
        self.timer.stop()
        self.cap.release()
        print("摄像头和定时器已释放。")
        event.accept()

# Step 3: 程序主入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ArmSortApp()
    main_window.show()
    sys.exit(app.exec_())
```

### 3.3. 运行与调试

1. **关键标定 (必做！)**: 打开 `main_app_pyqt.py` 文件，仔细修改**全局配置区**的 `SERIAL_PORT` 和 `WAREHOUSE1_ROIS` 变量，以匹配您的实际硬件情况。
2. **运行程序**: 在实验箱的 `~/my_arm_project_pyqt/` 目录下，打开终端，激活虚拟环境后执行：

    ```bash
    # 激活环境
    source venv/bin/activate
    # 运行主程序
    python main_app_pyqt.py
    ```

3. **测试**:
    * 一个PyQt5窗口将会出现，并显示实时摄像头画面。
    * 在“仓库一”摆放好数字积木。
    * 点击“开始排序”按钮。
    * 观察下方的状态标签，它会实时更新任务状态。
    * 观察机械臂是否按照预期的降序顺序执行搬运任务。
    