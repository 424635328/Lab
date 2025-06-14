# AI教学实验箱：数字积木排序机械臂项目 (纯Python版指南)

## 1. 项目概述

本项目旨在利用人工智能教学实验箱，采用**纯Python技术栈**完整复现一个从AI模型训练到硬件控制的端到端应用。项目通过摄像头识别“仓库一”中的数字积木，利用自行训练的深度学习模型进行识别，然后根据数字大小进行降序排序，并最终控制机械臂将积木按排序结果搬运至“仓库二”。

**核心特性:**

* **统一技术栈**: 从数据采集到最终应用，全程使用Python，简化开发流程。
* **端到端实现**: 涵盖数据采集、模型训练、模型优化、应用部署全流程。
* **环境适应性**: 软件算法需考虑并处理真实环境中的光照、位置、角度等干扰因素。
* **平台特定**: 教程针对 **Ubuntu 24.04 LTS PC** (用于训练) 和 **实验箱内置Linux系统** (用于部署) 进行了优化。

## 2. 技术栈

* **PC平台**: Ubuntu 24.04 LTS
* **实验箱平台**: 内置Linux系统 (ARMv8/aarch64)
* **核心语言**: **Python 3**
* **深度学习框架**: PaddlePaddle
* **模型部署框架**: PaddleLite
* **GUI框架**: **Tkinter** (Python内置)
* **计算机视觉库**: **OpenCV-Python**
* **串口通信库**: **PySerial**

## 3. Phase 0: 环境搭建

### 3.1. PC环境搭建 (Ubuntu 24.04 LTS)

此环境用于数据处理、模型训练与转换。
*(如果您的PC环境已按之前步骤配置好，可直接跳到 3.2)*

1. **安装系统及开发工具**:

    ```bash
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y build-essential git python3-pip python3-venv wget curl
    ```

2. **安装并配置`pyenv`以使用Python 3.11**:
    *(由于Ubuntu 24.04默认Python 3.12，而PaddlePaddle尚不支持，此步骤至关重要)*

    ```bash
    # 安装pyenv依赖
    sudo apt install -y make libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl
    # 安装pyenv
    curl https://pyenv.run | bash
    # 配置shell
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    exec "$SHELL"
    # 安装Python 3.11并设置为项目默认
    pyenv install 3.11.9
    mkdir ~/ai_arm_project
    cd ~/ai_arm_project
    pyenv local 3.11.9
    ```

3. **创建Python虚拟环境并安装依赖**:

    ```bash
    # 在项目目录 ~/ai_arm_project 下
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install paddlepaddle==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install opencv-python numpy
    ```

4. **配置`opt`模型转换工具**:
    *(与之前相同，确保`opt`命令可用)*

### 3.2. 实验箱环境搭建 (纯Python版)

此环境用于运行最终的Python应用程序。

1. **确认Python版本**: 在实验箱终端运行 `python3 --version`，记下版本号 (例如 `3.10.12`)。
2. **安装Python依赖库**:

    ```bash
    sudo apt update
    # Tkinter通常已内置，以防万一安装它
    sudo apt install -y python3-tk
    # 安装OpenCV, NumPy和PySerial
    pip3 install opencv-python numpy pyserial
    ```

3. **安装PaddleLite Python API**:
    * 在PC上，访问[PaddleLite Release页面](https://github.com/PaddlePaddle/Paddle-Lite/releases)，选择一个新版本（如v2.12.0）。
    * 找到与您**实验箱的Python版本和CPU架构(aarch64)**完全匹配的`.whl`文件。例如，如果实验箱是Python 3.10，就下载 `paddlelite-*-cp310-*-linux_aarch64.whl`。
    * 将下载的`.whl`文件通过`scp`传输到实验箱:

        ```bash
        # 在PC终端执行
        scp ./paddlelite-*.whl linux@<实验箱IP>:~/
        ```

    * 在实验箱上安装它:

        ```bash
        # 在实验箱终端
        pip3 install ~/paddlelite-*.whl
        ```

---

## Phase 1 & 2: 数据集构建与模型训练 (在PC上)

这部分与之前的指南**完全相同**。请严格遵循之前的步骤完成：

1. **在实验箱上采集数据** (使用`collect_data.py`脚本)。
2. **将数据传输到PC** (使用`scp -r`命令)。
3. **在PC上划分数据集** (使用`split_dataset.py`脚本)。
4. **在PC上训练模型** (使用`train.py`脚本)。
5. **在PC上转换模型** (使用`opt`工具)，得到最终的`lite_model/lenet.nb`文件。

---

## Phase 3: 纯Python应用开发与部署 (在实验箱上)

### 3.1. 准备工作

1. **创建项目目录**: 在实验箱上创建 `mkdir ~/ArmSortApp_Python`。
2. **传输模型**: 将PC上生成的`lite_model/lenet.nb`文件通过`scp`传输到实验箱的`~/ArmSortApp_Python/`目录下。

    ```bash
    # 在PC终端执行
    scp ~/ai_arm_project/lite_model/lenet.nb linux@<实验箱IP>:~/ArmSortApp_Python/
    ```

### 3.2. 编写纯Python应用程序

在实验箱的`~/ArmSortApp_Python/`目录下，创建一个名为 `main_app.py` 的文件，并将以下经过优化和注释的完整代码复制进去。

```python
# main_app.py
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
import cv2
import numpy as np
import serial
import threading
import time
from paddlelite.lite import CxxConfig, create_paddle_predictor

# --- 全局配置 (请务必修改以匹配您的硬件) ---
CAMERA_INDEX = 0
MODEL_PATH = "lenet.nb"  # 模型文件应与脚本在同一目录
SERIAL_PORT = '/dev/ttyS1'  # !!! 重要：修改为你的实际串口号 (可能是 /dev/ttyUSB0 等) !!!
BAUD_RATE = 115200

# !!! 重要：根据你的摄像头视野，手动标定这4个ROI矩形框 !!!
# 格式: (x_左上角, y_左上角, 宽度, 高度)
WAREHOUSE1_ROIS = [
    (50, 50, 100, 100),
    (200, 50, 100, 100),
    (50, 200, 100, 100),
    (200, 200, 100, 100)
]
# ---

class ArmSortApp:
    """主应用程序类，封装所有功能"""
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("AI机械臂排序系统 (纯Python版)")

        # 初始化核心组件
        self.predictor = self._init_model()
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise IOError(f"无法打开摄像头 {CAMERA_INDEX}")

        # 创建UI界面
        self._create_widgets()

        # 启动视频更新循环
        self.is_video_running = True
        self.update_video_feed()

    def _init_model(self):
        """加载PaddleLite模型"""
        self._update_status("正在加载AI模型...")
        try:
            config = CxxConfig()
            config.set_model_file(MODEL_PATH)
            predictor = create_paddle_predictor(config)
            print("PaddleLite模型加载成功。")
            self._update_status("AI模型加载成功。")
            return predictor
        except Exception as e:
            print(f"错误：模型加载失败！ {e}")
            self._update_status(f"错误：模型加载失败！")
            return None

    def _create_widgets(self):
        """创建所有Tkinter界面组件"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 视频显示标签
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, pady=10)

        # 开始按钮
        self.start_button = ttk.Button(main_frame, text="开始排序", command=self._start_task_thread)
        self.start_button.grid(row=1, column=0, pady=10)
        if self.predictor is None: self.start_button.config(state=tk.DISABLED)

        # 状态显示标签
        self.status_label = ttk.Label(main_frame, text="系统准备就绪。", font=("Helvetica", 12), anchor="center")
        self.status_label.grid(row=2, column=0, pady=10, sticky="ew")

    def update_video_feed(self):
        """定时更新摄像头画面"""
        if not self.is_video_running: return

        ret, frame = self.cap.read()
        if ret:
            # 在视频帧上绘制ROI框，方便调试
            for (x, y, w, h) in WAREHOUSE1_ROIS:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 转换图像格式以便在Tkinter中显示
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(40, self.update_video_feed) # 约等于25 FPS

    def _predict(self, roi_img):
        """对单个ROI图像进行预处理和AI推理"""
        if not self.predictor or roi_img is None: return -1

        # 预处理
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        img_data = (resized.astype('float32') / 255.0 - 0.5) / 0.5
        img_data = img_data[np.newaxis, np.newaxis, :, :] # 增加 batch 和 channel 维度

        # 运行推理
        input_tensor = self.predictor.get_input(0)
        input_tensor.from_numpy(img_data)
        self.predictor.run()
        output_tensor = self.predictor.get_output(0)
        output_data = output_tensor.numpy().flatten()
        
        # 返回概率最高的类别索引
        prediction = np.argmax(output_data)
        confidence = output_data[prediction]
        
        # 增加置信度阈值，过滤不确定的结果
        return prediction if confidence > 0.8 else -1

    def _send_arm_command(self, from_pos, to_pos):
        """通过串口发送机械臂控制指令"""
        try:
            # 使用'with'语句确保串口能被正确关闭
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
                # 定义通信协议: 帧头(0xAA) 指令(0x01) 数据1(from) 数据2(to) 帧尾(0xBB)
                command = bytearray([0xAA, 0x01, from_pos, to_pos, 0xBB])
                ser.write(command)
                print(f"发送串口指令: {command.hex()}")
        except serial.SerialException as e:
            self._update_status(f"串口错误: {e}")

    def _sorting_task(self):
        """核心业务逻辑：识别、排序、搬运"""
        self.start_button.config(state=tk.DISABLED)
        self._update_status("任务开始：正在识别积木...")

        ret, frame = self.cap.read()
        if not ret:
            self._update_status("错误：无法从摄像头捕获图像！")
            self.start_button.config(state=tk.NORMAL)
            return

        # 1. 识别所有积木
        results = []
        for i, (x, y, w, h) in enumerate(WAREHOUSE1_ROIS):
            # 确保ROI在图像范围内
            if x+w < frame.shape[1] and y+h < frame.shape[0]:
                roi = frame[y:y+h, x:x+w]
                digit = self._predict(roi)
                if digit != -1:
                    results.append({'digit': digit, 'from_pos': i})
        
        # 2. 排序并执行搬运
        if results:
            results.sort(key=lambda x: x['digit'], reverse=True) # 降序排序
            log_msg = "识别排序完成: " + " ".join([str(r['digit']) for r in results])
            self._update_status(log_msg)
            time.sleep(2) # 显示结果2秒

            for i, res in enumerate(results):
                from_p, to_p = res['from_pos'], i
                self._update_status(f"正在搬运 {res['digit']}: 从位置{from_p+1}到{to_p+1}...")
                self._send_arm_command(from_p, to_p)
                time.sleep(5) # 简化处理：假设机械臂动作需要5秒

            self._update_status("任务全部完成！")
        else:
            self._update_status("任务完成：未识别到任何积木。")

        self.start_button.config(state=tk.NORMAL)

    def _start_task_thread(self):
        """将耗时任务放在后台线程执行，避免UI卡顿"""
        task_thread = threading.Thread(target=self._sorting_task)
        task_thread.daemon = True # 设置为守护线程，主程序退出时线程也退出
        task_thread.start()

    def _update_status(self, msg):
        """线程安全的UI更新方法"""
        self.root.after(0, lambda: self.status_label.config(text=msg))

    def on_closing(self):
        """处理窗口关闭事件，释放资源"""
        self.is_video_running = False
        time.sleep(0.1) # 等待视频循环退出
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    # 程序入口
    main_window = tk.Tk()
    app = ArmSortApp(main_window)
    main_window.protocol("WM_DELETE_WINDOW", app.on_closing) # 绑定关闭事件
    main_window.mainloop()
```

### 3.3. 运行与调试

1. **关键标定 (必做！)**: 打开 `main_app.py` 文件，仔细修改**全局配置区**的 `SERIAL_PORT` 和 `WAREHOUSE1_ROIS` 变量，以匹配您的实际硬件情况。这是确保程序正常运行的最关键一步。
2. **运行程序**: 在实验箱的 `~/ArmSortApp_Python/` 目录下，打开终端并执行：

    ```bash
    python3 main_app.py
    ```

3. **测试**:
    * 一个Tkinter窗口将会出现，并显示实时摄像头画面以及绿色的ROI框。
    * 在“仓库一”摆放好数字积木。
    * 点击“开始排序”按钮。
    * 观察下方的状态标签，它会实时更新当前的任务状态。
    * 观察机械臂是否按照预期的降序顺序执行搬运任务。
