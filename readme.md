# AI教学实验箱：数字积木排序机械臂项目【终极版详细指南】

## 0. 项目准备与环境配置

在开始之前，请确保您的PC开发环境（用于训练模型）和实验箱AI运算单元环境（用于部署）已准备就绪。

### 0.1. PC端环境配置 (Windows/Linux)

1. **安装 Python (3.7+)**
2. **安装必要的库:**

    ```bash
    pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
    pip install opencv-python numpy
    pip install paddlelite # 用于获取opt工具的路径信息
    ```

3. **获取PaddleLite模型转换工具 (`opt`)**
    * 通常在`paddlelite`库的安装目录中可以找到。如果找不到，请从[PaddleLite Release页面](https://github.com/PaddlePaddle/Paddle-Lite/releases)下载预编译好的工具包。

### 0.2. 实验箱AI运算单元环境配置

1. **安装开发库 (参考手册第62-67页)**
    * 确保`gcc`, `g++`, `make`已安装。
    * 安装Qt5开发环境: `sudo apt-get install qt5-default qtcreator`
    * 安装OpenCV库: `sudo apt-get install libopencv-dev`
    * 安装PaddleLite C++预测库: 请从[PaddleLite Release页面](https://github.com/PaddlePaddle/Paddle-Lite/releases)下载针对ARMv8（aarch64）的C++预测库，并解压到特定目录，如`/home/linux/paddlelite/`。

---

## Phase 1: 高质量数据集构建

**目标：** 创建一个包含10个类别（数字0-9），能抵抗光照、位置、角度干扰的数据集。

### 1.1. 创建目录结构

在您的PC上创建一个项目文件夹，并在其中建立数据集目录。

```bash
mkdir ~/ai_arm_project
cd ~/ai_arm_project
mkdir -p digital_dataset/{0,1,2,3,4,5,6,7,8,9}
```

### 1.2. 数据采集

此步骤需要在**实验箱**上运行，通过网络将采集的图片传回PC，或直接在实验箱上用U盘拷贝。

**`collect_data.py` (在实验箱上运行)**

```python
import cv2
import os
import time

# --- 配置 ---
SAVE_ROOT = "/home/linux/Desktop/digital_images" # 在实验箱桌面创建文件夹保存
CAMERA_INDEX = 0
# ---

# 确保目录存在
if not os.path.exists(SAVE_ROOT):
    os.makedirs(SAVE_ROOT)
for i in range(10):
    if not os.path.exists(os.path.join(SAVE_ROOT, str(i))):
        os.makedirs(os.path.join(SAVE_ROOT, str(i)))

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("错误: 无法打开摄像头!")
    exit()

print("摄像头已启动。按数字'0'-'9'保存，按'q'退出。")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # 标注提示信息
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
        
        # 裁剪中心区域以减少背景干扰
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        crop_size = min(h, w) // 2
        cropped_img = frame[center_y - crop_size//2 : center_y + crop_size//2,
                            center_x - crop_size//2 : center_x + crop_size//2]
        
        cv2.imwrite(save_path, cropped_img)
        print(f"已保存 -> {save_path}")

cap.release()
cv2.destroyAllWindows()
```

**操作流程:**

1. 将积木放在摄像头正下方。
2. 运行脚本 `python3 collect_data.py`。
3. **多样化采集：**
   * 对每个数字，移动积木位置、轻微旋转、用手制造阴影，并在每种情况下按对应数字键保存。
   * 确保每个数字采集 **100-200** 张图像。
4. 将采集到的 `/home/linux/Desktop/digital_images` 文件夹通过`scp`或U盘，拷贝回PC的 `~/ai_arm_project/digital_dataset/`。

### 1.3. 划分数据集 (在PC上运行)

**`split_dataset.py`**

```python
import os
import random
import shutil

# --- 配置 ---
SOURCE_DIR = "digital_dataset"
DEST_DIR = "data_split"
SPLIT_RATIO = (0.8, 0.1, 0.1) # Train, Val, Test
# ---

if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR)

for-sub_dir in ['train', 'val', 'test']:
    for i in range(10):
        os.makedirs(os.path.join(DEST_DIR, sub_dir, str(i)), exist_ok=True)

for digit in os.listdir(SOURCE_DIR):
    digit_path = os.path.join(SOURCE_DIR, digit)
    if not os.path.isdir(digit_path): continue
    
    images = os.listdir(digit_path)
    random.shuffle(images)
    
    train_count = int(len(images) * SPLIT_RATIO[0])
    val_count = int(len(images) * SPLIT_RATIO[1])
    
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

运行 `python split_dataset.py`，你将得到一个结构化的 `data_split` 文件夹。

---

## Phase 2: 模型训练与转换 (在PC上运行)

### 2.1. 训练脚本

**`train.py`**

```python
import paddle
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms
from paddle.static import InputSpec

# 1. 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # 归一化到[-1, 1]
])

# 2. 加载数据集
train_dataset = DatasetFolder('data_split/train', transform=transform)
val_dataset = DatasetFolder('data_split/val', transform=transform)

# 3. 定义LeNet模型
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.flatten = paddle.nn.Flatten()
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
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
model.fit(
    train_dataset,
    val_dataset,
    epochs=10,
    batch_size=64,
    verbose=1
)

# 6. 保存模型用于部署
print("训练完成，保存推理模型...")
model.save('inference_model/lenet', training=False)
# 也可以使用以下方式，更明确地指定输入
# inputs = InputSpec([None, 1, 32, 32], 'float32', 'image')
# paddle.jit.save(model.network, 'inference_model/lenet', input_spec=[inputs])

print("推理模型已保存到 'inference_model' 目录。")
```

运行 `python train.py`。训练完成后，将在 `inference_model` 目录下生成 `lenet.pdmodel` 和 `lenet.pdiparams`。

### 2.2. 模型转换

1. 找到你的 `opt` 工具可执行文件。
2. 在命令行中执行：

    ```bash
    # 确保你的opt工具在当前路径，或者使用其绝对路径
    # 将 inference_model/ 目录下的模型进行转换
    ./opt --model_file=./inference_model/lenet.pdmodel \
          --param_file=./inference_model/lenet.pdiparams \
          --optimize_out_type=naive_buffer \
          --optimize_out=./lite_model \
          --valid_targets=arm
    ```

3. 检查结果：在 `./lite_model/` 目录下会生成 `lenet.nb` 文件，这就是我们最终需要的模型。

---

## Phase 3: 应用软件开发与部署

此阶段在**实验箱**上完成，使用`Qt Creator`。

### 3.1. 创建Qt项目

1. 打开`Qt Creator`，创建一个新的`Qt Widgets Application`项目，命名为`ArmSortApp`。
2. 在 `.pro` 文件中添加`OpenCV`和`PaddleLite`的依赖：

    ```qmake
    # ... 其他Qt模块 ...
    QT += serialport

    # 添加OpenCV
    INCLUDEPATH += /usr/include/opencv4
    LIBS += -L/usr/lib/aarch64-linux-gnu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs

    # 添加PaddleLite
    # 假设你把PaddleLite C++库解压到了 /home/linux/paddlelite
    INCLUDEPATH += /home/linux/paddlelite/include
    LIBS += -L/home/linux/paddlelite/lib -lpaddle_light_api_shared
    ```

    **注意：** 你的库路径可能不同，请根据实际情况修改。

### 3.2. 核心代码实现

以下是关键逻辑的伪代码和C++实现片段。你需要将其整合到你的Qt项目中（例如 `mainwindow.cpp`）。

**`mainwindow.h` (部分)**

```cpp
#include <opencv2/opencv.hpp>
#include "paddle_api.h" // PaddleLite头文件

class MainWindow : public QMainWindow
{
    // ...
private:
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor;
    void loadModel(const std::string& model_path);
    int predictImage(const cv::Mat& img);
    void runSortingTask();
};
```

**`mainwindow.cpp` (部分)**

```cpp
#include "mainwindow.h"

// 1. 加载模型
void MainWindow::loadModel(const std::string& model_path) {
    paddle::lite_api::MobileConfig config;
    config.set_model_from_file(model_path);
    predictor = paddle::lite_api::CreatePaddlePredictor(config);
    std::cout << "PaddleLite model loaded." << std::endl;
}

// 2. 图像预测函数
int MainWindow::predictImage(const cv::Mat& img) {
    if (!predictor) return -1;

    // 预处理
    cv::Mat gray, resized;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(32, 32));
    
    // 获取输入Tensor
    std::unique_ptr<paddle::lite_api::Tensor> input_tensor(predictor->GetInput(0));
    input_tensor->Resize({1, 1, 32, 32});
    auto* input_data = input_tensor->mutable_data<float>();
    
    // 填充数据并归一化
    for (int i = 0; i < 32 * 32; ++i) {
        input_data[i] = (resized.data[i] / 255.0f - 0.5f) / 0.5f;
    }

    // 执行预测
    predictor->Run();
    
    // 获取输出
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(predictor->GetOutput(0));
    const float* output_data = output_tensor->data<float>();
    
    // 找到最大概率的索引
    int max_idx = std::distance(output_data, std::max_element(output_data, output_data + 10));
    return max_idx;
}

// 3. 核心任务逻辑
void MainWindow::runSortingTask() {
    // a. 获取摄像头图像
    cv::Mat frame = ...; // 从摄像头捕获一帧

    // b. 定位四个仓库格子的ROI (Region of Interest)
    // 此处简化为硬编码，实际项目中应使用CV算法
    std::vector<cv::Rect> rois = {
        cv::Rect(x1, y1, w, h), cv::Rect(x2, y2, w, h),
        cv::Rect(x3, y3, w, h), cv::Rect(x4, y4, w, h)
    };

    // c. 识别并排序
    struct DetectionResult {
        int digit;
        int original_pos;
    };
    std::vector<DetectionResult> results;

    for (int i = 0; i < rois.size(); ++i) {
        cv::Mat block_img = frame(rois[i]);
        int digit = predictImage(block_img);
        if (digit != -1) { // 假设-1表示未识别或格子为空
            results.push_back({digit, i});
        }
    }

    std::sort(results.begin(), results.end(), [](const DetectionResult& a, const DetectionResult& b){
        return a.digit > b.digit; // 降序排序
    });

    // d. 生成并执行机械臂动作
    for (int i = 0; i < results.size(); ++i) {
        int from_pos = results[i].original_pos;
        int to_pos = i; // 目标位置按排序后的顺序
        
        // 发送串口指令控制机械臂
        // sendSerialCommand(from_pos, to_pos);
        std::cout << "Move block " << results[i].digit 
                  << " from pos " << from_pos 
                  << " to pos " << to_pos << std::endl;
        
        // 等待机械臂动作完成
        // QThread::msleep(5000); 
    }
}
```

### 3.3. 部署与运行

1. 将PC上训练好的 `lite_model/lenet.nb` 文件拷贝到实验箱的`ArmSortApp`项目可执行文件所在的目录。
2. 将下载的PaddleLite C++库的 `.so` 文件也拷贝到可执行文件目录，或者配置`LD_LIBRARY_PATH`。
3. 在Qt Creator中编译并运行你的项目。
4. 在界面上点击“开始排序”按钮，观察实验现象。

---

## 8. 常见问题排查（补充）

| 问题 | 可能原因 | 解决方案 |
| :--- | :--- | :--- |
| **Qt项目链接失败 (undefined reference)** | `.pro`文件中的库路径或库名称错误；未将`.so`文件放到链接路径。 | 仔细检查`.pro`文件中的`INCLUDEPATH`和`LIBS`变量，确保路径和库名称正确无误。使用`ldd ./YourAppName`命令检查依赖是否都已找到。 |
| **程序运行时闪退** | 模型文件加载失败（路径错误）；输入数据维度与模型不匹配。 | 在代码中加入日志打印和异常捕获。在调用`predictor->Run()`前后打印日志，确认路径和数据维度是否正确。 |
| **串口无响应** | 串口号、波特率等参数设置错误；权限问题。 | 确认M3控制器使用的串口号（如`/dev/ttyS1`）。在Linux下，可能需要将用户添加到`dialout`组以获取串口权限: `sudo usermod -a -G dialout your_username`。 |
