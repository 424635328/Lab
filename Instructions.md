# **实验指导手册**

欢迎来到人工智能教学实验箱的实战项目！本手册将手把手地引导您和您的团队，从零开始构建一个可以自动识别、排序并搬运数字积木的完整AI系统。请严格按照步骤操作，并理解每一步背后的原理。

## **第一部分：项目准备与环境搭建 (Phase 0)**

在开始编码前，我们需要确保PC（我们的“设计与训练中心”）和实验箱（我们的“执行与部署平台”）都已准备就绪。

### **步骤 0.1: 团队分工与信息核对**

1. **团队角色建议**:
    * **PC操作员 (1名)**: 负责PC上的所有操作，如环境搭建、模型训练、文件传输等。
    * **实验箱操作员 (1名)**: 负责实验箱上的所有操作，如采集数据、开发Qt应用等。
    * **物料与记录员 (1名)**: 负责摆放积木、记录实验数据（如IP地址、遇到的问题等）。

2. **关键信息核对 (重要！)**:
    * **实验箱用户名**: `linux`
    * **实验箱密码**: `1`
    * **实验箱IP地址**: 这是网络通信的“门牌号”，**现在就去获取它！**

    ---

> #### **【必备技能】如何确定实验箱的IP地址？**
    >
    > 这是所有网络操作的第一步。请**实验箱操作员**现在就完成它：
    >
    > 1. **连接硬件**: 确保实验箱已连接显示器、键盘和鼠标，并已开机进入桌面。
    > 2. **连接网络**: 确保实验箱已通过**网线**或**WiFi**连接到与PC**同一个局域网**（例如，连接到同一个路由器）。
    > 3. **打开终端**: 在实验箱桌面上，按快捷键 `Ctrl + Alt + T` 打开一个终端窗口。
    > 4. **执行命令**: 在终端中输入 `ifconfig` 并按回车。
    > 5. **查找IP**:
    >     * 如果是有线连接，在 `eth0` 条目下找到 `inet` 后面的地址（如 `192.168.1.101`）。
    >     * 如果是无线连接，在 `wlan0` 条目下找到 `inet` 后面的地址。
    >
    > **请记录下这个IP地址，后续所有步骤都会用到！**
    ---

### **步骤 0.2: PC端环境搭建 (PC操作员)**

此环境用于处理数据和训练我们的AI“大脑”。

1. **安装核心工具 (在PC终端)**:

    ```bash
    # 更新你的PC系统
    sudo apt update && sudo apt upgrade -y
    
    # 安装项目所需的所有基础工具，包括 rsync
    sudo apt install -y build-essential git python3-pip python3-venv wget curl rsync
    ```

2. **配置独立的Python环境 (在PC终端)**:
    > **为什么？** 我们的实验箱使用Python 3.5，而新PC可能是Python 3.12。为了保证训练环境与AI框架（PaddlePaddle）兼容，我们需要一个特定版本的Python。`pyenv`是管理多个Python版本的最佳工具。

    ```bash
    # 1. 安装pyenv和它的编译依赖
    sudo apt install -y make libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl
    curl https://pyenv.run | bash

    # 2. 让pyenv在你的终端里生效
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    exec "$SHELL"

    # 3. 安装并指定Python版本
    pyenv install 3.11.9
    ```

3. **创建项目并激活虚拟环境 (在PC终端)**:

    ```bash
    # 1. 创建项目文件夹
    mkdir ~/ai_arm_project
    cd ~/ai_arm_project

    # 2. 告诉pyenv，在这个文件夹里默认使用3.11.9版本
    pyenv local 3.11.9

    # 3. 创建一个隔离的Python“沙箱”（虚拟环境）
    python -m venv venv
    
    # 4. 进入这个“沙箱”
    source venv/bin/activate
    # 成功后，你的命令行提示符前会出现(venv)字样
    ```

4. **安装Python库 (在PC终端的(venv)环境下)**:

    ```bash
    pip install --upgrade pip
    pip install paddlepaddle==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install opencv-python numpy
    ```

5. **配置模型转换工具 (在PC终端)**:

    ```bash
    mkdir -p ~/tools
    cd ~/tools
    wget https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.12.0/opt_linux_x86
    chmod +x opt_linux_x86
    mv opt_linux_x86 opt
    echo 'export PATH=$PATH:~/tools' >> ~/.bashrc
    source ~/.bashrc
    opt # 验证是否成功
    ```

### **步骤 0.3: 实验箱环境搭建 (实验箱操作员)**

此环境用于运行最终的应用程序。

1. **安装基础软件 (在实验箱终端)**:

    ```bash
    sudo apt update
    sudo apt install -y qtcreator qt5-default libopencv-dev openssh-server
    ```

2. **准备AI预测库**:
    * **PC操作员**: 从[PaddleLite Release页面](https://github.com/PaddlePaddle/Paddle-Lite/releases)下载适用于 **ARMv8 (aarch64)** 的C++预测库 (`inference_lite_lib.linux.aarch64...tar.gz`)。
    * **PC操作员**: 使用`scp`将这个压缩包上传到实验箱的家目录：

        ```bash
        # 在PC终端执行，记得替换IP
        scp ./inference_lite_lib.linux.aarch64.gcc.tar.gz linux@192.168.1.101:~/
        ```

    * **实验箱操作员**: 在实验箱终端上解压它：

        ```bash
        mkdir -p ~/paddlelite_cpp
        tar -zxvf ~/inference_lite_lib.linux.aarch64.gcc.tar.gz -C ~/paddlelite_cpp
        ```

---

## **第二部分：数据采集与模型训练 (Phase 1 & 2)**

这是项目的核心AI部分，我们将在PC和实验箱之间协同工作。

### **步骤 1: 采集我们自己的数据集 (地点：实验箱)**

**目标**: 创建一个高质量、多样化的原始图像数据集。

1. **部署采集脚本 (PC操作员)**:
    * 在PC上创建一个名为 `collect_data.py` 的文件，并将下面给出的完整代码复制进去。
    * 使用`scp`将此脚本上传到**实验箱的桌面**:

        ```bash
        # 在PC终端执行，cd到脚本所在目录
        scp ./collect_data.py linux@192.168.1.101:~/Desktop/
        ```

2. **执行数据采集 (实验箱操作员 & 物料员)**:
    * **操作员**在实验箱终端运行脚本: `cd ~/Desktop/` 然后 `python3 collect_data.py`。
    * **全员配合**: 按照屏幕提示，**物料员**系统性地改变积木的**光照、位置、角度**，**操作员**在每次改变后按下对应的数字键进行拍照。
    * **质量要求**: 确保每个数字都采集**50-100张**高质量、多样化的图片。
    * **结束**: 采集完毕后，按 `q` 键退出。
    > **产出**: 实验箱桌面上生成一个 `digital_images` 文件夹。

### **步骤 2: 将数据集同步回PC (PC操作员)**

**目标**: 将“原材料”高效地运回PC进行“加工”。

1. **执行Rsync同步 (在PC终端)**:
    > **为什么用Rsync？** 因为数据集很大，如果中途需要增补图片，`rsync`只会传输新增的部分，极大地节省时间。

    ```bash
    # cd到PC的项目目录: ~/ai_arm_project/
    # 执行同步命令，将实验箱的文件夹同步到本地并重命名
    rsync -avz --progress linux@192.168.1.101:~/Desktop/digital_images/ ./digital_raw_data
    ```

2. **验证**: 在PC上检查 `digital_raw_data` 文件夹内容是否完整。

### **步骤 3: 训练并转换模型 (PC操作员)**

在PC终端，确保已激活虚拟环境 (`source venv/bin/activate`)，然后依次执行：

1. **数据划分**: 在PC项目目录中创建`split_dataset.py`文件（代码见上文），然后运行：

    ```bash
    python3 split_dataset.py
    ```

2. **模型训练**: 在PC项目目录中创建`train.py`文件（代码见上文），然后运行：

    ```bash
    python3 train.py
    ```

    > **注意**: 确认`train.py`中LeNet模型的`linear1`层输入特征为`16*6*6`。

3. **模型转换**:

    ```bash
    opt --model_file=./inference_model/lenet.pdmodel \
        --param_file=./inference_model/lenet.pdiparams \
        --optimize_out_type=naive_buffer \
        --optimize_out=./lite_model \
        --valid_targets=arm
    ```

    > **最终AI产出**: 一个名为 `lenet.nb` 的轻量级模型文件，位于 `lite_model` 文件夹中。

---

## **第三部分：应用部署与最终测试 (Phase 3)**

**目标**: 将我们的AI“大脑”植入实验箱的Qt应用程序中，让它真正动起来。

### **步骤 1: 部署最终模型 (PC操作员)**

1. **实验箱操作员**: 在实验箱上打开Qt Creator，新建一个`Qt Widgets Application`项目，命名为 `ArmSortApp`，保存在家目录 `~/` 下。
2. **PC操作员**: 使用`scp`将`lenet.nb`模型文件精确地上传到刚刚创建的Qt项目文件夹中：

    ```bash
    # 在PC终端执行，cd到 ~/ai_arm_project/lite_model/ 目录
    scp ./lenet.nb linux@192.168.1.101:~/ArmSortApp/
    ```

### **步骤 2: 开发Qt应用程序 (实验箱操作员)**

在实验箱的Qt Creator中，进行以下操作。所有代码均已提供，您只需复制、粘贴和**关键修改**。

1. **配置项目 (`.pro`文件)**: 双击`ArmSortApp.pro`，用上面提供的完整内容替换。之后，右键项目名 -> `Run qmake`。

2. **设计UI (`.ui`文件)**: 双击`mainwindow.ui`，拖入一个`QLabel`(改名为`videoLabel`)，一个`QPushButton`(`startButton`)，一个`QLabel`(`statusLabel`)。

3. **编写代码**: 将上面提供的`mainwindow.h`和`mainwindow.cpp`的完整代码，分别复制粘贴到对应的文件中。

### **步骤 3: 关键标定与最终测试 (全员)**

这是让程序在您的**特定物理环境**下正常工作的最后一步，也是最重要的一步。

1. **硬件连接**: 确保机械臂与实验箱主板的串口已连接。

2. **ROI标定 (实验箱操作员)**:
    * 打开`mainwindow.cpp`文件，找到 `initUI()` 函数。
    * 点击Qt Creator的绿色**运行**按钮。程序窗口会弹出。
    * **观察**视频画面中绿色方框的位置。它们很可能不准确。
    * **修改代码**: 根据目测，**反复修改 `warehouse1_rois` 的四个 `cv::Rect(x, y, w, h)` 的数值**，`x, y`是左上角坐标，`w, h`是宽高。
    * **重新编译运行**，直到四个绿色方框**精确地**框住“仓库一”的四个格子为止。

3. **串口配置 (实验箱操作员)**:
    * 打开`mainwindow.cpp`文件，找到 `initSerialPort()` 函数。
    * 确认 `setPortName("/dev/ttyS1")` 中的 `/dev/ttyS1` 是否是您机械臂实际使用的串口。
    * **如何确认？** 在实验箱终端运行 `ls /dev/tty*`。常见的串口名有 `ttyS0`, `ttyS1`, `ttyAMA0`, `ttyUSB0` 等。如果不确定，可以挨个尝试。

4. **最终测试**:
    * 在“仓库一”中随意摆放2-4块数字积木。
    * 点击应用程序界面上的**“开始排序”**按钮。
    * **观察**:
        * 下方的状态标签是否按“识别->排序->搬运”的流程更新信息？
        * 机械臂是否按照识别出的数字**降序**，将积木从“仓库一”搬运到“仓库二”的对应位置？
