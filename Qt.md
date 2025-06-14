# **【实验手册】Qt交叉编译项目创建与配置指南**

本手册将详细指导您如何在PC端的Qt Creator中，利用已准备好的交叉编译工具链和Sysroot，创建一个可以远程部署到实验箱的Qt项目。

## **修复: 同步实验箱的Sysroot**

**什么是Sysroot？** 简单来说，Sysroot是目标设备（实验箱）文件系统的一个“副本”，主要包含了编译和链接程序时所需的头文件 (`/usr/include`) 和库文件 (`/usr/lib`, `/lib`)。

**为什么需要它？** 当我们在PC上交叉编译Qt程序时，程序不仅需要我们自己的代码，还需要链接实验箱上安装的Qt库、OpenCV库等。通过Sysroot，我们的交叉编译器可以在PC上就找到这些位于实验箱上的库文件和头文件。

**操作步骤 (在您的PC终端上执行):**

1.  **在PC上创建Sysroot目录**:
    ```bash
    mkdir -p ~/dev-kits/exp-box-sysroot
    ```

2.  **使用`rsync`从实验箱同步文件**:
    > 这是`rsync`大显身手的又一个场景！我们只需要同步关键的库和头文件目录。
    ```bash
    # 将实验箱的 /lib 和 /usr 目录同步到我们PC上的sysroot文件夹中
    # 请务必将IP地址替换为实际值
    rsync -avz --progress --rsync-path="sudo rsync" linux@192.168.1.101:/lib ~/dev-kits/exp-box-sysroot/
    rsync -avz --progress --rsync-path="sudo rsync" linux@192.168.1.101:/usr ~/dev-kits/exp-box-sysroot/
    ```
    *   `--rsync-path="sudo rsync"`: 这是一个高级技巧，因为实验箱上的 `/lib`, `/usr` 等目录需要`sudo`权限才能完整读取，这个参数让`rsync`在远程执行时自动带上`sudo`。

3.  **修复Sysroot中的软链接 (关键步骤)**:
    Sysroot中的许多库文件是软链接（shortcuts），它们指向的是绝对路径（如 `/lib/aarch64-linux-gnu/libm.so.6`）。这些链接在PC上是无效的。我们需要一个脚本来将它们修正为相对路径。
    *   **下载修复脚本**:
        ```bash
        wget https://raw.githubusercontent.com/riscv/riscv-poky/master/scripts/sysroot-relativelinks.py
        chmod +x sysroot-relativelinks.py
        ```
    *   **执行修复**:
        ```bash
        ./sysroot-relativelinks.py ~/dev-kits/exp-box-sysroot
        ```
    脚本会自动遍历并修复所有无效的软链接。

---

## **前期准备确认**

在开始本节内容前，请确保您已完成以下工作，并记录好相关路径：

1.  **交叉编译工具链已解压**:
    *   **确认项**: Linaro工具链已成功解压。
    *   **关键路径**: 编译器 `aarch64-linux-gnu-g++` 的绝对路径。
    *   **示例路径**: `/opt/toolchains/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++`

2.  **实验箱Sysroot已同步并修复**:
    *   **确认项**: 已使用`rsync`从实验箱同步了`/lib`和`/usr`目录，并用`sysroot-relativelinks.py`脚本修复了软链接。
    *   **关键路径**: Sysroot根目录的绝对路径。
    *   **示例路径**: `/home/your_username/dev-kits/exp-box-sysroot`

3.  **实验箱网络信息**:
    *   **IP地址**: (例如: `192.168.1.101`)
    *   **用户名**: `linux`
    *   **密码**: `1`

## **第一步：配置Qt Creator开发环境**

我们将在Qt Creator中定义三个核心元素：**设备(Device)**、**编译器(Compiler)** 和最终的 **开发套件(Kit)**。

**操作地点**: PC端的Qt Creator

### **1.1 添加实验箱设备 (Device)**

这是告诉Qt Creator我们的目标硬件在哪里。

1.  打开Qt Creator，点击菜单栏 `Tools` -> `Options...` (macOS: `Qt Creator` -> `Preferences...`)。
2.  在弹出的窗口左侧，选择 `Devices`。
3.  点击 `Add...` 按钮，在弹出的列表中选择 `Generic Linux Device`，然后点击 `Start Wizard`。
4.  填写设备连接信息：
    *   **The name to identify this configuration**: `My Experiment Box` (或任何您喜欢的名字)
    *   **The device's host name or IP address**: `192.168.1.101` (**替换为您的实际IP**)
    *   **The username to log into the device**: `linux`
    *   **The authentication method**: `Password`
    *   **The password to use for authentication**: `1`
5.  点击 `Next`，Qt Creator会尝试连接实验箱。
6.  连接成功后，您会看到“Device test finished successfully”的消息。点击 `Finish` 完成添加。
    > **成功标志**: 设备列表中的 `My Experiment Box` 图标变为绿色。

### **1.2 添加交叉编译器 (Compiler)**

这是告诉Qt Creator用哪个“工具”来编译代码。

1.  在 `Options` 窗口左侧，选择 `Kits`，然后切换到顶部的 `Compilers` 标签页。
2.  点击 `Add`，在下拉菜单中选择 `GCC`，然后选择 `C++`。
3.  配置编译器信息：
    *   **Name**: `AArch64 GCC (Linaro 7.5)` (一个清晰易懂的名字)
    *   **Compiler path**: 点击 `Browse...`，导航到您之前解压的Linaro工具链路径，选择 `bin/aarch64-linux-gnu-g++` 文件。
        *   **示例**: `/opt/toolchains/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++`
    *   **ABI**: Qt Creator通常会自动检测并填充为 `arm-linux-generic-elf-64bit`。
4.  点击 `Apply` 保存设置。

### **1.3 创建交叉编译套件 (Kit)**

这是最关键的一步，它将**设备**、**Sysroot**和**编译器**“捆绑”在一起，形成一个完整的开发环境。

1.  在 `Options` 窗口中，切换到 `Kits` 标签页。
2.  点击 `Add` 按钮，开始创建新的Kit。
3.  详细配置Kit的每一个选项：
    *   **Name**: `ARM64 for ExpBox` (一个清晰的Kit名称)
    *   **Icon**: 保持默认。
    *   **File system name**: 保持默认。
    *   **Device type**: 在下拉菜单中选择 `Generic Linux Device`。
    *   **Device**: 在下拉菜单中选择我们刚刚创建的 `My Experiment Box`。
    *   **Sysroot**: **非常重要！** 点击 `Browse...`，选择您之前准备并修复好的Sysroot目录。
        *   **示例**: `/home/your_username/dev-kits/exp-box-sysroot`
    *   **Compiler**:
        *   **C++**: 在下拉菜单中，选择我们添加的 `AArch64 GCC (Linaro 7.5)`。
        *   **C**: 同样选择对应的C编译器 `aarch64-linux-gnu-gcc`。
    *   **Debugger**: 点击 `Edit...`，在弹出的窗口中，**Path**一栏浏览并选择工具链中的`bin/aarch64-linux-gnu-gdb`。
    *   **Qt version**: Qt Creator会自动检测到您Sysroot中的Qt库。在下拉菜单中选择它（例如，可能会显示为 `Qt 5.5.1 (qt5)`）。**必须选择一个**。
    *   **Qt mkspec**: 留空或让其自动填充。它应该会自动设置为类似 `linux-aarch64-gnu-g++` 的值。
4.  点击 `OK` 保存所有配置并关闭 `Options` 窗口。

> **祝贺您！** 您已经完成了最复杂的环境配置部分。

---

## **第二步：创建并配置一个交叉编译项目**

现在，我们来创建一个新的Qt项目，并指定使用我们刚刚创建的 `ARM64 for ExpBox` Kit。

### **2.1 创建新项目**

1.  在Qt Creator主界面，点击 `File` -> `New File or Project...`。
2.  选择 `Application (Qt)` -> `Qt Widgets Application` -> `Choose...`。
3.  **Project Name**: `RemoteArmSortApp`
4.  **Create in**: 选择您在PC上存放项目的目录 (例如 `~/ai_arm_project/`)。点击 `Next`。

### **2.2 选择开发套件 (Kit Selection)**

1.  在“Kit Selection”界面，您会看到多个可用的Kit。
2.  **取消勾选** `Desktop` (或其他默认的PC Kit)。
3.  **只勾选**我们刚刚创建的 `ARM64 for ExpBox`。
4.  点击 `Next`。

### **2.3 完成向导**

1.  在“Class Information”界面，保持默认设置，点击 `Next`。
2.  在“Project Management”界面，保持默认设置，点击 `Finish`。

项目已创建完毕。现在，这个项目的所有编译、部署和运行操作都将通过我们的交叉编译套件来执行。

---

## **第三步：配置部署并运行**

最后一步是告诉Qt Creator如何将编译好的程序上传到实验箱。

### **3.1 配置部署步骤**

1.  在Qt Creator主界面左侧，点击**显示器图标**，进入 `Projects` 模式。
2.  在 `Build & Run` 下，确保 `ARM64 for ExpBox` Kit是当前激活的配置。
3.  找到 `Run` 设置（在 `Build` 设置的下方）。
4.  在 `Run` 设置中，找到 `Deployment` 部分，点击 `Add Deploy Step`，在下拉菜单中选择 `Upload files via SFTP`。
5.  配置上传规则：
    *   **Source**: `%{buildDir}/RemoteArmSortApp` (这代表编译生成的可执行文件，`RemoteArmSortApp`是您的项目名)。
    *   **Target directory**: `/home/linux/Desktop` (您希望程序被上传到实验箱的哪个目录，桌面是个好选择)。
    *   **勾选**: `Ignore 'make install' errors`。

### **3.2 最终测试：Hello, Cross-Compilation!**

1.  **编写测试代码**: 打开 `mainwindow.cpp`，在构造函数中加入一行`qDebug()`输出：
    ```cpp
    #include <QDebug>
    
    MainWindow::MainWindow(QWidget *parent)
        : QMainWindow(parent)
        , ui(new Ui::MainWindow)
    {
        ui->setupUi(this);
        qDebug() << "Hello from PC! This app is running on the Experiment Box!";
    }
    ```

2.  **点击运行**: 点击Qt Creator左下角的绿色**运行按钮**。

3.  **观察整个流程**:
    *   **PC端**: 在Qt Creator的 `Compile Output` 窗口，您会看到编译过程飞速完成。
    *   **部署**: 编译成功后，状态栏会显示“Uploading files...”。
    *   **实验箱端**: 几乎在同时，一个空白的Qt窗口会**直接在实验箱的屏幕上弹出**！
    *   **PC端**: 在Qt Creator的 `Application Output` 窗口，您会看到从实验箱回传的调试信息：
        ```
        Hello from PC! This app is running on the Experiment Box!
        ```