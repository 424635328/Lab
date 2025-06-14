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