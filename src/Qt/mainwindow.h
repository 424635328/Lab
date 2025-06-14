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