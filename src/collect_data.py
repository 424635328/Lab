# collect_data.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os
import time

# --- 配置区 ---
DATASET_PATH = "digital_dataset"  # 数据集保存的根目录
CAMERA_INDEX = 0  # 摄像头索引号，通常为 0
# --- 配置区结束 ---

# 确保数据集目录存在
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
for i in range(10):
    dir_path = os.path.join(DATASET_PATH, str(i))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 打开摄像头
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("错误：无法打开摄像头！")
    exit()

print("摄像头已启动...")
print("按键盘上的数字 '0'-'9' 保存图像到对应文件夹。")
print("按 'q' 键退出。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("错误：无法读取视频帧！")
        break

    # 在窗口中显示实时视频流
    cv2.imshow('Data Collection - Press 0-9 to Save, Q to Quit', frame)

    key = cv2.waitKey(1) & 0xFF

    # 按 'q' 退出
    if key == ord('q'):
        break

    # 按数字键保存图像
    if ord('0') <= key <= ord('9'):
        digit = chr(key)
        save_dir = os.path.join(DATASET_PATH, digit)
        # 使用时间戳作为文件名，避免重复
        filename = f"{int(time.time() * 1000)}.jpg"
        save_path = os.path.join(save_dir, filename)
        
        # 保存图像
        cv2.imwrite(save_path, frame)
        print(f"成功保存图像 -> {save_path}")

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("采集完成！")