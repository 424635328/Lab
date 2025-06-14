# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# 文件名: collect_data.py
# 功能: 通过摄像头采集数字积木图像，并按数字分类保存
import cv2
import os
import time

# 保存图像的根目录，这里设置为桌面上的一个文件夹
SAVE_ROOT = "/home/linux/Desktop/digital_images"
# 摄像头设备索引，0通常代表默认的USB摄像头
CAMERA_INDEX = 0

# 确保根目录存在，并为0-9每个数字创建子目录
os.makedirs(SAVE_ROOT, exist_ok=True)
for i in range(10):
    os.makedirs(os.path.join(SAVE_ROOT, str(i)), exist_ok=True)

# 初始化摄像头
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("错误: 无法打开摄像头!")
    exit()

print("摄像头已启动。")
print("操作提示: 将数字积木置于镜头前，按下对应的数字键('0'-'9')进行拍照保存。")
print("按 'q' 键退出程序。")

while True:
    # 读取一帧图像
    ret, frame = cap.read()
    if not ret: 
        print("错误: 无法读取视频帧。")
        break
    
    # 创建一个用于显示的副本，并在上面添加提示文字
    display_frame = frame.copy()
    cv2.putText(display_frame, "Press 0-9 to save. Press 'q' to quit.", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Data Collection', display_frame)

    # 等待按键，1毫秒超时
    key = cv2.waitKey(1) & 0xFF
    
    # 如果按下 'q'，则退出循环
    if key == ord('q'):
        break
    
    # 如果按下的是数字键
    if ord('0') <= key <= ord('9'):
        digit = chr(key)
        save_dir = os.path.join(SAVE_ROOT, digit)
        # 使用时间戳生成唯一文件名，避免覆盖
        filename = f"{digit}_{int(time.time() * 1000)}.jpg"
        save_path = os.path.join(save_dir, filename)
        
        # 保存图像
        cv2.imwrite(save_path, frame)
        print(f"图像已保存 -> {save_path}")

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()