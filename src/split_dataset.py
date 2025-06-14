# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# 文件名: split_dataset.py
# 功能: 将原始数据集按比例划分为训练集、验证集和测试集
import os
import random
import shutil

# 源数据目录
SOURCE_DIR = "digital_raw_data"
# 目标目录
DEST_DIR = "data_split"
# 划分比例 (训练集, 验证集, 测试集)
SPLIT_RATIO = (0.8, 0.1, 0.1)

# 如果目标目录已存在，先删除，确保每次运行都是全新的划分
if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR)

# 创建目标目录结构
for sub_dir in ['train', 'val', 'test']:
    for i in range(10):
        os.makedirs(os.path.join(DEST_DIR, sub_dir, str(i)), exist_ok=True)

# 遍历每个数字类别
for digit in os.listdir(SOURCE_DIR):
    digit_path = os.path.join(SOURCE_DIR, digit)
    if not os.path.isdir(digit_path):
        continue
    
    # 获取该数字下的所有图片
    images = [f for f in os.listdir(digit_path) if f.endswith('.jpg')]
    random.shuffle(images) # 随机打乱顺序
    
    # 计算划分数量
    train_count = int(len(images) * SPLIT_RATIO[0])
    val_count = int(len(images) * SPLIT_RATIO[1])
    
    # 遍历图片并复制到对应目录
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