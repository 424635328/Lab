# split_dataset.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import shutil

SOURCE_DIR = "digital_raw_data"
DEST_DIR = "data_split"
SPLIT_RATIO = (0.8, 0.1, 0.1)

if os.path.exists(DEST_DIR): shutil.rmtree(DEST_DIR)

for sub_dir in ['train', 'val', 'test']:
    for i in range(10):
        os.makedirs(os.path.join(DEST_DIR, sub_dir, str(i)), exist_ok=True)

for digit in os.listdir(SOURCE_DIR):
    digit_path = os.path.join(SOURCE_DIR, digit)
    if not os.path.isdir(digit_path): continue
    
    images = [f for f in os.listdir(digit_path) if f.endswith('.jpg')]
    random.shuffle(images)
    
    train_count = int(len(images) * SPLIT_RATIO[0])
    val_count = int(len(images) * SPLIT_RATIO[1])
    
    for i, img_name in enumerate(images):
        source_img_path = os.path.join(digit_path, img_name)
        if i < train_count: dest_folder = 'train'
        elif i < train_count + val_count: dest_folder = 'val'
        else: dest_folder = 'test'
        dest_img_path = os.path.join(DEST_DIR, dest_folder, digit, img_name)
        shutil.copy(source_img_path, dest_img_path)

print(f"数据集划分完成，保存在 '{DEST_DIR}' 目录。")