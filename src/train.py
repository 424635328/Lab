# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# 文件名: train.py
# 功能: 使用划分好的数据集训练一个LeNet数字分类模型
import paddle
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms

# 1. 定义数据预处理流程
transform = transforms.Compose([
    transforms.Resize((32, 32)),    # 统一图像大小为32x32
    transforms.Grayscale(),         # 转换为灰度图
    transforms.ToTensor(),          # 转换为Tensor格式
    transforms.Normalize(mean=[0.5], std=[0.5]) # 归一化到[-1, 1]范围
])

# 2. 加载数据集
train_dataset = DatasetFolder('data_split/train', transform=transform)
val_dataset = DatasetFolder('data_split/val', transform=transform)

# 3. 定义LeNet模型结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 第一个卷积-池化层
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        # 第二个卷积-池化层
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        # 展平层，用于连接全连接层
        self.flatten = paddle.nn.Flatten()
        # 三个全连接层
        self.linear1 = paddle.nn.Linear(in_features=16*6*6, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=num_classes)
        # 激活函数
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

# 4. 实例化模型，并配置优化器、损失函数和评估指标
model = paddle.Model(LeNet())
model.prepare(
    paddle.optimizer.Adam(parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy()
)

# 5. 开始训练
print("开始训练...")
model.fit(train_dataset, val_dataset, epochs=10, batch_size=64, verbose=1)

# 6. 保存模型用于部署（保存为静态图格式）
print("训练完成，保存推理模型...")
paddle.jit.save(model.network, './inference_model/lenet',
                input_spec=[paddle.static.InputSpec(shape=[None, 1, 32, 32], dtype='float32')])
print("推理模型已保存到 'inference_model' 目录。")