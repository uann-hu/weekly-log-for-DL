# MNIST 手写数字识别

这是一个使用 PyTorch 和卷积神经网络（CNN）识别 MNIST 手写数字的入门项目。

## 项目功能

- 使用 MNIST 数据集训练 CNN
- 在测试集上评估模型准确率
- 保存训练好的模型参数
- 读取自己的手写数字图片并进行预测
- 实验比较 Dropout 和不同优化器的效果

## 项目文件

```text
main.py          # 训练和测试模型
predict.py       # 读取手写图片并预测
mnist_cnn.pt     # 训练好的模型参数
my_digit.png     # 自己的手写数字图片
requirements.txt # Python 依赖
```

## 环境

- Windows
- Python 3.11
- PyTorch
- torchvision
- NVIDIA RTX 3060（可选，用于 CUDA 加速）

## 安装依赖

建议使用 Conda 环境：

```powershell
conda create -n mnist-cnn python=3.11
conda activate mnist-cnn
pip install torch torchvision
```

## 训练模型

在项目目录下运行：

```powershell
python main.py --save-model
```

训练结束后会生成 `mnist_cnn.pt`。

## 预测自己的图片

将一张手写数字图片命名为 `my_digit.png`，放在项目目录下，然后运行：

```powershell
python predict.py
```

图片会经过灰度化、反色、裁剪、缩放和居中处理，再输入模型进行预测。

## 实验结果

在 MNIST 测试集上，模型准确率约为 99%。一次实验结果如下：

```text
Test set: Average loss: 0.0256, Accuracy: 9932/10000 (99%)
```

模型也成功识别了自己的手写数字图片。

## 学到的内容

- 卷积神经网络（CNN）
- 反向传播和梯度下降
- Dropout（暂退法）
- 学习率 `lr`
- Adam 和 Adadelta 优化器
- 图像预处理
- 模型保存、加载与推理

