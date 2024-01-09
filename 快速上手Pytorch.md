# **快速上手Pytorch**

## 一、使用anaconda创建python环境

> conda create -n pytorch python==3.6
>
> 解释：pytorch是环境的名称

## 二、激活环境

> conda activate pytorch
>
> 解释：pytorch是环境的名称

## 三、Python中的两大学习法宝

### 1. dir()

dir() ： 函数，能够让我们知到工具箱以及工具箱中的分隔区有什么东西

### 2. help()

help() ：函数，能够让我们知道每个工具箱是如何使用的，工具的使用的方法。

## 四、 三种python编辑器的比较

1. Python文件：所有代码一次全部执行（Shift + Enter）换行
2. Python控制台：按行执行代码（一般用于调试）
3. Jupyter笔记：任意行代码为块进行运行，后面代码的执行不影响前面的代码

## 五、 Pytroch加载数据

**Pytorch中有关于加载数据有两个类：**

### 1. Dataset

Dataset ：提供一种方式取获取数据及其label。

> 如何获取每一个数据及其label
>
> 告诉我们总共有多少的数据

### 2. Dataloader

Dataloader ：为后面的网络提供不同的数据形式。

> 获取帮助文档：help(Dataset) 或 Dataset??
>
> 常见的数据存储类型： 
> **————————————————————**
>
> ​		数据根目录
> ​		|— train文件
> ​			|— 第一类文件（名称，label）
> ​			|— 第二类文件 (名称，label)
>
> ​		|—test文件
> **————————————————————**
>
> ​		数据根目录
> ​		|— train文件
> ​			|— 数据文件一
> ​			|— 标签文件一
> ​			|— 数据文件二
> ​			|— 标签文件二
> ​		|—test 文件
>
> **（其中数据文件只存放数据标签文件存放数据文件名字对应的label）**

### Practice （Dataset的使用）：

```python
from PIL import Image    # 导入python中处理图片的库
from torch.utils.data import Dataset
import os   # 与处理文件路径相关的库

# 第一类数据存储的读取
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir   # python是不需要定义变量的，直接使用
        self.label_dir = label_dir   # self表示类变量，整个类中都可以使用
        self.path = os.path.join(self.root_dir, self.label_dir)  # 得到路径根路径与标签路径进行拼接
        self.img_path = os.listdir(self.path)   # 得到相应路径下的图片，并构建为一个列表

    # 返回图片与标签（参数idx为传入的索引）
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 拼接根路径，标签路径， 图片名称
        img = Image.open(img_item_path)  # 得到具体的图片（打开图片存放的路径）
        label = self.label_dir
        return img, label

    # 返回图片长度（魔法函数：可以指定python内置库中定义的方法）
    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":
    root_dir = "../data_set/train"  # 相对路径，返回到本文件的上一级文件
    ants_label_dir = "ants"
    bees_label_dir = "bees"
    # 创建蚂蚁实例
    ants_dataset = MyData(root_dir, ants_label_dir)
    # 创建蜜蜂实例
    bees_dataset = MyData(root_dir, bees_label_dir)
    # 将蚂蚁实例与蜜蜂实例进行拼接
    train_dataset = ants_dataset + bees_dataset
    # 使用索引会直接调用魔法函数__getitem__，返回两个值
    img, label = ants_dataset[15]
    # img.show()
    # 使用len会直接调用魔法函数__len__,返回图片长度
    print(len(train_dataset))
```


> 如何查看log中的图片数据？
>
> 在当前控制台中输入： **tensorboard --logdir=logs --port=6007**
> 解释： --logdir 指定了log文件夹的路径（可以使用相对路径）  --port 指定了端口号
>
> 然后在控制台中可以看到打开的网址 ： 一般为 127.0.0.0.1/6007

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")  # 指定存储文件的目录名称
# writer.add_image()  # 添加图片
# writer.add_scalar()  # 添加坐标图像
# writer.close()  # 关闭文件
for i in range(100):
    writer.add_scalar("y = x^2", i*i, i)  # 添加图像文件：参数分别为：名称，y轴，x轴

writer.close()
```

```python
add_image(tag, img_tensor, global_step=None, walltimie=None, dataformats='CHW')
'''
参数解释：
tag 代表图像标签
img_tensor 表示图片数据（数据类型为：torch.Tensor\ numpy.array\ string\ blobname）
global_step 表示迭代的步骤
dataformates 表示img_tensor中的每一维表示什么含义
opencv 读取的数据类型是numpy 类型
'''
```

### （一）Practice  (Tensorboard的使用 )：

```python
from torch.utils.tensorboard import SummaryWriter   # 用于存储数据
import numpy as np   # 导入numpy库
from PIL import Image

writer = SummaryWriter("logs")    # 指定文件存放的文件夹，如果没有则自动创建
image_path = "../data_set/train/ants/28847243_e79fe052cd.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)   # 使用numpy将图片数据转换为numpy.array数据类型
writer.add_image("test", img_array, 2, dataformats="HWC")  # 分别表示:标签， 图片源， 迭代步数， 维度含义(维度即图片的三个维度：高 宽 通道)

writer.close()   # 关闭通道
```



### （一）Practice  (Tensorboard的使用 )：

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "../data_set/train/bees/196658222_3fffd79c67.jpg"
img = Image.open(img_path)
writer = SummaryWriter("logs")

tensor_trains = transforms.ToTensor()  #  转换为tensor 类型
tensor_img = tensor_trains(img)

writer.add_image("Tensor_img", tensor_img)   # 添加图片
writer.close()
```

## 六、Transform

### （一）常见的Transform

Normalize(归一化)

```python
# 归一化的类的定义：
class Normalize(object):
# 需要传入两个列表, [M1,...,Mn], [S1,...,Sn] 代表各个维度的均值和方差，"n":表示有n个通道
# 实现方式：input[channel] = (input[channel] - mean[channel]) / std[channel]
```
- Practice 

``` python
# Normalize(归一化)
trans_norm = transforms.Normalize([0.2, 0.3, 0.5], [0.6, 0.2, 1])
img_norm = trans_norm(tensor_img)
writer.add_image("Normalize", img_norm, 4)
```

### （二）常见的Transform

Resize(等比缩放)

- Practice

```python
# Resize(图片放缩)
train_resize = transforms.Resize(218)
img_rasize = train_resize(img)
img_rasize_tensor = tensor_train(img_rasize)
writer.add_image("Resize", img_rasize_tensor, 3)
```

Compose(等比缩放，并且转换为tensor类型)

- Practice

```python
# Compose
train_resize_2 = transforms.Resize(2048)
train_compose = transforms.Compose([train_resize_2, tensor_train])
img_resize_2 = train_compose(img)
writer.add_image("Compose", img_resize_2, 2)
```

RandomCrop (随机等比裁剪)

- Practice

``` python
# RandomCrop(随机裁剪)
trans_random = transforms.RandomCrop(128)
trans_compose_2 = transforms.Compose([trans_random, tensor_train])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)
```

## 七、torchvision 中的数据集的使用（图像数据）

有pytorch中提供的一些标准的数据集，可在官方网页进行查看

- Practice

``` python
import torchvision
from torch.utils.tensorboard import SummaryWriter

# 设定一个tensorflow的数据类型转换
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 对pytorch中标准的数据进行下载，如果指定的路径下已经存在了数据集，则直接进行使用，train参数用于指定那些是训练集，那些是测试集
train_set = torchvision.datasets.CIFAR10(root=".\dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root=".\dataset", train=False,transform=dataset_transform, download=True)

# 对图片文件进行展示
writer = SummaryWriter(".\p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
```

## 八、 Dataloader 的使用

Dataloader 用于加载数据 

- Practice

```python
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# DataLoader的使用
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoc in range(2):
    step = 0
    for data in test_loader:
        img, targets = data
        writer.add_images("EpocTE {}".format(epoc), img, step)
        step += 1
writer.close()
```

## 九、 神经网络的基本架构

- Practice
``` python
# nn.module的使用
import torch
from torch import nn
class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


tudui = Module()
x = torch.tensor(1.0)   # 普通的相加计算,tensor是一个张量，也可以说是多维数组，与numpy不同，tensor可以在GPU上运行
out = tudui(x)
print(out)
```

## 十、Pytorch卷积操作

```python
import torch
import torch.nn.functional as F

# 定义输入图像
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
# 定义卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])


# 改变输入图像的形状
input = torch.reshape(input, (1, 1, 5, 5))  # 其中（1， 1， 5， 5）分别代表batchsize、 通道数、 宽度和高度, 执行这个步骤相当于给图片和卷积核添加了两个维度
kernel = torch.reshape(kernel, (1, 1, 3, 3))
# print(input.shape)
# print(kernel.shape)

output = F.conv2d(input, kernel, stride=1)
print(output)
# 改变卷积核的移动的步长为2
output2 = F.conv2d(input, kernel, stride=2)
print(output2)
# 卷积核的移动步长为2，并且在原始图像中填充一圈
output3 = F.conv2d(input, kernel, stride=2, padding=1)
print(output3)

```

## 十一、卷积神经网络中的各个层的定义

### 一、卷积层

``` python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()
writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, target = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 32, 32))  # 因为图片有固定的三个通道，但是这里的通道数量为6，因此要进行重新设定通道大小，将batchsize设置为-1，表示根据后面调整后的大小而进行自动设置，最后两个参数分别为图片的高度和宽度
    writer.add_images("output", output, step)
    step += 1

writer.close()
```

> **计算卷积后图像宽和高的公式：**
>
> $Input:(N, C_{in}, H_{in}, W_{in})$
>
> $Output:(N, C_{out}, H_{out}, W_{out})$
>
> $H_{out} = [\frac{H_{in} + 2 \times padding[0] - dilation[0] \times (kernel\_size[0] - 1) - 1}{stride[0]} + 1]$
>
> $W_{out} = [\frac{W_{in} + 2 \times padding[1] - dilation[1] \times (kernel\_size[0] - 1) - 1}{stride[1]} + 1]$

### 二、池化层

``` python
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor(), download=True)  # torchvision.transforms.ToTensor() 这里加上括号表示实例化
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


tudui = Tudui()
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("inputl", imgs, step)
    output = tudui(imgs)
    writer.add_images("outputl", output, step)
    step += 1

writer.close()
print("OK")
```

### 四、 线性层

```python
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("data", False, torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, 64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear = Linear(196608, 10)

    def forward(self, input):
        output = self.linear(input)
        return output

tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    print(output.shape)
    output = tudui(output)
    print(output.shape)
print("ok")
```

### 五、非线性激活层

```python
import torch
import torchvision
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("data", False, torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, 64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output

tudui = Tudui()
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input_relu", imgs, step)
    output = tudui(imgs)
    writer.add_images("output_relu", output, step)
    step += 1

writer.close()
print("OK")
```

## 十二、项目实战和Sequential的使用

### (一)反向传播

```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)  # torchvision.transforms.ToTensor() 这里加上括号表示实例化
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()  # 计算损失
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)  # 设置优化器

# 进行训练
for epoc in range(100):    # 设置迭代次数为100次，此时的最好的损失累加和为2.9519
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)   # 计算损失
        optim.zero_grad()   # 将梯度置为0
        result_loss.backward()   # 设置梯度，进行反向传播
        optim.step()   # 进行优化参数
        running_loss = running_loss + result_loss  # 累加损失
    print(running_loss)
```

### (二)  实战一：

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)   # 最后定义输出层为输出10个量

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

tudui = Tudui()
input = torch.ones([64, 3, 32, 32])
output = tudui(input)
print(output.shape)

```

### (三) 实战二：

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

tudui = Tudui()
print(tudui)
input = torch.ones([64, 3, 32, 32])
output = tudui(input)
print(output.shape)


# 可视化网路操作
writer = SummaryWriter("logs")
writer.add_graph(tudui, input)
writer.close()
```

## 十三、完整的项目

### （一） main.py

```python
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from model import *


# 准备数据
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 得到数据的数量
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为{}".format(train_data_size))
print("测试数据集的长度为{}".format(test_data_size))

# 利用dataloader来加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)
# 设置网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
# 添加tensorboard
writer = SummaryWriter("logs")
for i in range(epoch):
    print("-------------------第{}轮训练开始了----------------".format(i + 1))
    # 训练步骤开始
    tudui.train()  # 这行代码只对特定的层有作用
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}， loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_loss = total_test_loss + 1

    torch.save(tudui, "tudui{}.pth".format(i))
    print("模型已保存")

writer.close()
```

###  (二) model.py

```python
import torch
from torch import nn

# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.module(x)
        return x

if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output.shape)
```

