# 第 8 章 著名数据集及基准

作者: 张伟 (Charmve)

日期: 2021/06/06

- 第 8 章 [著名数据集及基准](https://charmve.github.io/computer-vision-in-action/#/chapter6/chapter6)
    - 8.1 数据集
        - 8.1.1 [ImageNet](https://image-net.org/)
        - 8.1.2 [MNIST](http://yann.lecun.com/exdb/mnist/)
        - 8.1.3 [COCO](https://cocodataset.org/)
        - 8.1.4 [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
    - 8.2 基准
    - 小结
    - 参考文献


## 8.1 数据集

### 8.1.1 [ImageNet](https://image-net.org/)


### 8.1.2 [MNIST](http://yann.lecun.com/exdb/mnist/)


### 8.1.3 [COCO](https://cocodataset.org/)


### 8.1.4 CIFAR-10：CIFAR-10数据集简介、下载、使用方法之详细攻略

#### 8.1.4.1 CIFAR-10简介

官网链接：[The CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)

![image](https://user-images.githubusercontent.com/29084184/120912825-164e0180-c6c5-11eb-9d4c-bb099d9498ec.png)

图8.1 CIFAR-10数据集

CIFAR-10是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。
每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。


(1) 与MNIST 数据集中目比， CIFAR-10 真高以下不同点:

- CIFAR-10 是3 通道的**彩色RGB 图像**，而MNIST 是**灰度图像**。
- CIFAR-10 的图片尺寸为**32 × 32 **， 而MNIST 的图片尺寸为28 × 28 ，比MNIST 稍大。
- 相比于手写字符， CIFAR-10 含有的是**现实世界中真实的物体**，不仅噪声很大，而且物体的比例、特征都不尽相同，这为识别带来很大困难。直接的线性模型如Softmax 在CIFAR-10 上表现得很差。
 

(2) TensorFlow 官方示例的CIFAR-10 代码文件

表8.1 TensorFlow官方示例的CIFAR-10代码文件

| 文件 | 用途 |
| -- | -- |
| cifar10.py |  建立CIFAR-10预测模型 |
| cifar10_input.py | 在TensorFlow 中读入CIFAR-10 训练文件 |
| cifar10_input_test.py | cifar10_input.py 的测试用例文件 |
| cifar10_train.py | 使用单个 GPU 或 CPU 训练模型 |
| cifar10_train_mutil_gpu.py | 使用多个 GPU 训练模型 |
| cifar10_cval.py | 在测试集上测试模型的性能 |
 

(3) CIFAR-10 数据集的数据文件名及用途

在CIFAR-10 数据集中，文件``data_batch_1.bin``、``data_batch_2.bin`` 、``··data_batch_5.bin`` 和``test_ batch.bin`` 中各有10000 个样本。一个样本由3073 个字节组成，第一个字节为标签``label`` ，剩下3072 个字节为图像数据。样本和样本之间没高多余的字节分割， 因此这几个二进制文件的大小都是30730000 字节。

表8.2 CIFAR-10 数据集的数据文件名及用途

| 文件名 | 文件用途 |
| -- | -- |
| batches.meta. bet | 文件存储了每个类别的英文名称。可以用记事本或其他文本文件阅读器打开浏览查看 | 
| data batch 1.bin、data batch 2.bin 、……、data batch 5.bin | 这5 个文件是CIFAR- 10 数据集中的训练数据。每个文件以二进制格式存储了10000 张32 × 32 的彩色图像和这些图像对应的类别标签。一共50000 张训练图像 |
| test batch.bin | 这个文件存储的是测试图像和测试图像的标签。一共10000 张 |
| readme.html | 数据集介绍文件|


#### 8.1.4.2 CIFAR-10下载

下载CIFAR-10 数据集的全部数据。

```shell
FLAGS = tf.app.flags.FLAGS          
cifar10.maybe_download_and_extract() 
 
 
>> Downloading cifar-10-binary.tar.gz 0.0%
……
>> Downloading cifar-10-binary.tar.gz 0.0%
>> Downloading cifar-10-binary.tar.gz 0.1%
……
>> Downloading cifar-10-binary.tar.gz 0.1%
>> Downloading cifar-10-binary.tar.gz 0.2%
……
>> Downloading cifar-10-binary.tar.gz 0.2%
>> Downloading cifar-10-binary.tar.gz 0.3%
……
>> Downloading cifar-10-binary.tar.gz 98.9%
……
>> Downloading cifar-10-binary.tar.gz 99.0%
……
>> Downloading cifar-10-binary.tar.gz 100.0%
Successfully downloaded cifar-10-binary.tar.gz 170052171 bytes.
```

#### 8.1.4.3 CIFAR-10使用方法

使用TF读取CIFAR-10 数据。

1. 用 ``tf.train.string_ input producer`` 建立队列。
2. 通过 ``reader.read`` 读数据。一个文件就是一张图片，因此用的 reader 是``tf.WholeFileReader()``。CIFAR-10 数据是以固定字节存在文件中的，一个文件中含再多个样本，因此不能使用``tf. WholeFileReader()``，而是用``tf.FixedLengthRecordReader()``。
3. 调用``tf. train.start_ queue_ runners`` 。
4. 最后，通过``sess.run()``取出图片结果。
