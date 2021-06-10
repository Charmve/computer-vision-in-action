<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

# 第 2 章 卷积神经网络

作者: 张伟 (Charmve)

日期: 2021/05/22

<p align="center">
    <a href="https://colab.research.google.com/github/Charmve/computer-vision-in-action/blob/main/notebooks/08_Neural_Networks.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" align="center" alt="Open in Colab">
    </a>
</p>

## 目录

  - 第 2 章 [卷积神经网络](https://charmve.github.io/computer-vision-in-action/#/chapter3/chapter3)
    - [2.1 从神经网络到卷积神经网络](#21-从神经网络到卷积神经网络)
      - [2.1.1 定义](#211-定义)
      - [2.1.2 卷积神经网络的架构](#212-卷积神经网络的架构)
    - [2.2 卷积网络的层级结构](#22-卷积网络的层级结构)
      - [2.2.1 数据输入层](#221-数据输入层)
      - [2.2.2 卷积计算层](#222-卷积计算层)
        - [(1) 卷积的计算](#1-卷积的计算)
        - [(2) 参数共享机制](#2-参数共享机制)
      - [2.2.3 非线性层（或激活层）](#223-非线性层或激活层)
      - [2.2.4 池化层](#224-池化层)
      - [2.2.5 全连接层](#225-全连接层)
    - [2.3 卷积神经网络的几点说明](#23-卷积神经网络的几点说明)
      - [2.3.1 训练算法](#231-训练算法)
      - [2.3.2 优缺点](#232-优缺点)
      - [2.3.3 典型CNN](#233-典型CNN)
      - [2.3.4 fine-tuning](#234-fine-tuning)
      - [2.3.5 常用框架](#235-常用框架)
    - 2.4 [实战项目 2 - 动手搭建一个卷积神经网络](chapter2_CNN-in-Action.md)
      - [2.4.1 卷积神经网络的前向传播](chapter2_CNN-in-Action.md#271-卷积神经网络的前向传播)
      - [2.4.2 卷积神经网络的反向传播](chapter2_CNN-in-Action.md#272-卷积神经网络的反向传播)
      - [2.4.3 手写一个卷积神经网络](chapter2_CNN-in-Action.md#273-手写一个卷积神经网络)
        - [1. 定义一个卷积层](chapter2_CNN-in-Action.md#1-定义一个卷积层)
        - [2. 构造一个激活函数](chapter2_CNN-in-Action.md#2-构造一个激活函数)
        - [3. 定义一个类，保存卷积层的参数和梯度](#3-定义一个类保存卷积层的参数和梯度)
        - [4. 卷积层的前向传播](chapter2_CNN-in-Action.md#4-卷积层的前向传播)
        - [5. 卷积层的反向传播](chapter2_CNN-in-Action.md#5-卷积层的反向传播)
        - [6. MaxPooling层的训练](chapter2_CNN-in-Action.md#6-MaxPooling层的训练)
      - [2.4.4 PaddlePaddle卷积神经网络源码解析](chapter2_CNN-in-Action.md#274-PaddlePaddle卷积神经网络源码解析)
    - [小结](#小结)
    - [参考文献](#参考文献)


## 2.1 从神经网络到卷积神经网络

我们知道神经网络的结构是这样的：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015815783.png#pic_center)

图1 神经网络结构

**那卷积神经网络跟它是什么关系呢？**


其实卷积神经网络依旧是层级网络，只是层的功能和形式做了变化，可以说是传统神经网络的一个改进。比如下图中就多了许多传统神经网络没有的层次。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015824549.png#pic_center)

图2 卷积神经网络的层级网络

### 2.1.1 定义
简而言之，卷积神经网络（Convolutional Neural Networks,CNN）是一种深度学习模型或类似于人工神经网络的多层感知器，常用来分析视觉图像。卷积神经网络的创始人是着名的计算机科学家 Yann LeCun <sup>1</sup>，目前在 Facebook 工作，他是第一个通过卷积神经网络在 MNIST 数据集上解决手写数字问题的人。

### 2.1.2 卷积神经网络的架构

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015847839.png#pic_center)

图3 卷积神经网络的层级网络

如上图所示，卷积神经网络架构与常规人工神经网络架构非常相似，特别是在网络的最后一层，即全连接。此外，还注意到卷积神经网络能够接受多个特征图作为输入，而不是向量。

## 2.2 卷积网络的层级结构

一个卷积神经网络主要由以下5层组成：
- 数据输入层 / Input layer   
- 卷积计算层 / CONV layer   
- ReLU激励层 / ReLU layer  
- 池化层 / Pooling layer   
- 全连接层 / FC layer

### 2.2.1 数据输入层
该层要做的处理主要是对原始图像数据进行预处理，其中包括：
- **去均值**：把输入数据各个维度都中心化为0，如下图所示，其目的就是把样本的中心拉回到坐标系原点上。
- **归一化**：幅度归一化到同样的范围，如下所示，即减少各维度数据取值范围的差异而带来的干扰，比如，我们有两个维度的特征A和B，A范围是0到10，而B范围是0到10000，如果直接使用这两个特征是有问题的，好的做法就是归一化，即A和B的数据都变为0到1的范围。
- **PCA/白化**：用PCA降维；白化是对数据各个特征轴上的幅度归一化

去均值与归一化效果图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015852202.png#pic_center)

图4 去均值与归一化效果图

去相关与白化效果图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015857896.png#pic_center)

图5 去相关与白化效果图

### 2.2.2 卷积计算层
这一层就是卷积神经网络最重要的一个层次，也是“卷积神经网络”的名字来源。
在这个卷积层，有两个关键操作：

- **局部关联**。每个神经元看做一个滤波器(filter)

- **窗口(receptive field)滑动**， filter对局部数据计算


先介绍卷积层遇到的几个名词：

- **深度/depth**（解释见下图）

- **步幅/stride** （窗口一次滑动的长度）

- **填充值/zero-padding**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015902721.png#pic_center)

图6 卷积层的深度

还记得我们在第一篇中提到的过滤器、感受野和卷积吗？很好。现在，要改变每一层的行为，有两个主要参数是我们可以调整的。选择了过滤器的尺寸以后，我们还需要选择步幅（stride）和填充（padding）。

**步幅** 控制着过滤器围绕输入内容进行卷积计算的方式。在第一部分我们举的例子中，过滤器通过每次移动一个单元的方式对输入内容进行卷积。过滤器移动的距离就是步幅。在那个例子中，步幅被默认设置为1。步幅的设置通常要确保输出内容是一个整数而非分数。让我们看一个例子。想象一个 7 x 7 的输入图像，一个 3 x 3 过滤器（简单起见不考虑第三个维度），步幅为 1。这是一种惯常的情况。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015909844.png#pic_center)

图7 步幅为 1

还是老一套，对吧？看你能不能试着猜出如果步幅增加到 2，输出内容会怎么样。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015916403.png#pic_center)

图8 步幅为 2


所以，正如你能想到的，感受野移动了两个单元，输出内容同样也会减小。注意，如果试图把我们的步幅设置成 3，那我们就会难以调节间距并确保感受野与输入图像匹配。正常情况下，程序员如果想让接受域重叠得更少并且想要更小的**空间维度（spatial dimensions）** 时，他们会增加步幅。



**填充值是什么呢？**

在此之前，想象一个场景：当你把 5 x 5 x 3 的过滤器用在 32 x 32 x 3 的输入上时，会发生什么？输出的大小会是 28 x 28 x 3。注意，这里空间维度减小了。如果我们继续用卷积层，尺寸减小的速度就会超过我们的期望。在网络的早期层中，我们想要尽可能多地保留原始输入内容的信息，这样我们就能提取出那些低层的特征。比如说我们想要应用同样的卷积层，但又想让输出量维持为 32 x 32 x 3 。为做到这点，我们可以对这个层应用大小为 2 的零填充（zero padding）。零填充在输入内容的边界周围补充零。如果我们用两个零填充，就会得到一个 36 x 36 x 3 的输入卷。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015926148.png#pic_center)

图9 填充值


如果我们在输入内容的周围应用两次零填充，那么输入量就为 32×32×3。然后，当我们应用带有 3 个 5×5×3 的过滤器，以 1 的步幅进行处理时，我们也可以得到一个 32×32×3 的输出。


如果你的步幅为 1，而且把零填充设置为

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015937835.png#pic_center)


K 是过滤器尺寸，那么输入和输出内容就总能保持一致的空间维度。


计算任意给定卷积层的输出的大小的公式是

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015944760.png#pic_center)


其中 O 是输出尺寸，K 是过滤器尺寸，P 是填充，S 是步幅。

#### (1) 卷积的计算
（注意，下面蓝色矩阵周围有一圈灰色的框，那些就是上面所说到的填充值）


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015951209.png#pic_center)

图10 卷积的计算

这里的蓝色矩阵就是输入的图像，粉色矩阵就是卷积层的神经元，这里表示了有两个神经元（w0,w1）。绿色矩阵就是经过卷积运算后的输出矩阵，这里的步长设置为2。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315015959625.png#pic_center)


蓝色的矩阵(输入图像)对粉色的矩阵（filter）进行矩阵内积计算并将三个内积运算的结果与偏置值b相加（比如上面图的计算：2+（-2+1-2）+（1-2-2） + 1= 2 - 3 - 3 + 1 = -3），计算后的值就是绿框矩阵的一个元素。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315020008164.png#pic_center)


下面的动态图形象地展示了卷积层的计算过程：
<table>
 <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/29084184/119461753-95147780-bd72-11eb-93d9-c12db5fbdfc5.png"/>
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/29084184/119461650-7e6e2080-bd72-11eb-989f-60f3ae3e423c.png"/>
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/29084184/119461671-84640180-bd72-11eb-9cc3-3fea23b201ca.png"/>
    </td>
 </tr>
 <tr>
    <td>
      (a)
    </td>
    <td>
      (b)
    </td>
    <td>
      (c)
    </td>
 </tr>
</table>

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315020043502.gif#pic_center)

图11 卷积层的计算过程

#### (2) 参数共享机制

在卷积层中每个神经元连接数据窗的权重是固定的，每个神经元只关注一个特性。神经元就是图像处理中的滤波器，比如边缘检测专用的Sobel滤波器，即卷积层的每个滤波器都会有自己所关注一个图像特征，比如垂直边缘，水平边缘，颜色，纹理等等，这些所有神经元加起来就好比就是整张图像的特征提取器集合。

需要估算的权重个数减少: AlexNet 1亿 => 3.5w

一组固定的权重和不同窗口内数据做内积: 卷积

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315020057448.png#pic_center)

图12 参数共享机制

### 2.2.3 非线性层（或激活层）
把卷积层输出结果做非线性映射。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315020104203.png#pic_center)

图13 激活层

CNN采用的激活函数一般为ReLU(The Rectified Linear Unit/修正线性单元)，它的特点是收敛快，求梯度简单，但较脆弱，图像如下。更多关于激活函数的内容请看 <a href ="../../附件/附件-B 常用激活函数总结.md">附件-B常用激活函数总结 </a>。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315020109940.png#pic_center)

图14 激活函数: ReLU

**激励层的实践经验：**
    ①不要用sigmoid！不要用sigmoid！不要用sigmoid！
    ② 首先试RELU，因为快，但要小心点
    ③ 如果2失效，请用Leaky ReLU或者Maxout
    ④ 某些情况下tanh倒是有不错的结果，但是很少

> 参见 Geoffrey Hinton（即深度学习之父）的论文：[Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) **墙裂推荐此论文！** 

### 2.2.4 池化层
池化层夹在连续的卷积层中间， 用于压缩数据和参数的量，减小过拟合。
简而言之，**如果输入是图像的话，那么池化层的最主要作用就是压缩图像。**

这里再展开叙述池化层的具体作用：
1. **特征不变性**，也就是我们在图像处理中经常提到的特征的尺度不变性，池化操作就是图像的resize，平时一张狗的图像被缩小了一倍我们还能认出这是一张狗的照片，这说明这张图像中仍保留着狗最重要的特征，我们一看就能判断图像中画的是一只狗，图像压缩时去掉的信息只是一些无关紧要的信息，而留下的信息则是具有尺度不变性的特征，是最能表达图像的特征。
2. **特征降维**，我们知道一幅图像含有的信息是很大的，特征也很多，但是有些信息对于我们做图像任务时没有太多用途或者有重复，我们可以把这类冗余信息去除，把最重要的特征抽取出来，这也是池化操作的一大作用。
3. 在一定程度上**防止过拟合**，更方便优化。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315020122740.png#pic_center)

图15 过拟合

池化层用的方法有 Max pooling 和 average pooling，而实际用的较多的是Max pooling。这里就说一下 Max pooling，其实思想非常简单。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315020126941.png#pic_center)

图16 池化层: Max pooling

对于每个2 * 2的窗口选出最大的数作为输出矩阵的相应元素的值，比如输入矩阵第一个2 * 2窗口中最大的数是6，那么输出矩阵的第一个元素就是6，如此类推。

### 2.2.5 全连接层
两层之间所有神经元都有权重连接，通常全连接层在卷积神经网络尾部。也就是跟传统的神经网络神经元的连接方式是一样的：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315020132591.png#pic_center)

图17 全连接层

一般CNN结构依次为:

1. INPUT
2. [[CONV -> RELU]N -> POOL?]M
3. [FC -> RELU]*K
4. FC


## 2.3 卷积神经网络的几点说明

### 2.3.1 训练算法
1.同一般机器学习算法，先定义Loss function，衡量和实际结果之间差距。
2.找到最小化损失函数的W和b， CNN中用的算法是SGD（随机梯度下降）。

### 2.3.2 优缺点
- 优点
  - 共享卷积核，对高维数据处理无压力
  - 无需手动选取特征，训练好权重，即得特征分类效果好
- 缺点
  - 需要调参，需要大样本量，训练最好要GPU
  - 物理含义不明确（也就说，我们并不知道没个卷积层到底提取到的是什么特征，而且神经网络本身就是一种难以解释的“黑箱模型”）

### 2.3.3 典型CNN

这部分将在 [第7章 经典卷积神经网络架构：原理与PyTorch实现](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现) 详细讲解。

- LeNet，这是最早用于数字识别的CNN

- AlexNet， 2012 ILSVRC比赛远超第2名的CNN，比

- LeNet更深，用多层小卷积层叠加替换单大卷积层。

- ZF Net， 2013 ILSVRC比赛冠军

- GoogLeNet， 2014 ILSVRC比赛冠军

- VGGNet， 2014 ILSVRC比赛中的模型，图像识别略差于GoogLeNet，但是在很多图像转化学习问题(比如object detection)上效果奇好


### 2.3.4  fine-tuning

**何谓fine-tuning？**

fine-tuning就是使用已用于其他目标、预训练好模型的权重或者部分权重，作为初始值开始训练。

那为什么我们不用随机选取选几个数作为权重初始值？原因很简单，第一，自己从头训练卷积神经网络容易出现问题；第二，fine-tuning能很快收敛到一个较理想的状态，省时又省心。

**那fine-tuning的具体做法是？**

- 复用相同层的权重，新定义层取随机权重初始值
- 调大新定义层的的学习率，调小复用层学习率


### 2.3.5 常用框架

**Caffe**
- 源于Berkeley的主流CV工具包，支持C++,python,matlab
- Model Zoo中有大量预训练好的模型供使用

**PyTorch**
- Facebook用的卷积神经网络工具包
- 通过时域卷积的本地接口，使用非常直观
- 定义新网络层简单

**TensorFlow**
- Google的深度学习框架
- TensorBoard可视化很方便
- 数据和模型并行化好，速度快

## 小结

卷积网络在本质上是一种输入到输出的映射，它能够学习大量的输入与输出之间的映射关系，而不需要任何输入和输出之间的精确的数学表达式，只要用已知的模式对卷积网络加以训练，网络就具有输入输出对之间的映射能力。

CNN一个非常重要的特点就是头重脚轻（越往输入权值越小，越往输出权值越多），呈现出一个倒三角的形态，这就很好地避免了BP神经网络中反向传播的时候梯度损失得太快。

卷积神经网络CNN主要用来识别位移、缩放及其他形式扭曲不变性的二维图形。由于CNN的特征检测层通过训练数据进行学习，所以在使用CNN时，避免了显式的特征抽取，而隐式地从训练数据中进行学习；再者由于同一特征映射面上的神经元权值相同，所以网络可以并行学习，这也是卷积网络相对于神经元彼此相连网络的一大优势。卷积神经网络以其局部权值共享的特殊结构在语音识别和图像处理方面有着独特的优越性，其布局更接近于实际的生物神经网络，权值共享降低了网络的复杂性，特别是多维输入向量的图像可以直接输入网络这一特点避免了特征提取和分类过程中数据重建的复杂度。

## 参考文献

[1] LeCun, Yann, et al. "Handwritten digit recognition with a back-propagation network." Advances in neural information processing systems. 1990.

[2] Nair, Vinod, and Geoffrey E. Hinton. "Rectified linear units improve restricted boltzmann machines." Icml. 2010.

<br>
<br>

<table align="center">
<tr>
<td>
<code>全面</code>&nbsp;<code>前沿</code>&nbsp;<code>免费</code>
<h1> 计算机视觉实战演练：算法与应用 <sup> 📌</sup>
<br><em>Computer Vision in Action</em></h1>

作者：张伟（Charmve）

<p align="center">
<a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/👓-Charmve-blue&logo=GitHub" alt="GitHub"></a>
<a href="https://github.com/Charmve/computer-vision-in-action"><img src="https://img.shields.io/badge/CV-Action-yellow" alt="CV-Action"></a>
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>
<a href="https://github.com/Charmve/computer-vision-in-action/edit/master/README.md"><img src="https://img.shields.io/github/stars/Charmve/computer-vision-in-action?style=social" alt="Stars"></a>
<a href="https://github.com/Charmve/computer-vision-in-action/edit/master/README.md"><img src="https://img.shields.io/github/forks/Charmve/computer-vision-in-action?style=social" alt="Forks"></a>
</p>

<div align="center">
	<img src="https://github.com/Charmve/computer-vision-in-action/blob/main/res/maiwei.png" width="220px" alt="logo:maiwei" title="有疑问，跑起来就会变成一朵花 ❀">
</div>

<br>

> <h4>在线阅读（内容实时更新）</h4>
> - 地址：https://charmve.github.io/computer-vision-in-action

> <h4>最新版PDF下载</h4>
> - 地址：https://github.com/charmve/computer-vision-in-action/releases

</td>
</tr>
</table>

<p align="center">
<img src="https://img-blog.csdnimg.cn/20200310232242296.png">
</p>




