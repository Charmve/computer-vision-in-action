<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>
<b>第 23 章 轻量级网络浅析</b>

作者: 张伟 (Charmve)

日期: 2021/06/06


- 第 23 章 轻量级网络浅析
    - 23.1 概述
    - 23.2 回顾基本卷积运算
    - 23.3 MobileNets
    - 23.4 ShuffleNets
    - 23.5 GhostNet
    - 23.6 SqueezeNet
    - 23.7 EfficientNets
    - 23.8 MicroNet
    - 23.9 YOLO系列
    - 23.10 SSD系列
    - 小结
    - 小练习
    - [参考文献](#六-参考文献)

---



# 第 23 章 轻量级网络浅析

源码快达 -> https://github.com/Charmve/computer-vision-in-action/tree/main/code/chapter23-轻量级网络浅析/



## 23.1 概述

深度神经网络模型被广泛应用在图像分类、物体检测等机器视觉任务中，并取得了巨大成功。然而，由于存储空间和功耗的限制，神经网络模型在嵌入式设备上的存储与计算仍然是一个巨大的挑战。

目前工业级和学术界设计轻量化神经网络模型主要有4个方向：

- (1) 人工设计轻量化神经网络模型；

- (2) 基于神经网络架构搜索（Neural Architecture Search,NAS）的自动化设计神经网络；

- (3) CNN模型压缩；

- (4) 基于AutoML的自动模型压缩。



本文首先回顾第二章节基本卷积计算单元，并基于这些单元介绍MobileNet V1&V2，ShuffleNet V1&V2的设计思路。其次，分别介绍经典轻量级网络，包括Google提出的MobileNet、Face++提出的ShuffleNet、华为提出的GhostNet、经典的YOLO系列和SSD系列网络，并且详细分析了微软和加州大学最新提出的MicroNet网络，YOLO最新版本YOLOX。最后，详细说明各轻量级网络的代码实现，并附上源代码。



## 23.2 回顾基本卷积运算

手工设计轻量化模型主要思想在于设计更高效的“网络计算方式”（主要针对卷积方式），从而使网络参数减少，并且不损失网络性能。本节概述了CNN模型（如MobileNet及其变体）中使用的基本卷积运算单元，并基于空间维度和通道维度，解释计算效率的复杂度。

### 23.2.1 标准卷积

![img](https://pic4.zhimg.com/80/v2-399614b034d4b25382b0f5e4c80107f7_720w.jpg)

图1 标准卷积计算图



HxW表示输入特征图空间尺寸（如图1所示，H和W代表特征图的宽度和高度，输入和输出特征图尺寸不变），N是输入特征通道数，KxK表示卷积核尺寸，M表示输出卷积通道数，则标准卷积计算量是HWNK²M。

![img](https://pic3.zhimg.com/80/v2-4acb7417537e845657901de022a9167a_720w.jpg)

图2 标准卷积计算过程



如图3所示标准卷积在空间维度和通道维度直观说明（以下示意图省略“spatial“，”channel“，”Input“，”Output“），输入特征图和输出特征图之间连接线表示输入和输出之间的依赖关系。以conv3x3为例子，输入和输出空间“spatial”维度密集连接表示局部连接；而通道维度是全连接，卷积运算都是每个通道卷积操作之后的求和(图2)，和每个通道特征都有关，所以“channel”是互相连接的关系。

![img](https://pic2.zhimg.com/80/v2-2d777b5051846271ffce36228fdcc575_720w.jpg)

图3 标准卷积：空间维度和通道维度示意图



### 23.2.2 Grouped Convolution

分组卷积是标准卷积的变体，其中输入特征通道被为G组(图4)，并且对于每个分组的信道独立地执行卷积，则分组卷积计算量是HWNK²M/G，为标准卷积计算量的1/G。

![img](https://pic2.zhimg.com/80/v2-baf2aeff4dc500d4a6d257513c0b2b95_720w.jpg)

图4 分组卷积：空间维度和通道维度示意图



Grouped Convlution最早源于AlexNet。AlexNet在ImageNet LSVRC-2012挑战赛上大显神威，以绝对优势夺得冠军，是卷积神经网络的开山之作，引领了人工智能的新一轮发展。但是AlexNet训练时所用GPU GTX 580显存太小，无法对整个模型训练，所以Alex采用Group convolution将整个网络分成两组后，分别放入一张GPU卡进行训练（如图5所示）。

![img](https://pic4.zhimg.com/80/v2-5c41e6327eeb662dc1c6f8c20c8badcb_720w.jpg)

图5 AlexNet网络架构



### 23.2.3 Depthwise convolution

Depthwise convolution[7]最早是由Google提出，是指将NxHxWxC输入特征图分为group=C组(既Depthwise 是Grouped Convlution的特殊简化形式)，然后每一组做k*k卷积，计算量为HWK²M（是普通卷积计算量的1/N，通过忽略通道维度的卷积显著降低计算量）。Depthwise相当于单独收集每个Channel的空间特征。

![img](https://pic4.zhimg.com/80/v2-e1ba817f6ccfa8dec34ebd1ee5af8907_720w.jpg)

图6 depthwise卷积



![img](https://pic3.zhimg.com/80/v2-5e3d86dd8cdcbe300cd85f1f616440e2_720w.jpg)

图7 Depthwise卷积：空间维度和通道维度示意图



### 23.2.4 Pointwise convolution

Pointwise是指对NxHxWxC的输入做 k个普通的 1x1卷积，如图8，主要用于改变输出通道特征维度。Pointwise计算量为HWNM。

Pointwise卷积相当于在通道之间“混合”信息。

![img](https://pic1.zhimg.com/80/v2-6183445ebd302f4358e3f27333edadec_720w.jpg)

图8 Pointwise卷积



![img](https://pic3.zhimg.com/80/v2-46e4dfaf83219c95d0ddec73a18ad38e_720w.jpg)

图9 Pointwise卷积：空间维度和通道维度示意图



### 23.2.5 Channel Shuffle

Grouped Convlution导致模型的信息流限制在各个group内，组与组之间没有信息交换，这会影响模型的表示能力。因此，需要引入group之间信息交换的机制，即Channel Shuffle操作。

Channel shuffle是ShuffleNet提出的（如图 5 AlexNet也有Channel shuffle机制），通过张量的reshape 和transpose，实现改变通道之间顺序。

![img](https://pic3.zhimg.com/80/v2-3868e1964596de344311df5596506726_720w.jpg)

图10 Channel shuffle：空间维度和通道维度示意图



如图10所示Channel shuffle G=2示意图，Channel shuffle没有卷积计算，仅简单改变通道的顺序。


MobileNet V1&V2,ShuffleNet V1&V2有一个共同的特点，其神经网络架构都是由基本Block单元堆叠，所以本章节首先分析基本Block架构的异同点，再分析整个神经网络的优缺点。

## 23.3 MobileNet

### 23.3.1 MobileNet V1

MobileNet V1是Google第一个提出体积小，计算量少，适用于移动设备的卷积神经网络。MobileNet V1之所以如此轻量，背后的思想是用深度可分离卷积（Depthwise separable convolution）代替标准的卷积，并使用宽度因子(width multiply)减少参数量。

深度可分离卷积把标准的卷积因式分解成一个深度卷积(depthwise convolution)和一个逐点卷积(pointwise convolution)。如1.1标准卷积的计算量是HWNK²M，深度可分离卷积总计算量是：

<img src="https://pic2.zhimg.com/80/v2-7f298a245b3893e3aa20f090121aeb9d_720w.jpg#" alt="img" style="zoom:50%;" />

一般网络架构中M（输出特征通道数）>>K²（卷积核尺寸） (e.g. K=3 and M ≥ 32)，既深度可分离卷积计算量可显著降低标准卷积计算量的1/8–1/9。

深度可分离卷积思想是channel相关性和spatial相关性解耦图12。

![img](https://pic1.zhimg.com/80/v2-3450e77fa31b290006e0ea4b77a16d08_720w.jpg)

图11 Channel shuffle：标准卷积和深度和分离卷积架构对比



![img](https://pic2.zhimg.com/80/v2-5fff373712ea6ff0ae595fd2131c456d_720w.jpg)

图12 深度可分离卷积：空间维度和通道维度示意图


为了进一步降低Mobilenet v1计算量，对输入输出特征通道数M和N乘以宽度因子α(α∈(0,1),d典型值0.25,0.5和0.75),深度可分离卷积总计算量可以进一降低为：

<img src="https://pic3.zhimg.com/80/v2-3cf034e53886c4a9fd79b2e68887f466_720w.jpg#" alt="img" style="zoom:40%;" />

### 23.3.2 MobileNet V2

MobileNet V1设计时参考传统的VGGNet等链式架构，既传统的“提拉米苏”式卷积神经网络模型，都以层叠卷积层的方式提高网络深度，从而提高识别精度。但层叠过多的卷积层会出现一个问题，就是梯度弥散(Vanishing)。残差网络使信息更容易在各层之间流动，包括在前向传播时提供特征重用，在反向传播时缓解梯度信号消失。于是改进版的MobileNet V2[3]增加skip connection，并且对ResNet和Mobilenet V1基本Block如下改进：

● 继续使用Mobilenet V1的深度可分离卷积降低卷积计算量。

● 增加skip connection，使前向传播时提供特征复用。

● 采用Inverted residual block结构。该结构使用Point wise convolution先对feature map进行升维，再在升维后的特征接ReLU，减少ReLU对特征的破坏。

![img](https://pic2.zhimg.com/80/v2-db5f38c980fa4659c0c60d615e9a5425_720w.jpg)

图13 Mobile V1, Mobile V2,ResNet架构对比

## 23.4 ShuffleNet

### 23.4.1 ShuffleNet V1

ShuffleNet是Face++提出的一种轻量化网络结构，主要思路是使用Group convolution和Channel shuffle改进ResNet，可以看作是ResNet的压缩版本。

图13展示了ShuffleNet的结构，其中(a)就是加入BatchNorm的ResNet bottleneck结构，而(b)和(c)是加入Group convolution和Channel Shuffle的ShuffleNet的结构。

![img](https://pic1.zhimg.com/80/v2-67792efa04876858b66eca4a27334f9c_720w.jpg)

图14 shuffle V1 Block架构



![img](https://pic1.zhimg.com/80/v2-e24b5fd15f7966a02acf56bd45993ca8_720w.jpg)

图15 ShuffleNet V1 Block：空间维度和通道维度示意图



如所示，ShuffleNet block最重要的操作是channel shuffle layer，在两个分组卷积之间改变通道的顺序，channel shuffle实现分组卷积的信息交换机制。

ResNet bottleneck计算量：

<img src="https://pic3.zhimg.com/80/v2-8280d63f6f9c490b994c333962d434b6_720w.jpg#" alt="img" style="zoom:33%;" />

``ShuffleNet stride=1`` 计算量：

<img src="https://pic1.zhimg.com/80/v2-cea6eaab7f668b416d5a505c0ebe731c_720w.jpg#" alt="img" style="zoom:33%;" />

对比可知，ShuffleNet和ResNet结构可知，ShuffleNet计算量降低主要是通过分组卷积实现。ShuffleNet虽然降低了计算量，但是引入两个新的问题：

1、channel shuffle在工程实现占用大量内存和指针跳转，这部分很耗时。

2、channel shuffle的规则是人工设计，分组之间信息交流存在随意性，没有理论指导。

### 23.4.2 ShuffleNet V2

Mobile V1&V2，shuffle Net V1 在评价维度的共同特征是：使用FLOPS作为模型的评价标准，但是在移动终端设备时需要满足各个条件：参数少、速度快和精度高，单一的参数少并不一定实现速度快和精度高。

Face++提出的ShuffeNet V2，实现使用直接指标（运算速度）代替间接评价指标（例如FLOPS），并在ARM等移动终端进行评估。并且基于减少计算量提出四个原则：

（1）使用输入和输出通道宽度不同增加卷积的计算量；

（2）组卷积增加MAC；

（3）多分支降低运算效率；

（4）元素级运算增加计算量。



如图16所示，

（a）ShuﬄeNet 基本单元；

（b）用于空间下采样 (2×) 的 ShuffleNet 单元；

（c）ShuﬄeNet V2 的基本单元；

（d）用于空间下采样 (2×) 的 ShuffleNet V2 单元。



ShuffleNet V2 引入通道分割（channel split）操作, 将输入的feature maps分为两部分：一个分支为shortcut流，另一个分支含三个卷积（且三个分支的通道数一样）。分支合并采用拼接（concat），让前后的channel数相同，最后进行Channel Shuffle（完成和ShuffleNet V1一样的功能）。元素级的三个运算channel split、concat、Channel Shuffle合并一个Element-wise，显著降低计算复杂度。

![img](https://pic3.zhimg.com/80/v2-7713deb7bafcf3501da78a62fc87687e_720w.jpg)

图16 ShuffeNet V1 VS ShuffeNet V2架构



ShuﬄeNet V2虽然提出减少计算量的四个原则，基本卷积单元仍采用Depthwise和Pointwise降低计算量，但是没有提出如何实现提高准确率，推断延迟等评价指标。

对比MobileNet V1&V2，ShuﬄeNet V1&V2模型（图17），手工设计轻量化模型主要得益于depth-wise convolution减少计算量，而解决信息不流畅的问题，MobileNet 系列采用了 point-wise convolution，ShuffleNet 采用的是 channel shuffle。

![img](https://pic2.zhimg.com/80/v2-fe43b69af9e6d4171ac1bf9d9a2ec23d_720w.jpg)

图17 卷积运算汇总参考图



## 23.5 GhostNet

GhostNet是华为诺亚方舟实验室在CVPR2020提出，可以在同样精度下，速度和计算量均少于SOTA方法。当前神经网络偏向于移动设备应用，一些重于模型的压缩，比如剪枝、量化、知识蒸馏等。另一些着重于高效的网络设计，比如 MobileNet, ShuffleNet 等。

训练好的网络里的feature map存在大量的冗余信息，相追似的 feature map 类似于 ghost，如下图所示：

![image](https://user-images.githubusercontent.com/29084184/130419230-af75642d-e930-4d45-ae62-38f34a4c5fb2.png)



作者基于“并非所有 feature map 都需要用卷积操作来得到，“ghost” feature map可以用更加廉价的操作来生成“，提出了 Ghost module。

![image](https://user-images.githubusercontent.com/29084184/130419280-26f75bdc-0762-472d-84ff-a81852f40583.png)

图18 Ghost module

Ghost module 如上图18所示，可以看到，包括两次卷积。假设output的通道数为 ![[公式]](https://www.zhihu.com/equation?tex=init%5C_channels%E2%88%97ratio) ，那么第一次卷积生成 ![[公式]](https://www.zhihu.com/equation?tex=init%5C_channels) 个 feature map。

第二次卷积：每个 feature map 通过映射生成 ![[公式]](https://www.zhihu.com/equation?tex=ratio-1) 个新的 feature map，这样会生成 ![[公式]](https://www.zhihu.com/equation?tex=+init%5C_channels%E2%88%97%28ratio%E2%88%921%29) 个 feature map。最后，把第一次卷积和第二次卷积得到的 feature map 拼接在一起，得到output，通道数为 ![[公式]](https://www.zhihu.com/equation?tex=init%5C_channels%E2%88%97ratio) 。

Ghost module 的代码如下所示，关键步骤我添加了备注说明：

```python3
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)
        
        # 第一次卷积：得到通道数为init_channels，是输出的 1/ratio
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential())
       
       # 第二次卷积：注意有个参数groups，为分组卷积
       # 每个feature map被卷积成 raito-1 个新的 feature map
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # 第一次卷积得到的 feature map，被作为 identity
        # 和第二次卷积的结果拼接在一起
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]
```

最有趣的是模块里，第二次卷积，作者也考虑了仿射变换、小波变换等，因为卷积运算有较好的硬件支持，作者更推荐卷积。

Ghost Bottleneck(G-bneck)与residual block类似，主要由两个Ghost模块堆叠二次，第一个模块用于增加特征维度，增大的比例称为expansion ratio，而第二个模块则用于减少特征维度，使其与输入一致。G-bneck包含stride=1和stride=2版本，对于stride=2，shortcut路径使用下采样层，并在Ghost模块中间插入stride=2的depthwise卷积。为了加速，Ghost模块的原始卷积均采用pointwise卷积。

![image](https://user-images.githubusercontent.com/29084184/130419410-a0c9ab68-331e-4e11-8e23-e3060c8e216b.png)

图19 Ghost Bottleneck(G-bneck)

在网络架构上，GhostNet 将 MobileNetV3 的 bottleneck block 替换为 Ghost bottleneck，部分 Ghost模块 加入了SE模块。



GitHub 源代码

https://github.com/Charmve/computer-vision-in-action/tree/main/code/chapter23-轻量级网络浅析/GhostNet/



## 23.6 SqueezeNet

论文对模型缩放进行深入研究，提出混合缩放方法，该方法可以更优地选择宽度、深度和分辨率的维度缩放比例，从而使得模型能够达到更高的精度。另外，论文通过NAS神经架构搜索提出EfficientNet，配合混合缩放方法，能够使用很少量的参数达到较高的准确率。

### 23.6.1 SqueezeNet的压缩策略

SqueezeNet的模型压缩使用了3个策略：

1. 将 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 卷积替换成 ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes1) 卷积：通过这一步，一个卷积操作的参数数量减少了9倍；
2. 减少 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 卷积的通道数：一个 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 卷积的计算量是 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3%5Ctimes+M+%5Ctimes+N) （其中 ![[公式]](https://www.zhihu.com/equation?tex=M) ， ![[公式]](https://www.zhihu.com/equation?tex=N) 分别是输入Feature Map和输出Feature Map的通道数），作者任务这样一个计算量过于庞大，因此希望将 ![[公式]](https://www.zhihu.com/equation?tex=M) ， ![[公式]](https://www.zhihu.com/equation?tex=N) 减小以减少参数数量；
3. 将降采样后置：作者认为较大的Feature Map含有更多的信息，因此将降采样往分类层移动。注意这样的操作虽然会提升网络的精度，但是它有一个非常严重的缺点：即会增加网络的计算量。

### 23.6.2 Fire模块

SqueezeNet是由若干个Fire模块结合卷积网络中卷积层，降采样层，全连接等层组成的。一个Fire模块由Squeeze部分和Expand部分组成（注意区分和Momenta的[SENet](https://zhuanlan.zhihu.com/p/47494490)[4]的区别）。Squeeze部分是一组连续的 ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes1) 卷积组成，Expand部分则是由一组连续的 ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes1) 卷积和一组连续的 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 卷积cancatnate组成，因此 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 卷积需要使用same卷积，Fire模块的结构见图1。在Fire模块中，Squeeze部分 ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes1) 卷积的**通道**数记做 ![[公式]](https://www.zhihu.com/equation?tex=s_%7B1x1%7D) ，Expand部分 ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes1) 卷积和 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 卷积的**通道**数分别记做 ![[公式]](https://www.zhihu.com/equation?tex=e_%7B1x1%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=e_%7B3x3%7D) （论文图画的不好，不要错误的理解成卷积的层数）。在Fire模块中，作者建议 ![[公式]](https://www.zhihu.com/equation?tex=s_%7B1x1%7D+%3C+e_%7B1x1%7D+%2B+e_%7B3x3%7D) ，这么做相当于在两个 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 卷积的中间加入了瓶颈层，作者的实验中的一个策略是 ![[公式]](https://www.zhihu.com/equation?tex=s_%7B1x1%7D+%3D+%5Cfrac%7Be_%7B1x1%7D%7D%7B4%7D+%3D+%5Cfrac%7Be_%7B3x3%7D%7D%7B4%7D) 。图20中 ![[公式]](https://www.zhihu.com/equation?tex=s_%7B1x1%7D%3D3) ， ![[公式]](https://www.zhihu.com/equation?tex=e_%7B1x1%7D+%3D+e_%7B3x3%7D+%3D+4) 。

![img](https://pic3.zhimg.com/80/v2-5fde12f060519e493cb059484514f88a_720w.jpg)

图20 SqueezeNet的Fire模块


下面代码片段是Keras实现的Fire模块，注意拼接Feature Map的时候使用的是Cancatnate操作，这样不必要求 ![[公式]](https://www.zhihu.com/equation?tex=e_%7B1x1%7D+%3D+e_%7B3x3%7D) 。

```python
def fire_model(x, s_1x1, e_1x1, e_3x3, fire_name):
    # squeeze part
    squeeze_x = Conv2D(kernel_size=(1,1),filters=s_1x1,padding='same',activation='relu',name=fire_name+'_s1')(x)
    # expand part
    expand_x_1 = Conv2D(kernel_size=(1,1),filters=e_1x1,padding='same',activation='relu',name=fire_name+'_e1')(squeeze_x)
    expand_x_3 = Conv2D(kernel_size=(3,3),filters=e_3x3,padding='same',activation='relu',name=fire_name+'_e3')(squeeze_x)
    expand = merge([expand_x_1, expand_x_3], mode='concat', concat_axis=3)
    return expand
```

图21是使用Keras自带的`plot_model`功能得到的Fire模块的可视图，其中 ![[公式]](https://www.zhihu.com/equation?tex=s_%7B1x1%7D+%3D+%5Cfrac%7Be_%7B1x1%7D%7D%7B4%7D+%3D+%5Cfrac%7Be_%7B3x3%7D%7D%7B4%7D%3D16) 。

![img](https://pic1.zhimg.com/80/v2-921ca51e1265508f3bf8a55261a1b878_720w.jpg)

图21 Keras可视化的SqueezeNet的Fire模块



### 23.6.3 SqueezeNet的网络架构

图3是SqueezeNet的几个实现，左侧是不加short-cut的SqueezeNet，中间是加了short-cut的，右侧是short-cut跨有不同Feature Map个数的卷积的。还有一些细节图3中并没有体现出来：

1. 激活函数默认都使用ReLU；
2. fire9之后接了一个rate为0.5的dropout；
3. 使用same卷积。

![img](https://pic3.zhimg.com/80/v2-5f8ff8cb94babde05e69365336e77a62_720w.jpg)

图22 SqueezeNet网络结构



表1：SqueezeNet网络参数

![img](https://pic3.zhimg.com/80/v2-3fb3ea5ca5a3b4cbe4afd448bc5f9396_720w.jpg)

表1给出了SqueezeNet的详细参数，我们的Keras实现如下面代码片段：

```python
def squeezeNet(x):
    conv1 = Conv2D(input_shape = (224,224,3), strides = 2, filters=96, kernel_size=(7,7), padding='same', activation='relu')(x)
    poo1 = MaxPool2D((2,2))(conv1)
    fire2 = fire_model(poo1, 16, 64, 64,'fire2')
    fire3 = fire_model(fire2, 16, 64, 64,'fire3')
    fire4 = fire_model(fire3, 32, 128, 128,'fire4')
    pool2 = MaxPool2D((2,2))(fire4)
    fire5 = fire_model(pool2, 32, 128, 128,'fire5')
    fire6 = fire_model(fire5, 48, 192, 192,'fire6')
    fire7 = fire_model(fire6, 48, 192, 192,'fire7')
    fire8 = fire_model(fire7, 64, 256, 256,'fire8')
    pool3 = MaxPool2D((2,2))(fire8)
    fire9 = fire_model(pool3, 64, 256, 256,'fire9')
    dropout1 = Dropout(0.5)(fire9)
    conv10 = Conv2D(kernel_size=(1,1), filters=1000, padding='same', activation='relu')(dropout1)
    gap = GlobalAveragePooling2D()(conv10)
    return gap
```

上面的代码，模型的summary，以及SqueezeNet的keras可视化见：[https://github.com/senliuy/CNN-Structures/blob/master/SqueezeNet.ipynb](https://link.zhihu.com/?target=https%3A//github.com/senliuy/CNN-Structures/blob/master/SqueezeNet.ipynb)。

### 23.6.4 SqueezeNet的性能以及深度压缩

图3左侧的SqueezeNet的性能（top1：57.5%，top5：80.3%）是可以类似AlexNet的（top1：57.2%，top5：80.3%）。从表1中我们可以看出SqueezeNet总共有1,248,424个参数，同性能的AlexNet则有58,304,586个参数（主要集中在全连接层，去掉之后有3,729,472个）。使用他们提出的Deep Compression[3]算法压缩后，模型的参数数量可以降到421，098个。

### 23.6.5 小结

SqueezeNet的压缩策略是依靠将 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 卷积替换成 ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes1) 卷积来达到的，其参数数量是等性能的AlexNet的2.14%。从参数数量上来看，SqueezeNet的目的达到了。SqueezeNet的最大贡献在于其开拓了模型压缩这一方向，之后的一系列文章也就此打开。

这里我们着重说一下SqueezeNet的缺点：

1. SqueezeNet的侧重的应用方向是嵌入式环境，目前嵌入式环境主要问题是实时性。SqueezeNet的通过更深的深度置换更少的参数数量虽然能减少网络的参数，但是其丧失了网络的并行能力，测试时间反而会更长，这与目前的主要挑战是背道而驰的；
2. 论文的题目非常标题党，虽然纸面上是减少了50倍的参数，但是问题的主要症结在于AlexNet本身全连接节点过于庞大，50倍参数的减少和SqueezeNet的设计并没有关系，考虑去掉全连接之后3倍参数的减少更为合适。
3. SqueezeNet得到的模型是5MB左右，0.5MB的模型还要得益于Deep Compression。虽然Deep Compression也是这个团队的文章，但是将0.5这个数列在文章的题目中显然不是很合适。


## 23.7 EfficientNet

该网络核心思想**是否存在一条准则来指导卷积网络的性能优化**，而结论是网络的宽度/深度/分辨率是有平衡关系的，而且存在一个常量的比例。基于以上的发现，论文提出简单有效的混合缩放方法(compound scaling method)，通过一系列的固定缩放因子，均匀地缩放网络的宽度，深度和分辨率，图2展示了各维度缩放以及混合缩放方法的示意图。

此外，由于网络的性能和效率十分依赖于基础网络，论文使用神经结构搜索来获取新的基础网络，再进行相应的缩放来获取一系列模型，称之为EfficientNets。

![image](https://user-images.githubusercontent.com/29084184/130424117-91b324d4-3f6a-40d0-9c89-b0a908cf6f5a.png)

图23 混合缩放方法



![image](https://user-images.githubusercontent.com/29084184/130427322-51f7cb64-978b-4cd9-8e00-c4cd69f0769e.png)

图24 不同缩放方法对应的计算量比较



### 23.7.1 如何统一三个维度的量度

对网络深度、宽度和分辨率中的任何维度进行缩放都可以提高精度，但是当模型足够大时，这种放大的收益会减弱。因此为了追去更好的精度和效率，在缩放时平衡网络所有维度至关重要。之前的一些工作已经开始在任意缩放网络深度和宽度，但是他们仍然需要复杂的人工微调。在本篇论文中，作者提出了一个新的复合缩放方法——使用一个复合系数$ϕ$统一缩放网络宽度、深度和分辨率，如下图所示。

![image-20210823173118090](C:\Users\zhangwei13\AppData\Roaming\Typora\typora-user-images\image-20210823173118090.png)

这里的 $α,β,γ$ 都是由一个很小范围的网络搜索得到的常量，直观上来讲，$ϕ$ 是一个特定的系数，可以控制用于资源的使用量，$α,β,γ$ 决定了具体是如何分配资源的。



### 23.7.2 Efficient 网络架构

Efficient是一种基于移动应用的baseline模型。受到MnasNet的启发，作者也开发了一种多目标的神经网络结构搜索同时优化精度和FLOPS，搜索空间和MnasNet相同，因为我们使用的搜索空间和MnasNet相似，所以得到的网络结构也很相似，不过他们的EfficientNet-B0稍微大了点，因为他们的FLOPS预算也比MnasNet中大（400M）。table 1展示了EfficientNet-B0的结构，它的主要构建块就是移动倒置瓶颈MBConv，其网络结构如下图所示。

![image](https://user-images.githubusercontent.com/29084184/130425119-441d5966-0d19-4d0a-b295-b0a801581054.png)

图25 MnasNet-A1 网络结构

表2 EfficientNet-B0 基准网络（Baseline Network）—— 每一行描述一个阶段 i，有 $L^i $层，输入分辨率为 $hH^i$，$W^ii$ 和输出通道 $C^i$。

![image-20210823173606377](C:\Users\zhangwei13\AppData\Roaming\Typora\typora-user-images\image-20210823173606377.png)

- 论文地址 https://arxiv.org/pdf/1905.11946.pdf

- GitHub 源码：https://github.com/Charmve/computer-vision-in-action/tree/main/code/chapter23-轻量级网络浅析/EfficientNet/



## 23.8 MicroNet

MicroNet主要是在MobileNet系列上进行改进和对比，提出了两项改进方法：

- **Micro-Factorized convolution** 将MobileNet中的point-wise卷积以及depth-wise卷积分解为低秩矩阵，从而使得通道数目和输入输出的连通性得到一个良好的平衡。

- **Dynamic Shift-Max** 使用一种新的激活函数，通过最大化输入特征图与其循环通道偏移之间的多重动态融合，来增强非线性特征。之所以称之为动态是因为，融合过程的参数依赖于输入特征图。

作者使用Micro-Factorized convolution和Dynamic Shift-Max建立了MicroNets家族，最终实现了低FLOPs的STOA表现。MicroNet-M1在ImageNet分类任务上仅仅用12 MFLOPs的计算量，就实现了61.1%的top1准确度，比MobileNetV3要强11.3% .

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9zMVpOcTByZFFlN2hac2ljMGVrcjQ0NWJhUXBUODNNemlhbDJ5YlRvUEpKQTRSWFdwaWNzUFpObVBMaFJuR0JPTG52eWQ5aWFpYjZWSlFQVlVyeGlhazNwNzRSdy82NDA?x-oss-process=image/format,png)

图26 MicroNet网络结构

 

## 23.9 YOLO系列



<img src="https://img-blog.csdn.net/20160317163739691" alt="img" style="zoom:67%;" />

参考链接

 https://blog.csdn.net/qq_18941713/article/details/90776197

 https://zhuanlan.zhihu.com/p/62366940



## 23.10 SSD系列



SSD: Single Shot MultiBox Detector

<img src="https://user-images.githubusercontent.com/29084184/130428592-103ebc66-4617-46b9-b717-7e0ea72af991.png#" alt="image" style="zoom:80%;" />
<img src="https://user-images.githubusercontent.com/29084184/130428614-d11cc81e-a417-43e4-8f31-1fc6a5c9bfe3.png#" alt="image" style="zoom:80%;" />

- 论文链接 https://arxiv.org/pdf/1512.02325.pdf
- GitHub 源码 https://github.com/amdegroot/ssd.pytorch



## 小结



## 小练习



## 参考文献

[1] Chollet, F.: Xception: Deep learning with depthwise separable convolutions. arXiv preprint (2016)

[2] Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., An- dreetto, M., Adam, H.: Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 (2017)

[3] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.: Inverted residuals and linear bottlenecks: Mobile networks for classification, detection and segmenta- tion. arXiv preprint arXiv:1801.04381 (2018)

[4] Zhang, X., Zhou, X., Lin, M., Sun, J.: Shufflenet: An extremely efficient convolu- tional neural network for mobile devices. arXiv preprint arXiv:1707.01083 (2017)

[5] Han, S., Mao, H., Dally, W.J., 2015a. Deep compression:Compressing deep neural networks with pruning,trained quantization and huffman coding. arXiv preprint arXiv:1510.00149(2015)

[6] Z. Qin, Z. Zhang, X. Chen, and Y. Peng, “FD-MobileNet: Improved MobileNet with a Fast Downsampling Strategy,” in arXiv:1802.03750, 2018.

[7] F. Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions,” in Proc. of CVPR, 2017.

[8] Iandola F N, Han S, Moskewicz M W, et al. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size[J]. arXiv preprint arXiv:1602.07360, 2016.
