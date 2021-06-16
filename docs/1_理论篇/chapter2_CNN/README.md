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

- 第 2 章 [卷积神经网络](chapter2_CNN.md)
    - 2.1 [从神经网络到卷积神经网络](chapter2_CNN.md#21-从神经网络到卷积神经网络)
      - 2.1.1 [定义](chapter2_CNN.md#211-定义)
      - 2.1.2 [卷积神经网络的架构](chapter2_CNN.md#212-卷积神经网络的架构)
    - 2.2 [卷积网络的层级结构](chapter2_CNN.md#22-卷积网络的层级结构)
      - 2.2.1 [数据输入层](chapter2_CNN.md#221-数据输入层)
      - 2.2.2 [卷积计算层](chapter2_CNN.md#222-卷积计算层)
      - 2.2.3 [非线性层（或激活层）](chapter2_CNN.md#223-非线性层或激活层)
      - 2.2.4 [池化层](chapter2_CNN.md#224-池化层)
      - 2.2.5 [全连接层](/chapter2_CNN.md#225-全连接层)
    - 2.3 [卷积神经网络的几点说明](/chapter2_CNN.md#23-卷积神经网络的几点说明)
    - 2.4 [实战项目 2 - 动手搭建一个卷积神经网络](/chapter2_CNN-in-Action.md)
      - 2.4.1 [卷积神经网络的前向传播](/chapter2_CNN-in-Action.md#271-卷积神经网络的前向传播)
      - 2.4.2 [卷积神经网络的反向传播](/chapter2_CNN-in-Action.md#272-卷积神经网络的反向传播)
      - 2.4.3 [手写一个卷积神经网络](/chapter2_CNN-in-Action.md#273-手写一个卷积神经网络)
        - [1. 定义一个卷积层](chapter2_CNN-in-Action.md#1-定义一个卷积层)
        - [2. 构造一个激活函数](chapter2_CNN-in-Action.md#2-构造一个激活函数)
        - [3. 定义一个类，保存卷积层的参数和梯度](chapter2_CNN-in-Action.md#3-定义一个类保存卷积层的参数和梯度)
        - [4. 卷积层的前向传播](chapter2_CNN-in-Action.md#4-卷积层的前向传播)
        - [5. 卷积层的反向传播](chapter2_CNN-in-Action.md#5-卷积层的反向传播)
        - [6. MaxPooling层的训练](chapter2_CNN-in-Action.md#6-MaxPooling层的训练)
      - 2.4.4 [PaddlePaddle卷积神经网络源码解析](chapter2_CNN-in-Action.md#274-PaddlePaddle卷积神经网络源码解析)
    - [小结](chapter2_CNN.md#小结)
    - [参考文献](/chapter2_CNN.md#参考文献)
    

