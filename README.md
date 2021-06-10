
<div id="outputFigDisplay" class="fig">
    <pre id="taag_output_text" style="left;" class="flag" contenteditable="true">
    <code>
    __  ___      _               _ ___    ____                             
   /  |/  /___ _(_)      _____  (_)   |  /  _/    _________  ____ ___      
  / /|_/ / __ `/ / | /| / / _ \/ / /| |  / /_____/ ___/ __ \/ __ `__ \     
 / /  / / /_/ / /| |/ |/ /  __/ / ___ |_/ /_____/ /__/ /_/ / / / / / /     
/_/  /_/\__,_/_/ |__/|__/\___/_/_/  |_/___/     \___/\____/_/ /_/ /_/      
    </code>
    </pre>
</div>

## Maiwei AI Lab - 迈微AI研习社

<a href="https://www.github.com/Charmve" target="_blank"><img src="https://img.shields.io/badge/创始人-@Charmve-000000.svg?style=flat-square&logo=GitHub"></a>  [![CircleCI](https://circleci.com/gh/Charmve/computer-vision-in-action.svg?style=svg)](https://circleci.com/gh/Charmve/computer-vision-in-action)

作者系迈微AI研习社创始人、CSDN博客专家，主要分享机器学习算法、计算机视觉等相关内容，每周研读顶会论文，持续关注前沿技术动态。公众号底部有菜单分类，关注我们，一起学习成长。

---

### 一种结合了代码、数据集和HTML的数字学习媒介
<br>

<table align="center">
<tr>
<td>
<code>全面</code>&nbsp;<code>前沿</code>&nbsp;<code>免费</code>
<h1> 计算机视觉实战演练：算法与应用 <sup> 📌</sup>
<br><em>Computer Vision in Action</em></h1>

作者：张伟（Charmve）

<p align="center">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/作者-@Charmve-000000.svg?logo=GitHub" alt="GitHub"></a>
  <a href="https://github.com/Charmve/computer-vision-in-action"><img src="https://img.shields.io/badge/CV-Action-yellow" alt="CV-Action"></a>
  <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>
  <a href="https://github.com/Charmve/computer-vision-in-action/edit/master/README.md"><img src="https://img.shields.io/github/stars/Charmve/computer-vision-in-action?style=social" alt="Stars"></a>
  <a href="https://github.com/Charmve/computer-vision-in-action/edit/master/README.md"><img src="https://img.shields.io/github/forks/Charmve/computer-vision-in-action?style=social" alt="Forks"></a>
</p>

<div align="center">
	<img src="https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/maiwei.png" width="220px" alt="logo:maiwei" title="有疑问，跑起来就会变成一朵花 ❀">
</div>

<br>

> <h4>在线阅读（内容实时更新）</h4>
> - 地址：~~https://charmve.github.io/computer-vision-in-action~~ (正在开发中。。。)

> <h4>最新版PDF下载</h4>
> - 地址：~~https://github.com/charmve/computer-vision-in-action/releases~~

</tr>
</td>
</table>

### 全书组织
:label: `fig_book_org`

<p align="center">
  <img src="res/ui/book_org.png" alt="book_org.png">
</p> 

<p align="right">
  <a href="res/计算机视觉实战演练：算法与应用_思维导图.pdf">全书详细思维导图</a>
</p>

本书详细介绍，请移步 [<b>序言</b>](/docs/book_preface.md)

本开源项目代表了我们的一种尝试：我们将教给读者概念、背景知识和代码；我们将在同一个地方阐述剖析问题所需的批判性思维、解决问题所需的数学知识，以及实现解决方案所需的工程技能。

我们的目标是创建一个为实现以下目标的统一资源：
1. 所有人均可在网上免费获取；
2. 提供足够的技术深度，从而帮助读者实际成为计算机视觉应用科学家：既理解数学原理，又能够实现并不断改进方法；
3. 包含可运行的代码，为读者展示如何在实际中解决问题。这样不仅直接将数学公式对应成实际代码，而且可以修改代码、观察结果并及时获取经验；
4. 允许我们和整个[社区](https://github.com/Charmve/computer-vision-in-action/discussions)不断快速迭代内容，从而紧跟仍在高速发展的计算机视觉领域；
5. 由包含有关技术细节问答的论坛作为补充，使大家可以相互答疑并交换经验。


### 目录

<table align="center">
<tr>
<td>

- <b><h4>[序言](/docs/book_preface.md)</h4></b>
- <b><h4>主要符号表</h4></b>
- <b><h4>绪论篇</h4></b>
  - 第 0 章  [计算机视觉概述](https://charmve.github.io/computer-vision-in-action/#/chapter0/chapter0)
    - 1. 概述
    - 2. 计算机视觉基本概念
    - 3. 典型的计算机视觉任务
        - 图像分类 
        - 目标识别与目标检测
        - 实例分割与语义分割
        - 3D 建模
    - 4. [发展历史回顾](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/0_绪论/4_发展历史回顾.md)
    - 5. 小练习
    - 小结
    - 参考文献
- <b><h4>理论篇</h4></b>
  - 第 1 章 [神经网络](https://charmve.github.io/computer-vision-in-action/#/chapter2/chapter2)
    - 1.1 [Softmax 回归](./docs/1_理论篇/chapter1_Neural-Networks/chapter1.1_Softmax回归.md)
    - 1.2 [反向传播算法](./docs/1_理论篇/chapter1_Neural-Networks/chapter1.2_Back-Propagation.md)
    - 1.3 [多层感知器](./docs/1_理论篇/chapter1_Neural-Networks/chapter1.3_多层感知器MLP.md)
    - 1.4 [神经学观点](./docs/1_理论篇/chapter1_Neural-Networks/chapter1.4_神经学观点.md)
    - 1.5 [实战项目 1 - 手写字分类](https://blog.csdn.net/Charmve/article/details/108531735)
    - 小结
    - 参考文献
  - 第 2 章 [卷积神经网络](docs/1_理论篇/chapter2_CNN/chapter2_CNN.md)
    - 2.1 [从神经网络到卷积神经网络](docs/1_理论篇/chapter2_CNN/chapter2_CNN.md#21-从神经网络到卷积神经网络)
      - 2.1.1 [定义](/docs/1_理论篇/chapter2_CNN/chapter2_CNN.md#211-定义)
      - 2.1.2 [卷积神经网络的架构](/docs/1_理论篇/chapter2_CNN/chapter2_CNN.md#212-卷积神经网络的架构)
    - 2.2 [卷积网络的层级结构](/docs/1_理论篇/chapter2_CNN/chapter2_CNN.md#22-卷积网络的层级结构)
      - 2.2.1 [数据输入层](/docs/1_理论篇/chapter2_CNN/chapter2_CNN.md#221-数据输入层)
      - 2.2.2 [卷积计算层](/docs/1_理论篇/chapter2_CNN/chapter2_CNN.md#222-卷积计算层)
      - 2.2.3 [非线性层（或激活层）](/docs/1_理论篇/chapter2_CNN/chapter2_CNN.md#223-非线性层或激活层)
      - 2.2.4 [池化层](/docs/1_理论篇/chapter2_CNN/chapter2_CNN.md#224-池化层)
      - 2.2.5 [全连接层](/docs/1_理论篇/chapter2_CNN/chapter2_CNN.md#225-全连接层)
    - 2.3 [卷积神经网络的几点说明](/docs/1_理论篇/chapter2_CNN/chapter2_CNN.md#23-卷积神经网络的几点说明)
    - 2.4 [实战项目 2 - 动手搭建一个卷积神经网络](/docs/1_理论篇/chapter2_CNN/chapter2_CNN-in-Action.md)
      - 2.4.1 [卷积神经网络的前向传播](/docs/1_理论篇/chapter2_CNN/chapter2_CNN-in-Action.md#271-卷积神经网络的前向传播)
      - 2.4.2 [卷积神经网络的反向传播](/docs/1_理论篇/chapter2_CNN/chapter2_CNN-in-Action.md#272-卷积神经网络的反向传播)
      - 2.4.3 [手写一个卷积神经网络](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/1_理论篇/chapter2_CNN/chapter2_CNN-in-Action.md#273-手写一个卷积神经网络)
        - [1. 定义一个卷积层](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/1_理论篇/chapter2_CNN/chapter2_CNN-in-Action.md#1-定义一个卷积层)
        - [2. 构造一个激活函数](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/1_理论篇/chapter2_CNN/chapter2_CNN-in-Action.md#2-构造一个激活函数)
        - [3. 定义一个类，保存卷积层的参数和梯度](#3-定义一个类保存卷积层的参数和梯度)
        - [4. 卷积层的前向传播](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/1_理论篇/chapter2_CNN/chapter2_CNN-in-Action.md#4-卷积层的前向传播)
        - [5. 卷积层的反向传播](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/1_理论篇/chapter2_CNN/chapter2_CNN-in-Action.md#5-卷积层的反向传播)
        - [6. MaxPooling层的训练](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/1_理论篇/chapter2_CNN/chapter2_CNN-in-Action.md#6-MaxPooling层的训练)
      - 2.4.4 [PaddlePaddle卷积神经网络源码解析](/docs/1_理论篇/chapter2_CNN/chapter2_CNN-in-Action.md#274-PaddlePaddle卷积神经网络源码解析)
    - [小结](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/1_理论篇/chapter2_CNN/chapter2_CNN.md#小结)
    - [参考文献](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/1_理论篇/chapter2_CNN/chapter2_CNN.md#参考文献)
  - 第 3 章 [图像分类](https://charmve.github.io/computer-vision-in-action/#/chapter3/chapter3)
    - 3.1 [数据驱动方法]()
    - 3.2 [k 最近邻算法](/docs/1_理论篇/chapter3_Image-Classification/chapter32_knn.md)
    - 3.3 [线性分类](/docs/1_理论篇/chapter3_Image-Classification/chapter33_line-classification.md)
    - 3.4 [逻辑回归 LR](/docs/1_理论篇/chapter3_Image-Classification/chapter34_Logistic-Regression.md) 
    - 3.5 [实战项目 3 - 表情识别](https://blog.csdn.net/charmve/category_9754344.html)
    - 3.6 [实战项目 4 - 使用卷积神经网络对CIFAR10图片进行分类](http://mp.weixin.qq.com/s?__biz=MzIxMjg1Njc3Mw%3D%3D&chksm=97bef597a0c97c813e185e1bbf987b93d496c6ead8371364fd175d9bac46e6dcf7059cf81cb2&idx=1&mid=2247487293&scene=21&sn=89684d1c107177983dc1b4dca8c20a5b#wechat_redirect)
    - [小结](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/1_理论篇/chapter3_Image-Classification/chapter3_Image-Classification.md#小结)
    - [参考文献](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/1_理论篇/chapter3_Image-Classification/chapter3_Image-Classification.md#参考文献)
  - 第 4 章 [递归神经网络](https://charmve.github.io/computer-vision-in-action/#/chapter4/chapter4)
    - 4.1 [递归神经网络 RNN](/docs/1_理论篇/chapter4_递归神经网络/chapter4.1_递归神经网络.md)
    - 4.2 [循环神经网络的从零开始实现](/docs/1_理论篇/chapter4_递归神经网络/chapter4.2_循环神经网络的从零开始实现.md)
    - 4.3 [循环神经网络的简洁实现](/docs/1_理论篇/chapter4_递归神经网络/chapter4.3_循环神经网络的简洁实现.md)
    - 4.4 [长短期记忆人工神经网络 LSTM](/docs/1_理论篇/chapter4_递归神经网络/chapter4.4_长短期记忆人工神经网络LSTM.md)
    - 4.5 [门控循环单元（GRU）](/docs/1_理论篇/chapter4_递归神经网络/chapter4.5_门控循环单元.md)
    - 小结
    - 参考文献
  - 第 5 章 [模型拟合与优化算法]()
    - 5.1 优化与深度学习
    - 5.2 梯度下降和随机梯度下降
    - 5.3 小批量随机梯度下降
    - 5.4 动量法
    - 5.5 AdaGrad算法
    - 5.6 RMSProp算法
    - 5.7 AdaDelta算法
    - 5.8 Adam算法
    - 小结
    - 参考文献
- <b><h4>实战篇</h4></b>
  - 第 6 章 [深度学习环境搭建](https://charmve.github.io/computer-vision-in-action/#/chapter6/chapter6)
    - 6.1 [深度学习环境搭建指南](docs/2_实战篇/chapter6_深度学习环境搭建/chapter6.1_深度学习环境搭建指南.md)
    - 6.2 [Pytorch 基础使用介绍](docs/2_实战篇/chapter6_深度学习环境搭建/chapter6.2_Pytorch-基础使用介绍.md)
      - 6.2.1 [Tensors](#621-tensors)
      - 6.2.2 [Operations](#622-operations)
      - 6.2.3 [Numpy桥梁](#623-numpy桥梁)
      - 6.2.4 [CUDA Tensors](#624-cuda-tensors)
    - 6.3 [Python](./notebooks/02_Python.ipynb)
    - 6.4 [Numpy 基础使用](./notebooks/03_NumPy.ipynb)
    - 6.5 [Pandas 基础使用](./notebooks/04_Pandas.ipynb)
    - 6.6 [OpenCV 安装及基础使用](./notebooks/02_Python.ipynb)
    - 6.7 [Jupyter Notebook 配置及基础使用](./notebooks/01_Notebooks.ipynb)
    - 小结
    - 参考文献
  - 第 7 章 [经典卷积神经网络架构：原理与PyTorch实现](https://github.com/Charmve/Semantic-Segmentation-PyTorch)
    - 7.1 [卷积神经网络（LeNet）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/7.1%20卷积神经网络（LeNet）.md)
    - 7.2 [深度卷积神经网络（AlexNet）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/7.2%20深度卷积神经网络（AlexNet）.md)
    - 7.3 [使用重复元素的网络（VGG）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/7.3%20使用重复元素的网络（VGG）.md)
    - 7.4 [含并行连结的网络（GoogLeNet）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/7.4%20含并行连结的网络（GoogLeNet）.md)
    - 7.5 [残差网络（ResNet）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/)
    - 7.6 [二阶网络编码解码（U-Net）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/)
    - 7.7 [稠密连接网络（DenseNet）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/)
    - 7.8 [实例分割网络（SegNet）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/)
    - 7.9 [Mask-RCNN](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/)
    - 7.10 [区域卷积神经网络（R-CNN）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/7.9%20区域卷积神经网络（R-CNN）.md)
    - 7.11 [全卷积网络（FCN）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/)
    - 小结
    - 参考文献
  - 第 8 章 [著名数据集及基准](docs/2_实战篇/chapter8_著名数据集及基准)
    - 8.1 [数据集](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.1_著名数据集.md)
        - 8.1.1 [常见数据集](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.1_著名数据集.md#811-常见数据集)
          - 8.1.1.1 [ImageNet](https://image-net.org/)
          - 8.1.1.2 [MNIST](http://yann.lecun.com/exdb/mnist/)
          - 8.1.1.3 [COCO](https://cocodataset.org/)
          - 8.1.1.4 [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
        - 8.1.2 [Pytorch数据集及读取方法简介](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.1_著名数据集.md#812-pytorch数据集及读取方法简介)
        - 8.1.3 [数据增强简介](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.1_著名数据集.md#813-数据增强简介)
        - [总结](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.1_著名数据集.md#总结)
    - 8.2 [基准](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.2_基准BenchMark.md)
    - 8.3 [实战项目 5 - Kaggle比赛：图像分类（CIFAR-10）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/7.12%20实战Kaggle比赛：图像分类（CIFAR-10）.md)
    - 8.4 [实战项目 6 - Kaggle比赛：狗的品种识别（ImageNet Dogs）](docs/2_实战篇/chapter7_经典卷积神经网络架构-原理与PyTorch实现/7.13%20实战Kaggle比赛：狗的品种识别（ImageNet%20Dogs）.md)
    - 小结
    - 参考文献
  - 第 9 章 [检测与分割实战项目](https://charmve.github.io/computer-vision-in-action/#/chapter9/chapter9)
    - 9.1 语义分割
      - 9.1.1 [语义分割 PyTorch 版](https://github.com/Charmve/Semantic-Segmentation-PyTorch)
      - 9.1.2 实战项目 7
    - 9.2 目标检测
      - 9.2.1 常用网络
      - 9.2.2 实战项目 8
    - 9.3 [实例分割](/docs/2_实战篇/chapter9_检测与分割实战项目/9.3%20实例分割.md)
      - 9.3.1 [常用网络](/docs/2_实战篇/chapter9_检测与分割实战项目/9.3%20实例分割.md#931-常用网络)
      - 9.3.2 [实战项目 9 - 实时高分辨率背景抠图](/docs/2_实战篇/chapter9_检测与分割实战项目/9.3%20实例分割.md#932-实战项目-8-实时高分辨率背景抠图)
      - 9.3.3 新方法：[滑动窗口](https://blog.csdn.net/Charmve/article/details/108915225), [PointRend](https://blog.csdn.net/Charmve/article/details/108892076), PolarMask
    - 小结
    - 参考文献
  - 第 10 章 [项目实战综合](https://charmve.github.io/computer-vision-in-action/#/chapter10/chapter10)
    - 10.1 [手写字识别](https://blog.csdn.net/Charmve/article/details/108531735)
    - 10.2 [文本检测](https://github.com/Charmve/Scene-Text-Detection)
    - 10.3 [车道线检测](https://github.com/Charmve/Awesome-Lane-Detection)
      - 10.3.1 [常用网络]()
      - 10.3.2 [实战项目 10 - 车道线检测项目实战](https://blog.csdn.net/Charmve/article/details/116678477)
    - 10.4 [镜面检测](https://github.com/Charmve/Mirror-Glass-Detection)
    - 10.5 [图像抠图 Matting](https://github.com/Charmve/TimeWarp)
    - 小结
    - 参考文献
- <b><h4>进阶篇</h4></b>
  - 第 11 章 [可视化和理解](https://charmve.github.io/computer-vision-in-action/#/chapter5/chapter5)
    - 11.1 表征可视化
    - 11.2 对抗实例
    - 11.3 [DeepDream 和风格迁移](/docs/3_进阶篇/chapter12-生成对抗模型/chapter12.3.3_neural-style.md)
    - 11.4 [实战项目 11: PyTorch 如何使用TensorBoard](/docs/3_进阶篇/chapter11-可视化和理解/chapter11-可视化和理解.md)
      - 11.4.1 [创建 TensorBoard](/docs/3_进阶篇/chapter11-可视化和理解/chapter11-可视化和理解.md#1141-创建-tensorboard)
      - 11.4.2 [写入 TensorBoard](/docs/3_进阶篇/chapter11-可视化和理解/chapter11-可视化和理解.md#1142-写入-tensorboard)
      - 11.4.3 [使用 TensorBoard 检查模型](/docs/3_进阶篇/chapter11-可视化和理解/chapter11-可视化和理解.md#1143-使用-tensorboard-检查模型)
      - 11.4.4 [向 TensorBoard 添加 "Projector"](/docs/3_进阶篇/chapter11-可视化和理解/chapter11-可视化和理解.md#1144-向-tensorboard-添加-projector)
      - 11.4.5 [使用 TensorBoard 跟踪模型训练](/docs/3_进阶篇/chapter11-可视化和理解/chapter11-可视化和理解.md#1145-使用-tensorboard-跟踪模型训练)
      - 11.4.6 [使用 TensorBoard 评估训练好的模型](/docs/3_进阶篇/chapter11-可视化和理解/chapter11-可视化和理解.md#1146-使用-tensorboard-评估训练好的模型)
      - [小结](/docs/3_进阶篇/chapter11-可视化和理解/chapter11-可视化和理解.md#小结)
    - 小结
    - 参考文献
  - 第 12 章 [生成对抗模型](https://charmve.github.io/computer-vision-in-action/#/chapter6/chapter6)
    - 12.1 Pixel RNN/CNN
    - 12.2 [自编码器 Auto-encoder](/docs/3_进阶篇/chapter12-生成对抗模型/chapter12_2-自编码器Auto-encoder.md)
    - 12.3 生成对抗网络 GAN
      - 12.3.1 原理
      - 12.3.2 项目实战
        - [StyleGAN](https://github.com/Charmve/VOGUE-Try-On)
        - [StyleGAN 2.0](https://blog.csdn.net/Charmve/article/details/115315353)
      - 12.3.3 [实战项目11 - 样式迁移](/docs/3_进阶篇/chapter12-生成对抗模型/chapter12.3.3_neural-style.md)
    - 12.4 [变分自编码器 Variational Auto-encoder, VAE](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter12-生成对抗模型/chapter12_4-变分自编码器VAE.md)
      - 12.4.1 [概述](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter12-生成对抗模型/chapter12_4-变分自编码器VAE.md#1241-概述)    
      - 12.4.2 [基本原理](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter12-生成对抗模型/chapter12_4-变分自编码器VAE.md#1242-基本原理)        
        - 12.4.2.1 [定义](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter12-生成对抗模型/chapter12_4-变分自编码器VAE.md#1-定义)        
        - 12.4.2.2 [理论基础：三要素](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter12-生成对抗模型/chapter12_4-变分自编码器VAE.md#2-理论基础三要素) 
        - 12.4.2.3 [推导过程](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter12-生成对抗模型/chapter12_4-变分自编码器VAE.md#3-推导过程)            
      - 12.4.3 [VAE v.s. AE 区别与联系](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter12-生成对抗模型/chapter12_4-变分自编码器VAE.md#1243-vae-vs-ae-区别与联系)    
      - 12.4.4 [变分自编码器的代码实现](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter12-生成对抗模型/chapter12_4-变分自编码器VAE.md#1244-变分自编码器的代码实现)    
      - 12.4.5 [卷积变分自编码器的实现与简单应用](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter12-生成对抗模型/chapter12_4-变分自编码器VAE.md#1245-卷积变分自编码器的实现与简单应用)        
      - 小结     
      - [参考文献](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter12-生成对抗模型/chapter12_4-变分自编码器VAE.md#参考文献) 
    - 参考文献
  - 第 13 章 [深度增强学习](https://charmve.github.io/computer-vision-in-action/#/chapter6/chapter6)
    - 13.1 方法梯度，硬性关注
    - 13.2 Q - 学习，评价器
    - 小结
    - 参考文献
  - 第 14 章 [视频理解](https://charmve.github.io/computer-vision-in-action/#/chapter1/chapter1)
    - 14.1 概述
    - 14.2 行为理解
    - 14.3 主流方法
    - 小结
    - 参考文献
  - 第 15 章 [迁移学习](./docs/3_进阶篇/chapter15_迁移学习)
    - 15.1 [概述](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/3_进阶篇/chapter15_迁移学习/chapter15_迁移学习概述.md#151-迁移学习概述)
      - 15.1.1 [背景](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/3_进阶篇/chapter15_迁移学习/chapter15_迁移学习概述.md#1511-背景)
      - 15.1.2 [定义及分类](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/3_进阶篇/chapter15_迁移学习/chapter15_迁移学习概述.md#1512-定义及分类)
      - 15.1.3 [关键点](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/3_进阶篇/chapter15_迁移学习/chapter15_迁移学习概述.md#1513-关键点)
    - 15.2 [基于实例的迁移](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/3_进阶篇/chapter15_迁移学习/chapter15_迁移学习概述.md#152-基于实例的迁移)
    - 15.3 [基于特征的迁移](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/3_进阶篇/chapter15_迁移学习/chapter15_迁移学习概述.md#153-基于特征的迁移)
    - 15.4 [基于共享参数的迁移](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/3_进阶篇/chapter15_迁移学习/chapter15_迁移学习概述.md#154-基于共享参数的迁移)
    - 15.5 [深度学习和迁移学习结合](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/3_进阶篇/chapter15_迁移学习/chapter15_迁移学习概述.md#155-深度学习和迁移学习结合)
    - 15.7 [实战项目12 - 蚂蚁和蜜蜂的分类问题](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/3_进阶篇/chapter15_迁移学习/chapter15_迁移学习的应用.md)
      - 15.7.1 [迁移学习在计算机视觉领域的应用](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/3_进阶篇/chapter15_迁移学习/chapter15_迁移学习的应用.md#1571-迁移学习在计算机视觉领域的应用)
      - 15.7.2 [实战项目: 蚂蚁和蜜蜂的分类问题](https://github.com/Charmve/computer-vision-in-action/tree/main/docs/3_进阶篇/chapter15_迁移学习/chapter15_迁移学习的应用.md#1572-实战项目-蚂蚁和蜜蜂的分类问题)
    - [小结](#小结)
    - [参考文献](#参考文献)
  - 第 16 章 [计算机视觉中的注意力机制 Attention is All You Need](./notebooks/14_Attention.ipynb)
    - 16.1 概述
    - 16.2 Attention with RNNs
    - 16.3 [Self-attention 自注意力](https://mp.weixin.qq.com/s/nUd7YtCci1_AwQ4nOwK9bA)
    - 16.4 软注意力（soft-attention）
      - 16.4.1 空间域注意力
      - 16.4.2 通道注意力
      - 16.4.3 Positional encoding
      - 16.4.4 混合域模型
      - 16.4.5 Masked attention
      - 16.4.6 Multi-head attention
    - 16.5 强注意力（hard attention）
    - 16.6 [Attention九层塔 - 注意力机制的九重理解](/docs/3_进阶篇/chapter16-注意力机制%20Attention%20is%20All%20You%20Need/chapter16_Attention-is-All-You-Need.md)
    - 小结
    - 参考文献
  - 第 17 章 [跨界模型 Transformer](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md)
    - 17.1 [思想和框图](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#%E4%B8%80%E6%80%9D%E6%83%B3%E5%92%8C%E6%A1%86%E5%9B%BE)
    - 17.2 [实现细节](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#二实现细节)
      - [17.2.1 Encoder](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#2-1-Encoder)
      - [17.2.2 Decoder](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#2-2-Decoder)
      - [17.2.3 Self-Attention](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#2-3-Self-Attention)
      - [17.2.4 Multi-Headed Attention](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#2-4-Multi-Headed-Attention)
      - [17.2.5 Positional Encoding](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#2-5-Positional-Encoding)
    - 17.3 [应用任务和结果](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#三-应用任务和结果)
      - 17.3.1 [NLP领域](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#3-1-NLP领域)
      - 17.3.2 [CV领域](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#3-2-CV领域)
        - 17.3.2.1 [检测DETR](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#3-2-1-检测DETR)
        - 17.3.2.2 [分类ViT](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#3-2-2-分类ViT)
        - 17.3.2.3 [分割SETR](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#3-2-3-分割SETR)
        - 17.3.2.4 [Deformable-DETR](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#3-2-4-Deformable-DETR)
    - 17.4 [优点及分析](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#四-优点及分析)
    - 17.5 [缺点及分析](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#五-缺点及分析)
    - [实战项目 13 - 基于transformer的视频实例分割网络VisTR (CVPR2021)](https://blog.csdn.net/Charmve/article/details/115339803)
    - 小结
    - [参考文献](https://github.com/Charmve/computer-vision-in-action/blob/main/docs/3_%E8%BF%9B%E9%98%B6%E7%AF%87/chapter17-%E8%B7%A8%E7%95%8C%E6%A8%A1%E5%9E%8B%20Transformer/chapter17_Transformer.md#六-参考文献)
  - 第 18 章 [知识蒸馏](https://mp.weixin.qq.com/s/e3c_-rs2rncmWhbm-cU5rA)
    - 18.1 概要
    - 18.2 KD主要方法
      - 18.2.1 Logits(Response)-based Knowledge
      - 18.2.2 Feature-based Knowledge
      - 18.2.3 Relation-based Knowledge
    - 18.3 知识蒸馏的应用 NLP-BERT
    - 18.4 常见疑问解答
    - [实战项目 14 - 支付宝CVPR细粒度视觉分类挑战赛夺冠方案解读](https://mp.weixin.qq.com/s/RTkBQJ7Uj86Wxt7HmwWKzA)
    - 小结
    - [参考文献](#参考文献)
  - 第 19 章 [Normalization 模型](https://blog.csdn.net/Charmve/article/details/107650487)
    - 19.1 从Mini-Batch SGD说起
    - 19.2 Normalization到底是在做什么
    - 19.3 Batch Normalization如何做
      - 19.3.1 前向神经网络中的BN
      - 19.3.2 CNN网络中的BN
      - 19.3.3 Batch Norm的四大罪状
    - 19.4 Layer Normalization、Instance Normalization及Group Normalization
      - 19.4.1 Layer Normalization
      - 19.4.2 Instance Normalization
      - 19.4.3 Group Normalization
      - 19.4.4 用一个故事来总结
    - 19.5 Normalization操作的Re-Scaling不变性
    - 19.6 Batch Normalization为何有效
    - 小结
    - [参考文献](#参考文献)
  - 第 20 章 [模型压缩与裁剪](https://mp.weixin.qq.com/s/e3c_-rs2rncmWhbm-cU5rA) 
    - 20.1 概述
    - 20.2 模型压缩
      - 20.2.1 线性或非线性量化（1990~2014 - 至今）
      - 20.2.2 结构或非结构剪枝（1989~2014 - 至今
      - 20.2.3 网络结构搜索（2016 - 至今）
      - 20.2.4 权重矩阵的低秩分解（）
      - 20.2.5 知识蒸馏（2014-至今）
    - 20.3 模型优化加速
      - 20.3.1 Op-Level 的快速算法
      - 20.3.2 Layer0-level 的快速算法
      - 20.3.3 硬件计算单元优化算法
      	- CPU、GPU和NPU
      	- ASIC 和 FPGA
      	- PIM（NDP）
    - 20.4 优化工具与库
      - 20.4.1 TensorRT Nvidia)
      - 20.4.2 TVM (Tensor Virtual Machine)
      - 20.4.3 Tensor Comprehension  (Facebook)
      - 20.4.4 Distiller (Intel)
    - 小结
    - [参考文献](#六-参考文献)
  - <b><h4>附录</h4></b>
    - A 矩阵
    - B [常用激活函数总结](https://mp.weixin.qq.com/s?__biz=MzIxMjg1Njc3Mw==&mid=2247484495&idx=1&sn=0bbb2094d93169baf20eedb284bc668f)
    - C [梯度下降法](https://blog.csdn.net/Charmve/article/details/106089198)
  - <b><h4>[后记](/docs/book_postscript.md)</h4></b>
  - <b><h4>[参考文献](#参考文献-1)</h4></b>

<br>
- <b>更新中 ...</b>

</tr>
</td>
</table>

## 如何食用

1. Install
```
pip3 install -r requirements.txt
```
2. Set up
```
python3 setup.py
```
3. 按书中内容先后顺序逐章阅读，或者选取特定章节祥读 📁 <code>docs/</code> <sup>1</sup>，动手实践章节代码，在代码文件 📁 <code>code/</code> <sup>2</sup> 下找到对应代码，本地测试或者Colab 📁 <code>notebooks/</code> <sup>3</sup> 在线测试。

### 🔍 Browse Folders
- 📁 <code>L0CV/</code> - 专为本项目建立的 L0CV 包 
- 📁 <code>code/</code> - 书中完整代码
- 📁 <code>datasets/</code> - 本书所用数据集
  - 📁 <code>images/</code> - 经典图像处理图片
- 📁 <code>docs/</code> - 全书按照一下几个篇目进行编写
  - 📁 <code>0_绪论/</code> - 全书绪论
  - 📁 <code>1_理论篇/</code> - 基本理论和算法基础
  - 📁 <code>2_实战篇/</code> - 项目实战教程
  - 📁 <code>3_进阶篇/</code> - 最新CV模型和算法
  - 📁 <code>附件/</code> - 所需数学和统计学知识
- 📁 <code>models/</code> - 封装可用的网络模型
- 📁 <code>notebooks/</code> - 全书 Colab notebook，可在线测试
  - 📄 <code>01_Notebooks.ipynb</code> - Notebooks 基本介绍和使用
  - 📄 <code>02_Python.ipynb</code> - Python 编程简明教程
  - 📄 <code>03_NumPy.ipynb</code> - NumPy 的使用
  - 📄 <code>04_Pandas.ipynb</code> - Pandas 的使用
  - 📄 <code>05_PyTorch.ipynb</code> - PyTorch 的使用
  - 📄 <code>06_Linear_Regression.ipynb</code>  - 线性回归
  - ...
  - 📄 <code>16_CVAE.ipynb</code> - 卷积变分自编码器
  - 📄 <code>17_TL-ants-bees-classification.ipynb</code>  - 迁移学习的应用：蚂蚁和蜜蜂的分类问题
- 📁 <code>res/</code> - ui 图片
- 📄 <code>README.md</code> - 全书介绍及目录


## 常见问题

- **在线教程页面无法打开**: 

    测试中存在部分人打不开在线教程的情况。

    部分小伙伴反馈尝试切换浏览器后可以正常打开了，如果仍然不行，最有效的解决办法是科学上网。

- **无法加载图片的解决办法**: 

    根本解决办法还是科学上网，也可以尝试修改host文件看下是否能解决。

    [windows解决方案：修改host文件](https://blog.csdn.net/u011583927/article/details/104384169)

- **公式无法正常显示解决办法**：

    GitHub中的Markdown原生是不支持LATEX公式显示的，如果你喜欢在本项目中直接浏览教程，可以安装Chrome的`MathJax Plugin for Github`插件让大部分公式正常显示。而docs文件夹已经利用docsify被部署到了GitHub Pages上，包含公式的章节强力建议使用 [《计算机视觉实战演练：算法与应用》 在线阅读](https://charmve.github.io/computer-vision-in-action) 进行学习。
    
    当然如果你还想跑一下运行相关代码的话还是得把本项目clone下来，然后运行code文件夹下相关代码。

## 致谢
<a href="https://maiweiai.github.io/"><img src="https://github.com/Charmve/computer-vision-in-action/blob/main/res/ui/maiwei_ai.png" height="36" alt="迈微AI研习社" title="迈微AI研习社"> </a> <a href="https://madewithml.com/"><img src="https://madewithml.com/static/images/logo.png" height="30" alt="Made With ML" title="Made With ML"> </a> &nbsp;&nbsp; <a href="https://www.epubit.com/"><img src="https://cdn.ptpress.cn/pubcloud/3/app/0718A6B0/cover/20191204BD54009A.png" height="30" alt="异步社区" title="异步社区"> </a>  &nbsp;&nbsp; <a href="https://360.cn"><img src="https://p3.ssl.qhimg.com/t011e94f0b9ed8e66b0.png" height="36" alt="奇虎360" title="奇虎360"> </a>

## 参考文献

--> Go to see [<b>REFERENCE.md</b>](docs/REFERENCE.md), &nbsp; 感谢前人的杰出工作，我才得以写出此书。

## 关注我们
<div align=center>
<p>扫描下方二维码，然后回复关键词“<b>计算机视觉实战教程</b>”，即可加入“读者交流群”</p>
<img src="https://user-images.githubusercontent.com/29084184/116501908-a63da600-a8e4-11eb-827c-7772655e0079.png" width = "250" height = "270" alt="迈微AI研习社是一个专注AI领域的开源组织，作者系CSDN博客专家，主要分享机器学习算法、计算机视觉等相关内容，每周研读顶会论文，持续关注前沿技术动态。底部有菜单分类，关注我们，一起学习成长。">
</div>

- 若本书里没有你想要理论和实战项目，或者你发现本书哪个地方有错误，请毫不犹豫地去本仓库的 Issues（ 地址 https://github.com/charmve/computer-vision-in-action/issues ）进行反馈，在对应版块提交你希望补充的内容或者勘误信息，作者通常会在 24 小时以内给您回复，超过 24 小时未回复的话可以邮件联系我（微信 MaiweiE_com）；
- 同时，我也欢迎大家加入本项目的建设中来，欢迎 [pull request](https://github.com/charmve/computer-vision-in-action/pulls)！

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"> 知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a> 进行许可。

## CONTRIBUTION

作者会尽最大努力来确保书中内容的准确性，但难免会存在疏漏与不妥之处。欢迎您将发现的问题反馈给我们，帮助提升本开放资源的质量。

<table>
<tr>
<td>
<div class="f5 contribution">
  <h3 class="f4">Help us make these docs great!</h3>
  <p class="color-text-secondary f6">All <b><i>VC-action</i></b> docs are open source. See something that's wrong or unclear? <a href="https://github.com/Charmve/computer-vision-in-action/pulls">Submit a pull request</a>.</p>
  <a class="btn btn-outline" href="https://github.com/Charmve/computer-vision-in-action/edit/main/README.md">
    <svg version="1.1" width="16" height="16" viewBox="0 0 16 16" class="octicon octicon-git-pull-request" aria-hidden="true"><path fill-rule="evenodd" d="M7.177 3.073L9.573.677A.25.25 0 0110 .854v4.792a.25.25 0 01-.427.177L7.177 3.427a.25.25 0 010-.354zM3.75 2.5a.75.75 0 100 1.5.75.75 0 000-1.5zm-2.25.75a2.25 2.25 0 113 2.122v5.256a2.251 2.251 0 11-1.5 0V5.372A2.25 2.25 0 011.5 3.25zM11 2.5h-1V4h1a1 1 0 011 1v5.628a2.251 2.251 0 101.5 0V5A2.5 2.5 0 0011 2.5zm1 10.25a.75.75 0 111.5 0 .75.75 0 01-1.5 0zM3.75 12a.75.75 0 100 1.5.75.75 0 000-1.5z"></path></svg>
    Make a contribution
  </a></div>
  <br><p class="color-text-secondary f6 mt-2">Or, <a href="https://github.com/Charmve/computer-vision-in-action/blob/main/CONTRIBUTING.md" target="_blank">learn how to contribute.</a></p>


<div>
  <h3 class="mb-2 f4">
    Still need help?
  </h3>
  <a id="ask-community" href="https://github.com/Charmve/computer-vision-in-action/discussions" class="btn btn-outline mr-4 mt-2">
    <svg version="1.1" width="16" height="16" viewBox="0 0 16 16" class="octicon octicon-people" aria-hidden="true"><path fill-rule="evenodd" d="M5.5 3.5a2 2 0 100 4 2 2 0 000-4zM2 5.5a3.5 3.5 0 115.898 2.549 5.507 5.507 0 013.034 4.084.75.75 0 11-1.482.235 4.001 4.001 0 00-7.9 0 .75.75 0 01-1.482-.236A5.507 5.507 0 013.102 8.05 3.49 3.49 0 012 5.5zM11 4a.75.75 0 100 1.5 1.5 1.5 0 01.666 2.844.75.75 0 00-.416.672v.352a.75.75 0 00.574.73c1.2.289 2.162 1.2 2.522 2.372a.75.75 0 101.434-.44 5.01 5.01 0 00-2.56-3.012A3 3 0 0011 4z"></path></svg>
    Ask our community
  </a>
  <a id="contact-us" href="https://github.com/Charmve" class="btn btn-outline mt-2">
    <svg version="1.1" width="16" height="16" viewBox="0 0 16 16" class="octicon octicon-comment-discussion" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 2.75a.25.25 0 01.25-.25h8.5a.25.25 0 01.25.25v5.5a.25.25 0 01-.25.25h-3.5a.75.75 0 00-.53.22L3.5 11.44V9.25a.75.75 0 00-.75-.75h-1a.25.25 0 01-.25-.25v-5.5zM1.75 1A1.75 1.75 0 000 2.75v5.5C0 9.216.784 10 1.75 10H2v1.543a1.457 1.457 0 002.487 1.03L7.061 10h3.189A1.75 1.75 0 0012 8.25v-5.5A1.75 1.75 0 0010.25 1h-8.5zM14.5 4.75a.25.25 0 00-.25-.25h-.5a.75.75 0 110-1.5h.5c.966 0 1.75.784 1.75 1.75v5.5A1.75 1.75 0 0114.25 12H14v1.543a1.457 1.457 0 01-2.487 1.03L9.22 12.28a.75.75 0 111.06-1.06l2.22 2.22v-2.19a.75.75 0 01.75-.75h1a.25.25 0 00.25-.25v-5.5z"></path></svg>
    Contact support
  </a>
</div>
<br>
</td>
</tr>
<table>


## Citation

Use this bibtex to cite this repository:
```
@misc{computer-vision-in-action,
  title={计算机视觉实战演练：算法与应用（Computer Vision in Action）},
  author={Charmve},
  year={2021.06},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/Charmve/computer-vision-in-action}},
}
```

## Stargazers Over Time

[![Stargazers over time](https://starchart.cc/Charmve/computer-vision-in-action.svg)](https://starchart.cc/Charmve/computer-vision-in-action)

<br>
<p align="center">Feel free to ask any questions, open a PR if you feel something can be done differently!</p>
<h2 align="center">🌟Star this repository🌟</h2>
<p align="center">Created by <a href="https://github.com/Charmve">Charmve</a> & <a href="https://github.com/MaiweiAI">maiwei.ai</a> Community | Deployed on <a href="https://app.gitbook.com/@charmve/s/computer-vision-in-action/">GitBook</a></p>

<br>
* <i>Update on Apr 29，2021 @<a href="https://github.com/Charmve" target="_blank">Charmve</a>, 
    <a class="github-button"
        href="https://github.com/Charmve/Surface-Defect-Detection"
        data-icon="octicon-star" data-show-count="true"
        aria-label="Star Charmve/Surface-Defect-Detection on GitHub">Star</a> 
    and 
    <a class="github-button"
        href="https://github.com/Charmve/Surface-Defect-Detection/fork"
        data-icon="octicon-repo-forked" data-show-count="true"
        aria-label="Fork Charmve/Surface-Defect-Detection on GitHub">Fork</a>
</i>
