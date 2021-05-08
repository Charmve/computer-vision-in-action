<table align="center">
<tr ><td>
<h1> 计算机视觉实战演练：算法与应用 <sup> 📌</sup>
<br><em>Computer Vision in Action</em></h1>

作者：张伟（Charmve）

<p align="center">
<a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/👓-Charmve-blue" alt="GitHub"></a>
<a href="https://github.com/Charmve/computer-vision-in-action"><img src="https://img.shields.io/badge/CV-Action-yellow" alt="CV-Action"></a>
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>
<a href="https://github.com/Charmve/computer-vision-in-action/edit/master/README.md"><img src="https://img.shields.io/github/stars/Charmve/computer-vision-in-action?style=social" alt="Stars"></a>
<a href="https://github.com/Charmve/computer-vision-in-action/edit/master/README.md"><img src="https://img.shields.io/github/forks/Charmve/computer-vision-in-action?style=social" alt="Forks"></a>
</p>

<div id="outputFigDisplay" align="center">
	<img src="../res/maiwei.png" width="240px" alt="logo:maiwei">
</div>

> <h4>在线阅读（内容实时更新）</h4>
> - 地址：https://charmve.github.io/computer-vision-in-action

> <h4>最新版PDF下载</h4>
> - 地址：https://github.com/charmve/computer-vision-in-action/releases

### 目录
- [<b><h4>绪论</h4></b>](https://charmve.github.io/computer-vision-in-action/#/chapter0/chapter0)
  1. 概述
  2. 计算机视觉基本概念
  3. 典型的计算机视觉任务
      - 图像分类 
      - 目标识别与目标检测
      - 实例分割与语义分割
      - 3D 建模
  4. 发展历史回顾
  5. 小练习
  6. 本书编写逻辑
- <b><h4>理论篇</h4></b>
  - 第 1 章 [图像分类](https://charmve.github.io/computer-vision-in-action/#/chapter1/chapter1)
    - 数据驱动方法
    - k 最近邻算法
    - 线性分类
    - 逻辑回归 LR 
    - [实战项目 1 - 手写字分类](https://blog.csdn.net/Charmve/article/details/108531735)
  - 第 2 章 [神经网络](https://charmve.github.io/computer-vision-in-action/#/chapter2/chapter2)
    - 反向传播算法
    - 多层感知器
    - 神经学观点
    - 实战项目 2
  - 第 3 章 [卷积神经网络](https://charmve.github.io/computer-vision-in-action/#/chapter3/chapter3)
    - 概述
    - 卷积和池化
    - 损失函数和优化
    - 线性分类Ⅱ
    - 进阶模型表示与图像特征
    - 优化，随机梯度下降
    - [实战项目 3 - 动手搭建一个卷积神经网络](https://blog.csdn.net/Charmve/article/details/106076844)
  - 第 4 章 [递归神经网络](https://charmve.github.io/computer-vision-in-action/#/chapter4/chapter4)
    - 递归神经网络 RNN
    - 长短期记忆人工神经网络 LSTM
    - 门控循环单元（GRU） 
- <b><h4>实战篇</h4></b>
  - 第 1 章 [深度学习环境搭建](https://charmve.github.io/computer-vision-in-action/#/chapter1/chapter1)
    - PyTorch
    - OpenCV
    - Numpy
  - 第 2 章 [经典卷积神经网络架构：原理与PyTorch实现](https://charmve.github.io/computer-vision-in-action/#/chapter5/chapter5)
    - AlexNet
    - VGG
    - GoogleNet
    - ResNet 
    - U-Net 
    - SegNet 
    - Mask-RCNN
    - Full-CNN
    - 实战项目 4
  - 第 3 章 [著名数据集](https://charmve.github.io/computer-vision-in-action/#/chapter6/chapter6)
    - ImageNet
    - COCO
    - CIFAR-10
  - 第 4 章 [检测与分割实战项目](https://charmve.github.io/computer-vision-in-action/#/chapter6/chapter6)
    - 语义分割
      - [语义分割 PyTorch 版](https://github.com/Charmve/Semantic-Segmentation-PyTorch)
      - 实战项目 6
    - 目标检测
      - 常用网络
      - 实战项目 7
    - 实例分割 
      - 常用网络 
      - 实战项目 8
      - 新方法：滑动窗口
  - 第 5 章 [图像分类项目实战](https://charmve.github.io/computer-vision-in-action/#/chapter6/chapter6)
    - [手写字识别](https://blog.csdn.net/Charmve/article/details/108531735)
    - [文本检测](https://github.com/Charmve/Scene-Text-Detection)
    - [车道线检测](https://github.com/Charmve/Awesome-Lane-Detection)
      - 常用网络 
      - 实战项目 9
    - [镜面检测](https://github.com/Charmve/Mirror-Glass-Detection)
    - [图像抠图 Matting](https://github.com/Charmve/TimeWarp)
- <b><h4>进阶篇</h4></b>
  - 第 1 章 [卷积神经网络进化](https://charmve.github.io/computer-vision-in-action/#/chapter1/chapter1)

  - 第 2 章 [可视化和理解](https://charmve.github.io/computer-vision-in-action/#/chapter5/chapter5)
    - 表征可视化
    - 对抗实例
    - DeepDream 和风格迁移
    - 实战项目 10
  - 第 3 章 [生成对抗模型](https://charmve.github.io/computer-vision-in-action/#/chapter6/chapter6)
    - Pixel RNN/CNN
    - 自编码器 Auto-encoder
    - 生成对抗网络 GAN
      - 原理
      - 项目实战
        - [StyleGAN](https://github.com/Charmve/VOGUE-Try-On)
        - [StyleGAN 2.0](https://blog.csdn.net/Charmve/article/details/115315353)
  - 第 4 章 [深度增强学习](https://charmve.github.io/computer-vision-in-action/#/chapter6/chapter6)
    - 方法梯度，硬性关注
    - Q - 学习，评价器
  - 第 5 章 [迁移学习]()
  - 第 6 章 [注意力机制 Attention is All You Need](./notebooks/14_Attention.ipynb)
  - 第 7 章 [跨界模型 Transformer](https://blog.csdn.net/charmve/category_10954850.html)
  - 第 8 章 [知识蒸馏]()
  - 第 9 章 [Normalization 模型]()

- <b>更新中 ...</b>
</tr>
</td>
</table>

## 致谢
<a href="https://maiweiai.github.io/"><img src="../res/maiwei_ai.png" height="36" alt="迈微AI研习社" title="迈微AI研习社"> </a> <a href="https://madewithml.com/"><img src="https://madewithml.com/static/images/logo.png" height="30" alt="Made With ML" title="Made With ML"> </a> &nbsp;&nbsp;<a href="https://www.epubit.com/"><img src="https://cdn.ptpress.cn/pubcloud/3/app/0718A6B0/cover/20191204BD54009A.png" height="30" alt="异步社区" title="异步社区"> </a>  &nbsp;&nbsp; <a href="https://360.cn"><img src="https://p3.ssl.qhimg.com/t011e94f0b9ed8e66b0.png" height="36" alt="奇虎360" title="奇虎360"> </a> 

## 参考文献

--> [<b>REFERENCE.md</b>](REFERENCE.md)
