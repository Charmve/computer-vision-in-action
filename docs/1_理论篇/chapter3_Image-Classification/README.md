<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

# 第 3 章 图像分类

- 第 3 章 [图像分类](https://charmve.github.io/computer-vision-in-action/#/chapter3/chapter3)
    - 3.1 [数据驱动方法](https://cs231n.github.io/classification/)
      - 3.1.1 语义上的差别
      - 3.1.2 图像分类任务面临着许多挑战
      - 3.1.3 数据驱动的方法
    - 3.2 [k 最近邻算法](/chapter32_knn.md)
      - 3.2.1 k 近邻模型
      - 3.2.2 k 近邻模型三个基本要素
      - 3.2.3 KNN算法的决策过程
      - 3.2.4 k 近邻算法Python实现
      - 小结
      - 参考文献
    - 3.3 [支持向量机](./docs/1_理论篇/chapter3_Image-Classification/chapter3.3.1_支持向量机.md)
      - 3.3.1 概述
      - 3.3.2 线性支持向量机
      - 3.3.3 从零开始实现支持向量机
      - 3.3.4 支持向量机的简洁实现
    - 3.4 [Softmax 回归](./docs/1_理论篇/chapter3_Image-Classification/chapter3.3.2_Softmax回归.md)
      - 3.4.1 softmax回归模型
      - 3.4.2 从零开始实现softmax回归
      - 3.4.3 softmax回归的简洁实现
    - 3.5 [逻辑回归 LR](../../../notebooks/07_Logistic_Regression.ipynb)
      - 3.5.1 逻辑回归模型
      - 3.5.2 从零开始实现逻辑回归
      - 3.5.3 逻辑回归的简洁实现
    - 3.6 [实战项目 3 - 表情识别](https://blog.csdn.net/charmve/category_9754344.html)
    - 3.7 [实战项目 4 - 使用卷积神经网络对CIFAR10图片进行分类](http://mp.weixin.qq.com/s?__biz=MzIxMjg1Njc3Mw%3D%3D&chksm=97bef597a0c97c813e185e1bbf987b93d496c6ead8371364fd175d9bac46e6dcf7059cf81cb2&idx=1&mid=2247487293&scene=21&sn=89684d1c107177983dc1b4dca8c20a5b#wechat_redirect)
    - [小结](#小结)
    - [参考文献](#参考文献)

---

点击每个章节的超链接可查看对应章节，
<br>

## 图像分类

**目标**：这一章我们将介绍图像分类问题。所谓图像分类问题，就是已有固定的分类标签集合，然后对于输入的图像，从分类标签集合中找出一个分类标签，最后把分类标签分配给该输入图像。虽然看起来挺简单的，但这可是计算机视觉领域的核心问题之一，并且有着各种各样的实际应用。在后面的课程中，我们可以看到计算机视觉领域中很多看似不同的问题（比如物体检测和分割），都可以被归结为图像分类问题。

例子：以下图为例，图像分类模型读取该图片，并生成该图片属于集合 {cat, dog, hat, mug}中各个标签的概率。需要注意的是，对于计算机来说，图像是一个由数字组成的巨大的3维数组。在这个例子中，猫的图像大小是宽248像素，高400像素，有3个颜色通道，分别是红、绿和蓝（简称RGB）。如此，该图像就包含了248X400X3=297600个数字，每个数字都是在范围0-255之间的整型，其中0表示全黑，255表示全白。我们的任务就是把这些上百万的数字变成一个简单的标签，比如“猫”。

![image](https://user-images.githubusercontent.com/29084184/121795127-3db24a80-cc40-11eb-9859-d1fab3e10019.png)

图3.0 图像分类的任务，就是对于一个给定的图像，预测它属于的那个分类标签（或者给出属于一系列不同标签的可能性）。图像是3维数组，数组元素是取值范围从0到255的整数。数组的尺寸是宽度x高度x3，其中这个3代表的是红、绿和蓝3个颜色通道。


ENJOY！


<br><br>

作者: 张伟 (Charmve)

日期: 2021/06/13

