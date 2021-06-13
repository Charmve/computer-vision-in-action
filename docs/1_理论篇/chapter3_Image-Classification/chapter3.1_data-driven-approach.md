<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

# 第 3 章 图像分类

作者: 张伟 (Charmve)

日期: 2021/06/13

- 第 3 章 [图像分类](./docs/1_理论篇/chapter3_Image-Classification/)
    - 3.1 [数据驱动方法](https://cs231n.github.io/classification/)
      - 3.1.1 语义上的差别
      - 3.1.2 图像分类任务面临着许多挑战
      - 3.1.3 数据驱动的方法
    - 3.2 [k 最近邻算法](chapter32_knn.md)
      - 3.2.1 [k 近邻模型](chapter32_knn.md#321-k-近邻模型)
      - 3.2.2 [k 近邻模型三个基本要素](chapter32_knn.md#322-k-近邻模型三个基本要素)
      - 3.2.3 [KNN算法的决策过程](chapter32_knn.md#323-k-KNN算法的决策过程)
      - 3.2.4 [k 近邻算法Python实现](chapter32_knn.md#324-k-近邻算法Python实现)
      - 小结
      - 参考文献
    - 3.3 [支持向量机](chapter3.3_支持向量机.md)
      - 3.3.1 概述
      - 3.3.2 线性支持向量机
      - 3.3.3 从零开始实现支持向量机
      - 3.3.4 支持向量机的简洁实现
    - 3.4 [Softmax 回归](chapter3.4_Softmax回归.md)
      - 3.4.1 softmax回归模型
      - 3.4.2 从零开始实现softmax回归
      - 3.4.3 softmax回归的简洁实现
    - 3.5 [逻辑回归 LR](../../../notebooks/07_Logistic_Regression.ipynb)
      - 3.5.1 逻辑回归模型
      - 3.5.2 从零开始实现逻辑回归
      - 3.5.3 逻辑回归的简洁实现
    - 3.6 [实战项目 3 - 表情识别](https://blog.csdn.net/charmve/category_9754344.html)
    - 3.7 [实战项目 4 - 使用卷积神经网络对CIFAR10图片进行分类](http://mp.weixin.qq.com/s?__biz=MzIxMjg1Njc3Mw%3D%3D&chksm=97bef597a0c97c813e185e1bbf987b93d496c6ead8371364fd175d9bac46e6dcf7059cf81cb2&idx=1&mid=2247487293&scene=21&sn=89684d1c107177983dc1b4dca8c20a5b#wechat_redirect)
    - [小结](./docs/1_理论篇/chapter3_Image-Classification/README.md#小结)
    - [参考文献](./docs/1_理论篇/chapter3_Image-Classification/README.md#参考文献)

---

## 3.1 数据驱动方法

### 3.1.1 语义上的差别

图像分类是计算机视觉中一个核心的任务，人脑和计算机中对图像的描述在语义上不同，图像在计算机中以 channel * height * width 的形式描述，如下图3.1所示。

![image](https://user-images.githubusercontent.com/29084184/121795072-d1374b80-cc3f-11eb-9181-dbd3e9ddc67b.png)

图3.1 

### 3.1.2 图像分类任务面临着许多挑战

- 视角变换 Viewpoint variation

![image](https://user-images.githubusercontent.com/29084184/121795071-c7ade380-cc3f-11eb-95be-fb1e0e57fa4f.png)

- 背景混乱 Background Clutter

![image](https://user-images.githubusercontent.com/29084184/121795068-c11f6c00-cc3f-11eb-8b44-2def4c80bd77.png)

- 光照条件 IIIumination

![image](https://user-images.githubusercontent.com/29084184/121795061-b82e9a80-cc3f-11eb-8d05-7a9c1f799432.png)

- 遮挡 Occlusion

![image](https://user-images.githubusercontent.com/29084184/121795060-b1078c80-cc3f-11eb-8c05-5e2eab907d02.png)

- 变形 Deformation

![image](https://user-images.githubusercontent.com/29084184/121795054-a220da00-cc3f-11eb-9dde-777987b148f8.png)

- 类间差异 Intraclass variation

![image](https://user-images.githubusercontent.com/29084184/121795058-a947e800-cc3f-11eb-87c2-ddf9895012ac.png)


### 3.1.3 图像分类器

图像分类器不像排序算法，知道每一步该做什么

![image](https://user-images.githubusercontent.com/29084184/121794985-6423b600-cc3f-11eb-8525-0be6011ac086.png)

![image](https://user-images.githubusercontent.com/29084184/121794991-76055900-cc3f-11eb-8c5d-cabb95f12292.png)


### 3.1.4 数据驱动的方法

分为train和predict两步，用大量的图像数据进行训练，再使用训练得到的模型进行预测，具体为：

- (1) 收集图标和标签的数据集
- (2) 使用机器学习训练一个分类器。
- (3) 使用分类器测试新图片。

最初始的分类器：最近邻算法，下一节将会详细介绍。

![image](https://user-images.githubusercontent.com/29084184/121794976-5837f400-cc3f-11eb-93e0-b7ec8fb998f5.png)


```python

```

```python

```
