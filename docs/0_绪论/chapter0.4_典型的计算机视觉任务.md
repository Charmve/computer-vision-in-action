<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

# 第 0 章 计算机视觉概述

- 第 0 章 计算机视觉概述
    - 0.1 [概述](chapter0.1_概述.md)
      - 0.1.1 什么是计算机视觉
      - 0.1.2 计算机视觉解决什么问题
      - 0.1.3 行业应用
    - 0.2 [计算机视觉基本概念](docs/0_绪论/chapter0.2_计算机视觉基本概念.md)
    - 0.3 [发展历史回顾](docs/0_绪论/chapter0.3_发展历史回顾.md)
    - 0.4 [典型的计算机视觉任务](docs/0_绪论/chapter0.4_典型的计算机视觉任务.md)
      - 0.4.1 图像分类 
      - 0.4.2 目标识别与目标检测
      - 0.4.3 实例分割与语义分割
      - 0.4.4 3D 建模
	- 0.5 [国内外优秀的计算机视觉团队汇总](docs/0_绪论/chapter0.5_国内外优秀的计算机视觉团队汇总.md)
    - 小练习
    - 小结
    - 参考文献
  
---

## 0.4 典型的计算机视觉任务

作者: 张伟 (Charmve)

日期: 2021/06/13


### 0.4.1 [图像分类](../../1_理论篇/chapter3_Image-Classification/)

### 0.4.2 目标识别与目标检测

#### (1) 目标识别

#### （2）训练目标检测模型

- **Viola–Jones 方法**

有很多种方法可以解决目标检测问题。很多年来，Paul Viola 和 Michael Jones 在论文《Robust Real-time Object Detection》中提出的方法成为流行的方法。

尽管该方法可用来检测大量对象类别，但它最初是受人脸检测目标的启发。该方法快速、直接，是傻瓜相机中所使用的算法，它可以在几乎不浪费处理能力的情况下执行实时人脸检测。

该方法的核心特征是：基于哈尔特征与大量二分类器一起训练。哈尔特征表示边和线，计算简单。

![image](https://user-images.githubusercontent.com/29084184/121795297-09d82480-cc42-11eb-8174-c05f11b95ae8.png)

图0.4 哈尔特征

（图源：https://docs.opencv.org/3.4.3/haar_features.jpg）

尽管比较基础，但在人脸检测这一特定案例下，这些特征可以捕捉到重要元素，如鼻子、嘴或眉间距。该监督方法需要很多正类和负类样本。

- **基于卷积神经网络的方法**

深度学习变革了机器学习，尤其是计算机视觉。目前基于深度学习的方法已经成为很多计算机视觉任务的前沿技术。

其中，R-CNN 易于理解，其作者提出了一个包含三个阶段的流程：

1. 利用区域候选（region proposal）方法提取可能的对象。

2. 使用 CNN 识别每个区域中的特征。

3. 利用支持向量机（SVM）对每个区域进行分类。

![image](https://user-images.githubusercontent.com/29084184/121795344-63405380-cc42-11eb-8f2e-ada2c71df48c.png)

图0.5 R-CNN 架构

（图源：https://arxiv.org/abs/1311.2524）

该区域候选方法最初由论文《Selective Search for Object Recognition》提出，尽管 R-CNN 算法并不在意使用哪种区域候选方法。步骤 3 非常重要，因为它减少了候选对象的数量，降低了计算成本。

这里提取的特征没有哈尔特征那么直观。总之，CNN 可用于从每个区域候选中提取 4096 维的特征向量。鉴于 CNN 的本质，输入应该具备同样的维度。这也是 CNN 的弱点之一，很多方法解决了这个问题。回到 R-CNN 方法，训练好的 CNN 架构要求输入为 227 × 227 像素的固定区域。由于候选区域的大小各有不同，R-CNN 作者通过扭曲图像的方式使其维度满足要求。


![image](https://user-images.githubusercontent.com/29084184/121795352-6fc4ac00-cc42-11eb-8d80-0a823ef382ed.png)

图0.6 满足 CNN 输入维度要求的扭曲图像示例。



### 0.4.3 实例分割与语义分割


### 0.4.4 3D 建模
