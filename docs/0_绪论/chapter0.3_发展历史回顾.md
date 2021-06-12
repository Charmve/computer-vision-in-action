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
  - 0.2 [计算机视觉基本概念](chapter0.2_计算机视觉基本概念.md)
  - 0.3 发展历史回顾
  - 0.4 [典型的计算机视觉任务](chapter0.4_典型的计算机视觉任务.md)
      - 图像分类 
      - 目标识别与目标检测
      - 实例分割与语义分割
      - 3D 建模
  - 0.5 [国内外优秀的计算机视觉团队汇总](chapter0.5_国内外优秀的计算机视觉团队汇总.md)
  - 小练习
  - 小结
  - 参考文献
  
---


<br>

![image](https://user-images.githubusercontent.com/29084184/121769873-9b8b5780-cb98-11eb-91e9-e367753ef9cc.png)

图1 Hubel and Wiesel Experiment

![image](https://user-images.githubusercontent.com/29084184/121769995-4439b700-cb99-11eb-9226-991f2a620ef7.png)

图2 LeCun等人用神经网络进行手写字识别

![image](https://user-images.githubusercontent.com/29084184/121770500-36396580-cb9c-11eb-972b-ceb1422cb740.png)

图3 （左）八张 ILSVRC-2010 测试图像和我们模型认为最有可能的五个标签。 正确的标签写在每张图像下，分配给正确标签的概率也用红色条显示（如果它恰好在前 5 个）。 （右）第一列中的五个 ILSVRC 2010 测试图像。 剩余的列显示了六个训练图像，它们在最后一个隐藏层中产生特征向量，与测试图像的特征向量的欧几里德距离最小。


## 0.3 发展历史回顾

作者: 张伟 (Charmve)

日期: 2021/04/29

| 年份  | 事件 | 相关论文/Reference | 
|--|--|--|
|  1959      | Hubel 和Wiesel 对猫进行了实验（为了研究视觉的工作方式） | Hubel, D. H., & Wiesel, T. N. (1959). Receptive fields of single neurones in the cat's striate cortex. The Journal of physiology, 148(3), 574-591. | 
|  1963       | 计算机视觉领域的先驱Larry Roberts 在他的博士论文中试图提取「积木世界（Block World）」的3D 几何信息 | Roberts, L. S. (1963).ERGASILUS NERKAE N. SP. (COPEPODA: CYCLOPOIDA) FROM BRITISH COLUMBIA WITH A DISCUSSION OF THE COPEPODS OF THE E. CAERULEUS GROUP.Journal of Zoology, 41:115-124. | 
|  1966       | Summer Vision 项目启动，人们普遍认为这就意味着计算机视觉的诞生 | Papert, S. A. (1966). The summer vision project.|
|  1982       | David Marr 的《视觉（Vision）》一书影响和激励了这一领域的一代研究者，该书暗示了以「层」的方式看待图像的思想 | Marr, D., & Vision, A. (1982). A computational investigation into the human representation and processing of visual information. WH San Francisco: Freeman and Company, 1(2). |
| 20世纪80年代 | 光学字符识别（OCR）技术开始在工业应用中使用| Tanaka H.; Hirakawa Y.; Kaneku S. (1982). Recognition of distorted patterns using Viterbi algorithm. IEEE T. Pattern Anal. Mach. lntell. 4, 18-25.//Shildhar M.; Badreldin A.(1985). A high accuracy syntactic recognition algorithm for handwritten numerals, IEEE T. Syst. Man Cyb. 15.//Tampi K.R.; Chetlur S. S. (1986). Segmentation of handwritten characters, Proc. 8th Int. J. Conf. Pattern Recognition. pp684-686.|
|   1990      | 神经网络技术（CNN）开始被用于手写识别 | LeCun, Y., Boser, B. E., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W. E., & Jackel, L. D. (1990). Handwritten digit recognition with a back-propagation network. In *Advances in neural information processing systems* (pp. 396-404). |
|   2001      | Viola 和Jones 开始了面部检测研究；计算机视觉的研究重心发生转移，从建模物体的 3D 形状转向了识别物体是什么 | Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. In Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on (Vol. 1, pp. I-I). IEEE. |
|   2009      | ImageNet 建立 | Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009, June). Imagenet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on (pp. 248-255). IEEE.|
|   2012      | AlexNet 在ImageNet 竞赛中获胜 | Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In *Advances in neural information processing systems* (pp. 1097-1105).|
|   2014      | Christian Szegedy等人提出了对抗样本（Adversarial Examples）这个概念 | Szegedy, C.; Zaremba, W. (2014).Intriguing properties of neural networks.arXiv:1312.6199v4.|
|   2018      | 加拿大约克大学、Ryerson 大学的研究者们提出了使用「双流卷积神经网络」的动画生成方法，其参考了人类感知动态纹理画面的双路径模式。 | Tesfaldet, M.; Brubaker, M. A.; Derpanis, K. G. (2018).Two-Stream Convolutional Networks for Dynamic Texture Synthesis.arXiv:1706.06982.|
