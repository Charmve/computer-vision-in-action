<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

# 第 8 章 著名数据集及基准

- 第 8 章 [著名数据集及基准](https://charmve.github.io/computer-vision-in-action/#/chapter8/chapter8)
    - 8.1 数据集
        - 8.1.1 [常见数据集](#811-常见数据集)
          - 8.1.1.1 [ImageNet](https://image-net.org/)
          - 8.1.1.2 [MNIST](http://yann.lecun.com/exdb/mnist/)
          - 8.1.1.3 [COCO](https://cocodataset.org/)
          - 8.1.1.4 [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
        - 8.1.2 [Pytorch数据集及读取方法简介](#812-pytorch数据集及读取方法简介)
        - 8.1.3 [数据增强简介](#813-数据增强简介)
        - [小练习]
        - [小结](#小结)
    - 8.2 [基准](chapter8.2_基准BenchMark.md)
    - 8.3 [评价指标](/chapter8.3_评价指标.md)
    - 小结
    - 参考文献

--- 

## 8.3 评价指标

作者: 张伟 (Charmve)

日期: 2021/06/12

<br>

### 8.3.1 常用指标

1. **每个检测物体的分类准确度**；

2. **预测框与真实框的重合度（IOU）**：如果设定IOU的阈值为0.5，当一个预测框与一个真实框的IOU值大于该阈值时，被判定为真阳（TP），反之被判定为假阳（FP）

3. 模型是否找到图片中的所有物体（召回，recall）：如存在某些模型没有预测出的真实框称之为假阴（FN）。

4. **综合得到mAP**：在PascalVOC中，mAP是各类别AP的平均，$precision = TP / (TP + FP)$，指预测框为true的数量比上所有预测框的数量

5. **召回率**：$recall = TP / (TP + FN)$。指找到的某一类别物体的数量比上图像中所有这类物体的数量。

### 8.3.2 真阳、假阳与真阴、假阴详解

- True Positive(TP)：既是正样本又被预测为正样本的个数，即检测正确，检测中的IOU≥阈值。

- False Positive(FP)：负样本被预测为了正样本的个数，即检测错误，检测中的IOU＜阈值。

- False Negative(FN)：既是负样本又被预测为负样本的个数，也即ground truth未被检测到。

- True Negative(TN)：正样本被预测为了负样本的个数。TN最后不会被应用于评价算法的性能。阈值和评价的尺度有关，通常被设定为0.5，0.75或者0.95。

### 8.3.3 IOU（Intersection Over Union）详解

IOU用于计算两个边界框之间的交集。它需要一个ground truth边界框Bgt和一个预测边界框Bp。通过应用IOU，我们可以判断检测是否有效（TP）或不有效（FP）。

IOU由预测边界框和ground truth边界框之间的重叠区域除以它们之间的结合区域得出：

$$IOU = \frac{area(B_p \bigcap B_{gt})}{area(B_p \bigcup  B_{gt})}$$

### 8.3.4 性能指标

评价一个目标检测算法是否有效，我们通常关注精度和速度两个方面。精度的评价指标通常有两个：检测准确率（Precision）以及召回率（Recall）。速度的评价指标通常为检测速度（Speed）。计算检测准确率和召回率的公式如下：

$$Precision = \frac{TP}{TP + FP} = \frac{TP}{all detections}$$
$$Recall = \frac{TP}{TP + FN} = \frac{TP}{all ground truths}$$

最常用的评价指标为检测平均精度( Average Precision，AP)，它被定义为正确识别的物体数占总识别的物体个数的百分数。而评估所有类别的检测准确度的指标为平均精度均值( Mean Average Precision，mAP)，定义为所有类别检测的平均准确度，通常将mAP作为检测算法性能评估的最终指标。平均召回率( Avreage Recall，AR) 表示正确识别的物体数占测试集中识别的物体个数的百分数。此外，为了评估一个检测器的实时性，通常采用每秒处理帧数(Frames Per Second，FPS)指标评价其执行速度。FPS值越大，说明检测器的实时性越好。【张索非】

表1 COCO 数据集主要评价指标

|指标|含义|
|--|--|
|$AP_{bb}$|$IOU=0.5：0.05：0.95$ 时，AP的取值|
|$AP_{50}$|$IOU=0.5$ 时，AP的取值|
|$AP_{75}$|$IOU=0.75$ 时，AP的取值|
|$AP_{S}$|物体面积小于322时，AP的取值|
|$AP_{M}$|物体面积介于322~962时，AP的取值|
|$AP_{L}$|物体面积大于962时，AP的取值|

