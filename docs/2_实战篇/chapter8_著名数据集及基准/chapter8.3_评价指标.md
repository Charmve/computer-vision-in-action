<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

**第 8 章 著名数据集及基准**

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

# 8.3 评价指标

作者: 张伟 (Charmve)

日期: 2021/06/12

<br>

## 8.3.1 常用指标

1. **每个检测物体的分类准确度**；

2. **预测框与真实框的重合度（IOU）**：如果设定IOU的阈值为0.5，当一个预测框与一个真实框的IOU值大于该阈值时，被判定为真阳（TP），反之被判定为假阳（FP）

3. 模型是否找到图片中的所有物体（召回，recall）：如存在某些模型没有预测出的真实框称之为假阴（FN）。

4. **综合得到mAP**：在PascalVOC中，mAP是各类别AP的平均，$precision = TP / (TP + FP)$，指预测框为true的数量比上所有预测框的数量

5. **召回率**：$recall = TP / (TP + FN)$。指找到的某一类别物体的数量比上图像中所有这类物体的数量。

## 8.3.2 真阳、假阳与真阴、假阴详解

- True Positive(TP)：既是正样本又被预测为正样本的个数，即检测正确，检测中的IOU≥阈值。

- False Positive(FP)：负样本被预测为了正样本的个数，即检测错误，检测中的IOU＜阈值。

- False Negative(FN)：既是负样本又被预测为负样本的个数，也即ground truth未被检测到。

- True Negative(TN)：正样本被预测为了负样本的个数。TN最后不会被应用于评价算法的性能。阈值和评价的尺度有关，通常被设定为0.5，0.75或者0.95。

## 8.3.3 IOU（Intersection Over Union）详解

IOU用于计算两个边界框之间的交集。它需要一个ground truth边界框Bgt和一个预测边界框Bp。通过应用IOU，我们可以判断检测是否有效（TP）或不有效（FP）。

IOU由预测边界框和ground truth边界框之间的重叠区域除以它们之间的结合区域得出：

$$IOU = \frac{area(B_p \bigcap B_{gt})}{area(B_p \bigcup  B_{gt})}$$

## 8.3.4 ROC 曲线

**ROC**：横坐标为假阳性率（False Positive Rate，FPR）；纵坐标为真阳性率（True Positive Rate，TPR）

$$ FPR = \frac{FP}{N} \ TPR = \frac{TP}{P} $$

其中P是真实的正样本的数量，N是真实的负样本的数量，TP是P个正样本中被分类器预测为正样本的个数，FP是N个负样本中被预测为正样本的个数。



> 【如何绘制ROC曲线】通过不断移动分类器的“截断点”来生成曲线上的一组关键点。在二分类问题中，模型输出一般是预测样本为正例的概率，在输出最终的正例负例之前，我们需要制定一个阈值。大于该阈值的样本判定为正例，小于该阈值的样本判定为负例。通过动态调整截断点，绘制每个截断点对应位置，再连接所有点得到最终的ROC曲线。

## 8.3.5 AUC 曲线

AUC是指ROC曲线下的面积大小。计算AUC值只要沿着ROC横轴做积分就可以。AUC取值一般在0.5~1之间。AUC越大，分类性能越好。AUC表示预测的正例排在负例前面的概率。

指标想表达的含义，简单来说其实就是随机抽出一对样本（一个正样本，一个负样本），然后用训练得到的分类器来对这两个样本进行预测，预测得到正样本的概率大于负样本概率的概率

![img](https://camo.githubusercontent.com/6249908e3f11116daa52274a71816941676b673102a6e84cb6f151b898fad71b/68747470733a2f2f696d672d626c6f672e6373646e2e6e65742f3230313530393234313533313537383032)

AUC为0.5表明对正例和负例没有区分能力，对于不论真实类别是1还是0，分类器预测为1的概率是相等的。

我们希望分类器达到的效果：对于真实类别为1的样本，分类器预测为1的概率（TPR）要大于真实类别为0而预测类别为1的概率（FPR），即y>x

AUC的计算方法同时考虑了分类器对于正例和负例的分类能力，在样本不平衡的情况下，依然能够对分类器作出合理的评价。



思路：

1. 首先对预测值进行排序，排序的方式用了python自带的函数sorted，详见注释。
2. 对所有样本按照预测值从小到大标记rank，rank其实就是index+1，index是排序后的sorted_pred数组中的索引
3. 将所有正样本的rank相加，遇到预测值相等的情况，不管样本的正负性，对rank要取平均值再相加
4. 将rank相加的和减去正样本排在正样本之后的情况，再除以总的组合数，得到auc



## 8.3.4 性能指标

评价一个目标检测算法是否有效，我们通常关注精度和速度两个方面。精度的评价指标通常有两个：检测准确率（Precision）以及召回率（Recall）。速度的评价指标通常为检测速度（Speed）。计算检测准确率和召回率的公式如下：

$$Precision = \frac{TP}{TP + FP} = \frac{TP}{all detections}$$
$$Recall = \frac{TP}{TP + FN} = \frac{TP}{all ground truths}$$

最常用的评价指标为检测平均精度( Average Precision，AP)，它被定义为正确识别的物体数占总识别的物体个数的百分数。而评估所有类别的检测准确度的指标为平均精度均值( Mean Average Precision，mAP)，定义为所有类别检测的平均准确度，通常将mAP作为检测算法性能评估的最终指标。平均召回率( Avreage Recall，AR) 表示正确识别的物体数占测试集中识别的物体个数的百分数。此外，为了评估一个检测器的实时性，通常采用每秒处理帧数(Frames Per Second，FPS)指标评价其执行速度。FPS值越大，说明检测器的实时性越好。


## 8.3.5 图像质量评价指标
### PSNR

PSNR, Peak Signal-to-Noise Ratio 峰值信噪比。

给定一个大小为 $m×n$ 的干净图像 $I$ 和噪声图像 $K$ ，均方误差 $MSE$ 定义为：

$$
MSE = {1\over mn} \sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2 
$$

然后 $PSNR(dB)$ 就定义为：

$$
PSNR = 10 \cdot log_{10}{{MAX_I}^2\over MSE} 
$$

其中 ${MAX_I}^2$ 为图片可能的最大像素值。如果每个像素都由 8 位二进制来表示，那么就为 255。通常，如果像素值由 $B$ 位二进制来表示，那么 $ MAX_I = 2^B - 1$。

一般地，针对 ``uint8`` 数据，最大像素值为 255,；针对浮点型数据，最大像素值为 1。

上面是针对灰度图像的计算方法，如果是彩色图像，通常有三种方法来计算。

- 分别计算 RGB 三个通道的 PSNR，然后取平均值。
- 计算 RGB 三通道的 MSE ，然后再除以 3 。
- 将图片转化为 YCbCr 格式，然后只计算 Y 分量也就是亮度分量的 PSNR。

其中，第二和第三种方法比较常见。

```python3
# im1 和 im2 都为灰度图像，uint8 类型

# method 1
diff = im1 - im2
mse = np.mean(np.square(diff))
psnr = 10 * np.log10(255 * 255 / mse)

# method 2
psnr = skimage.measure.compare_psnr(im1, im2, 255)
```

备注：``compare_psnr(im_true, im_test, data_range=None)`` 函数原型可见, [此处](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_psnr)。

针对超光谱图像，我们需要针对不同波段分别计算 PSNR，然后取平均值，这个指标称为 MPSNR。

### SSIM

SSIM，Structural SIMilarity 结构相似性。$SSIM$ 公式基于样本 $x$ 和 $y$ 之间的三个比较衡量：**亮度 (luminance)**、**对比度 (contrast)** 和**结构 (structure)**。

$$
l(x,y) = {2 \mu _x \mu _y + c_1 \over {\mu _x}^2 + {\mu _y}^2 + c_1}\\ 
$$
$$
c(x,y) = {2 \sigma _x \sigma _y + c_2 \over {\sigma _x}^2 + {\sigma _y}^2 + c_2}\\ 
$$
$$
s(x,y) = {\sigma _{xy} + c_3 \over \sigma _x \sigma _y + c_3}
$$

$$
\left\{\begin{matrix}
l(x,y) = {2 \mu _x \mu _y + c_1 \over {\mu _x}^2 + {\mu _y}^2 + c_1}\\ 
c(x,y) = {2 \sigma _x \sigma _y + c_2 \over {\sigma _x}^2 + {\sigma _y}^2 + c_2}\\ 
s(x,y) = {\sigma _{xy} + c_3 \over \sigma _x \sigma _y + c_3}
\end{matrix}\right.
$$

一般取 $ c_3 = c_2 /2 $。

- $\mu _x$ 为 $x$ 的均值
- $\mu _x$ 为 $y$ 的均值
- ${\sigma_x}^2$ 为 $x$ 的方差
- ${\sigma_y}^2$ 为 $y$ 的方差
- $\sigma _{xy}$ 为 $x$ 和 $y$ 的协方差
- $c_1 = (k_{1}L)^2$, $c_2=(k_{2}L)^2$ 为两个常数，避免除零
- $L$ 为像素值的范围，$2^B-1$
- $k_1= 0.01, k_2 = 0.03$ 为默认值

那么

$$
SSIM(x,y) = [l(x,y)^\alpha \cdot c(x,y)^\beta  \cdot s(x,y)^\gamma]
$$

将$\alpha, \beta, \gamma$设为 1，可以得到

$$
SSIM (x,y) = {(2 \mu _x \mu _y + c_1)(2 \sigma_{xy} + c_2)} \over {(\mu _x^2 + \mu _y^2 + c_1）)(\sigma _x^2 + \sigma _y^2 +c_2)}
$$

每次计算的时候都从图片上取一个 $N×N$ 的窗口，然后不断滑动窗口进行计算，最后取平均值作为全局的 SSIM。

```python3
# im1 和 im2 都为灰度图像，uint8 类型
ssim = skimage.measure.compare_ssim(im1, im2, data_range=255)
```

备注：``compare_ssim(X, Y, win_size=None, gradient=False, data_range=None, multichannel=False, gaussian_weights=False, full=False, **kwargs)`` 函数原型可见, [此处](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim)。

针对超光谱图像，我们需要针对不同波段分别计算 SSIM，然后取平均值，这个指标称为 ``MSSIM``。

<br>

----

表1 COCO 数据集主要评价指标

|指标|含义|
|--|--|
|$AP_{bb}$|$IOU=0.5：0.05：0.95$ 时，AP的取值|
|$AP_{50}$|$IOU=0.5$ 时，AP的取值|
|$AP_{75}$|$IOU=0.75$ 时，AP的取值|
|$AP_{S}$|物体面积小于322时，AP的取值|
|$AP_{M}$|物体面积介于322~962时，AP的取值|
|$AP_{L}$|物体面积大于962时，AP的取值|

