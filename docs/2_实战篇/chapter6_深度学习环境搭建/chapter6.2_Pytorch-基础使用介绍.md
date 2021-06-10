<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

# 第 6 章 深度学习环境搭建

作者: 张伟 (Charmve)

日期: 2021/06/06

- 第 6 章 [深度学习环境搭建](https://charmve.github.io/computer-vision-in-action/#/chapter6/chapter6)
    - 6.1 [深度学习环境搭建指南](https://blog.csdn.net/Charmve/article/details/107739506)
    - 6.2 Pytorch 基础使用介绍
      - 6.2.1 [Tensors](#621-tensors)
      - 6.2.2 [Operations](#622-operations)
      - 6.2.3 [Numpy桥梁](#623-numpy桥梁)
      - 6.2.4 [CUDA Tensors](#624-cuda-tensors)
    - 6.3 [Python](../../../notebooks/02_Python.ipynb)
    - 6.4 [Numpy 基础使用](../../../notebooks/03_NumPy.ipynb)
    - 6.5 [Pandas 基础使用](../../../notebooks/04_Pandas.ipynb)
    - 6.4 [OpenCV 安装及基础使用](../../../notebooks/02_Python.ipynb)
    - 6.7 [Jupyter Notebook 配置及基础使用](../../../notebooks/01_Notebooks.ipynb)
    - 小结
    - 参考文献

## 6.2 Pytorch 基础使用介绍

<p align="center">
    <a href="https://colab.research.google.com/github/Charmve/computer-vision-in-action/blob/main/notebooks/05_PyTorch.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" align="center" alt="Open in Colab">
    </a>
</p>

Pytorch是什么？

Pytorch是一个基于python的科学计算包，主要面向两部分受众：

- 一个为了充分发挥GPU能力而设计的Numpy的GPU版本替代方案

- 一个提供更大灵活性和速度的深度学习研究平台

本节将会介绍Pytorch的一些基本使用和操作，帮助你熟悉和上手Pytorch，让我们开始吧～

### 6.2.1 Tensors

Tensors(张量)是PyTorch中一个非常重要的概念，可以类比Numpy中的``ndarrays``，本质上就是一个多维数组，是任何运算和操作间数据流动的最基础形式。

首先让我们加载torch库

```python
import torch
```

首先，让我们来看下如何生成一些简单的Tensor。

构建一个未初始化的5\*3的空矩阵（张量）的代码如下:

```python
x = torch.empty(5, 3)
print(type(x))
print(x)
```

输出：

```
<class 'torch.Tensor'>
tensor([[7.7050e+31, 6.7415e+22, 1.2690e+31],
        [6.1186e-04, 4.6165e+24, 4.3701e+12],
        [7.5338e+28, 7.1774e+22, 3.7386e-14],
        [6.6532e-33, 1.8337e+31, 1.3556e-19],
        [1.8370e+25, 2.0616e-19, 4.7429e+30]])
```

注意，对于未初始化的张量，它的取值是不固定的，取决于它创建时分配的那块内存的取值。

`torch.zeros` 和 `torch.ones`也是非常常用的Tensor初始化函数：

```python
tensor_ones = torch.ones(2,3)
print(tensor_ones)
tensor_zeros = torch.zeros(2,3)
print(tensor_zeros)
```

输出：

```
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

可以通过dtype属性来指定tensor的数据类型。

这里我们再次构建一个使用0填充的tensor，将dtype属性设置为长整型，并打印结果的数据类型，注意观察区别。

```python
print(tensor_zeros.dtype)
tensor_zeros_int = torch.zeros(2, 3, dtype=torch.long)
print(tensor_zeros_int.dtype)
```

输出：

```
torch.float32
torch.int64
```

和Numpy类似，除了常见的0/1取值的初始化，我们还可以进行随机初始化，或者直接用现有数据进行张量的初始化。

`torch.rand` 和 `torch.randn` 是两个常用的随机初始化函数。

`torch.rand`用于生成服从区间 [0,1) 匀分布的随机张量，示例：

```python
x = torch.rand(5, 3)
print(x)
```

输出：

```
tensor([[0.6544, 0.1733, 0.2569],
        [0.0680, 0.5781, 0.2667],
        [0.5051, 0.5366, 0.2776],
        [0.4903, 0.9934, 0.1181],
        [0.4497, 0.6201, 0.1952]])
```

`torch.randn`用于生成服从均值为0、方差为1正太分布的随机张量，示例：

```python
x = torch.randn(4, 4)
print(x)
```

输出：

```
tensor([[-1.2787, -1.8935, -0.1098, -0.5664],
        [ 1.2988,  0.5578, -1.7803,  0.9369],
        [ 0.7574, -0.4856, -1.5168, -0.5782],
        [ 0.9653, -1.0099,  0.4913, -0.1843]])
```

我们还可以直接用现有数据进行张量的初始化，例如python的列表``List``，例如：

```python
x = torch.tensor([5.5, 3])
print(x)
```

输出：

```
tensor([5.5000, 3.0000])
```

也可以基于已有的tensor来创建新的tensor，通常是为了**复用已有tensor的一些属性**，包括shape和dtype。观察下面的示例：

```python
x = torch.tensor([5.5, 3], dtype=torch.double)
print(x.dtype)
x = x.new_ones(5, 3)      # new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size
```

输出：

```
torch.float64
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[-0.4643, -0.1942, -0.4497],
        [ 0.8090,  1.6753, -0.0678],
        [-1.1654, -0.3817,  0.3495],
        [ 2.3517,  1.7688,  0.0251],
        [-0.6314, -1.2776,  0.8398]])
```

可以看到，`new_ones`函数复用了x的`dtype`属性，`randn_like`函数复用了x的`shape`同时通过手动指定数据类型覆盖了原有的`dtype`属性。

如果获取tensor的size？很简单。

```python
x_size = x.size()
print(x_size)
row, col = x_size
print(row, col)
```

输出：

```
torch.Size([5, 3])
5 3
```

`torch.Size`本质上是一个`tuple`，通过上面的例子也可以看出，它支持元组的操作。

最后介绍一个非常实用的函数`torch.arange`,用于生成一定范围内等间隔的一维数组。参数有三个，分别是范围的起始值、范围的结束值和步长，使用示例：

```python
torch.arange(1, 10, 1)
```

输出：

```
tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### 6.2.2 Operations

Operations(操作)涉及的语法和函数很多，大多数都是相通的，下面我们列举一些常用操作及其用法示例。

首先来看下张量间的元素级的四则运算，**加减乘除**的用法。

有多种不同的使用方法，以加法为例：

*加法：语法1*

```python
y = torch.rand(5, 3)
z1 = x + y
print(z1)
```

输出：

```
tensor([[ 1.7110, -0.0545,  0.3557],
        [ 0.1074,  2.3876,  1.2455],
        [-0.1204,  1.1829,  0.6250],
        [ 0.0873,  1.3114,  1.0403],
        [-0.6172,  1.2261, -1.0718]])
```

*加法：语法2*

```python
z2 = torch.add(x, y)
print(z2)
```

输出：

```
tensor([[ 1.7110, -0.0545,  0.3557],
        [ 0.1074,  2.3876,  1.2455],
        [-0.1204,  1.1829,  0.6250],
        [ 0.0873,  1.3114,  1.0403],
        [-0.6172,  1.2261, -1.0718]])
```

*加法：语法3*，通过参数的形式进行输出tensor的保存。

```python
z3 = torch.empty(5, 3)
torch.add(x, y, out=z3)
print(z3)
```

输出：

```
tensor([[ 1.7110, -0.0545,  0.3557],
        [ 0.1074,  2.3876,  1.2455],
        [-0.1204,  1.1829,  0.6250],
        [ 0.0873,  1.3114,  1.0403],
        [-0.6172,  1.2261, -1.0718]])
```

*加法：语法4*，通过in-place操作直接将计算结果覆盖到y上。

```python
y.add_(x)
print(y)  
```

输出：

```
tensor([[ 1.7110, -0.0545,  0.3557],
        [ 0.1074,  2.3876,  1.2455],
        [-0.1204,  1.1829,  0.6250],
        [ 0.0873,  1.3114,  1.0403],
        [-0.6172,  1.2261, -1.0718]])
```

在Pytorch中，我们约定凡是会覆盖函数调用主体的`in-place`操作，都以后缀`_`结束，例如：`x.copy_(y)`，`x.t_()`，都会改变`x`的取值。

张量之间的减法、点乘和点除的用法是类似的：

```python
a =torch.randn(2,3)
print(a)
b =torch.randn(2,3)
print(b)

c =torch.sub(a, b)
print(c)
d =torch.mul(a, b)
print(d)
e =torch.div(a, b)
print(e)
```

运行后，输出的内容如下：

```
tensor([[ 0.3580, -0.5780, -0.6883],
        [ 0.0883, -0.9064,  0.5411]])
tensor([[-1.7065,  0.7099,  0.6574],
        [-1.3688,  0.1767,  0.3669]])
tensor([[ 2.0645, -1.2880, -1.3457],
        [ 1.4571, -1.0831,  0.1742]])
tensor([[-0.6109, -0.4104, -0.4525],
        [-0.1209, -0.1602,  0.1986]])
tensor([[-0.2098, -0.8142, -1.0471],
        [-0.0645, -5.1292,  1.4748]])
```

当然，张量和常数间的基本运算也是支持的。

```python
a =torch.randn(2,3)
print(a)
b =torch.mul(a,10)
print(b)
c =torch.div(a,10)
print(c)
```

运行后，输出的内容如下：

```
tensor([[-1.6306,  0.2616,  0.0606],
        [-0.4596, -1.3998,  0.9431]])
tensor([[-16.3055,   2.6162,   0.6061],
        [ -4.5963, -13.9981,   9.4307]])
tensor([[-0.1631,  0.0262,  0.0061],
        [-0.0460, -0.1400,  0.0943]])
```

上面介绍了`torch.mul`用来计算张量间的点乘，而进行矩阵乘法计算需要用到`torch.mm`函数。

```python
a =torch.randn(2, 3)
print(a)
b =torch.randn(3, 2)
print(b)
c =torch.mm(a, b)
print(c)
```

输出的内容如下：

```
tensor([[-1.0442,  0.2439,  1.3658],
        [-1.0813,  0.3178,  0.4006]])
tensor([[ 0.4317,  0.3242],
        [-0.2085,  0.5584],
        [ 0.7991,  0.0926]])
tensor([[ 0.5898, -0.0759],
        [-0.2129, -0.1360]])
```


下面我们来看看其他的一些基础操作。

`torch.abs`函数可以用来计算张量的绝对值

```python
a =torch.randn(2, 3)
print(a)
b =torch.abs(a)
print(b)
```

输出：

```
tensor([[-0.8631,  0.3728,  1.0428],
        [ 0.5944,  0.5572,  0.6256]])
tensor([[0.8631, 0.3728, 1.0428],
        [0.5944, 0.5572, 0.6256]])
```

`torch.pow`函数用于进行求幂操作。

```python
import torch
a = torch.randn(2, 3)
print(a)
b = torch.pow(a, 2)
print(b)
```

输出：

```
tensor([[ 0.0115,  0.2041, -2.9827],
        [ 0.6467,  0.3175, -0.4201]])
tensor([[1.3127e-04, 4.1647e-02, 8.8963e+00],
        [4.1824e-01, 1.0082e-01, 1.7648e-01]])
```

在Pytorch中，我们可以使用标准的`Numpy-like`的索引操作，例如：

```python
print(x[:, 1])
```

输出：

```
tensor([-0.0211, -1.4157,  0.1453, -1.3401, -0.8556])
```

Resize操作：

如果你想要对tensor进行类似`resize/reshape`的操作，你可以使用`torch.view`。

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # 使用-1时pytorch将会自动根据其他维度进行推导
print(x.size(), y.size(), z.size())  
```

输出：

```
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

如果有一个tensor，只包含一个元素，你可以使用`.item()`来获得它对应的Python形式的取值。

```python
x = torch.randn(1)
print(x)
print(x.size())
print(x.item())
print(type(x.item())) 
```

输出：

```
tensor([0.4448])
torch.Size([1])
0.4447539746761322
<class 'float'>
```


通常我们在需要控制张量的取值范围不越界时，需要用到`torch.clamp`函数，它可以对输入参数按照自定义的范围进行裁剪，最后将参数裁剪的结果作为输出。输入参数一共有三个，分别是需要进行裁剪的Tensor变量、裁剪的下边界和裁剪的上边界。

```python
import torch
a =torch.randn(2,3)
print(a)
b =torch.clamp(a, -0.5, 0.5)
print(b)
```

输出的内容如下：

```
tensor([[-0.1076,  1.4202,  1.5780],
        [-1.3722, -1.7166, -1.0581]])
tensor([[-0.1076,  0.5000,  0.5000],
        [-0.5000, -0.5000, -0.5000]])
```


想要学习更多？PyTorch官方提供了更多关于tensor操作的介绍，点击[这里](https://pytorch.org/docs/stable/torch.html)了解更多，介绍了100多个Tensor运算，包括转置，索引，切片，数学运算，线性代数，随机数等。

### 6.2.3 Numpy桥梁

Pytorch中可以很方便的将Torch的Tensor同Numpy的ndarray进行互相转换，相当于在Numpy和Pytorch间建立了一座沟通的桥梁，这将会让我们的想法实现起来变得非常方便。

**注：Torch Tensor 和 Numpy ndarray 底层是分享内存空间的，也就是说改变其中之一会同时改变另一个（前提是你是在CPU上使用Torch Tensor）。**

将一个Torch Tensor 转换为 Numpy Array：

```python
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
```

输出：

```
tensor([1., 1., 1., 1., 1.])
[1. 1. 1. 1. 1.]
```

我们来验证下它们的取值是如何互相影响的。

```python
a.add_(1)
print(a)
print(b)
```

输出：

```
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```

将一个 Numpy Array 转换为 Torch Tensor：

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b) 
```

输出：

```
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

注：所有CPU上的Tensors，除了CharTensor均支持与Numpy的互相转换。

### 6.2.4 CUDA Tensors

`6.1 深度学习环境配置指南`一节，我们介绍了如何配置深度学习环境，以及如何安装GPU版本的pytorch，可以通过以下代码进行验证：

```python
print(torch.cuda.is_available())
```

如果输出`True`，代表你安装了GPU版本的pytorch

```
True
```

Tensors可以通过`.to`函数移动到任何我们定义的设备`device`上，观察如下代码：

```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!  
```

tensor([0.5906], device='cuda:0')
tensor([0.5906], dtype=torch.float64)


## 小结

1. 本节主要介绍了Tensors、Operations、利用Numpy桥梁将Torch的Tensor同Numpy的ndarray进行互相转换以及CUDA Tensors的基本使用，在学习的过程中借助命令行（Command Line）实际尝试，观察各种运算、操作的差别，本章内容都编写了对应的 Jupyter Notebook, 大家可直接在Colab上验证。
2. 本章节只是介绍了基本的PyTorch使用，满足对之后章节的入门条件，更多资料可参考PyTorch官方教程，以及之后的实战项目中更多的应用。

<p align="center">
    <a href="https://colab.research.google.com/github/Charmve/StegaStamp/blob/master/notebooks/05_PyTorch.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" align="center" alt="Open in Colab">
    </a>
</p>

## 参考资料

1. 毗湿奴·布拉马尼亚(Vishnu Subramanian)[印度].《PyTorch深度学习》. 人民邮电出版社出版[M]. ISBN: 9787115508980. 2020年1月

2.  PyTorch official website. INSTALL PYTORCH. http://pytorch.org

