<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

# 第 12 章 生成对抗模型

作者: 张伟 (Charmve)

日期: 2021/05/19

- 第 12 章 [生成对抗模型](https://charmve.github.io/computer-vision-in-action/#/chapter6/chapter6)
    - 12.1 Pixel RNN/CNN
    - 12.2 自编码器 Auto-encoder
    - 12.3 生成对抗网络 GAN
      - 12.3.1 原理
      - 12.3.2 项目实战
        - [StyleGAN](https://github.com/Charmve/VOGUE-Try-On)
        - [StyleGAN 2.0](https://blog.csdn.net/Charmve/article/details/115315353)
    - [12.4 变分自编码器 Variational Auto-encoder, VAE](#124-变分自编码器-variational-auto-encoder-vae)
      - [12.4.1 概述](#1241-概述)    
      - [12.4.2 基本原理](#1242-基本原理)        
        - [12.4.2.1 定义](#1-定义)        
        - [12.4.2.2 理论基础：三要素](#2-理论基础三要素) 
        - [12.4.2.3 推导过程](#3-推导过程)            
      - [12.4.3 VAE v.s. AE 区别与联系](#1243-vae-vs-ae-区别与联系)    
      - [12.4.4 变分自编码器的代码实现](#1244-变分自编码器的代码实现)    
      - [12.4.5 卷积变分自编码器的实现与简单应用](#1245-卷积变分自编码器的实现与简单应用)             
      - [参考文献](#参考文献) 
    - 小结
    - 参考文献

------


# 12.4 变分自编码器 Variational Auto-encoder, VAE

- [12.4.1 概述](#1241-概述)    
- [12.4.2 基本原理](#1242-基本原理)        
	- [1. 定义](#1-定义)        
	- [2. 理论基础：三要素](#2-理论基础三要素) 
	- [3. 推导过程](#3-推导过程)           
		- [Step 1 - 引入隐变量 z](#step-1---引入隐变量-z)            
		- [Step 2 - Decoder 过程](#step-2---decoder-过程)            
		- [Step 3 - Encoder 过程](#step-3---encoder-过程)            
		- [Step 4 - ELBO](#step-4-elbo)            
		- [Step 5 - 消除 KL 项](#step-5-消除-kl-项)            
		- [Step 6 - 解决 ELBO](#step-6-解决-elbo)            
		- [推导结果](#推导结果)    
- [12.4.3 VAE v.s. AE 区别与联系](#1243-vae-vs-ae-区别与联系)    
- [12.4.4 变分自编码器的代码实现](#1244-变分自编码器的代码实现)    
- [12.4.5 卷积变分自编码器的实现与简单应用](#1245-卷积变分自编码器的实现与简单应用)            
	- [1. 安装依赖](#1-安装依赖)            
	- [2. 导入相关库与加载数据集](#2-导入相关库与加载数据集)            
	- [3. VAE 模型](#3-vae-模型)                
		- [(1) 重参数化技巧 (Reparameterization Trick)](#1-重参数化技巧-reparameterization-trick)            
		- [(2) 网络结构 (Network architecture)](#2-网络结构-network-architecture)            
	- [4. 定义损失函数和优化器](#4-定义损失函数和优化器)            
	- [5. 训练模型与生成图片](#5-训练模型与生成图片)            
	- [6. 生成过渡图像](#6-生成过渡图像)    
- [参考文献](#参考文献)
- 附录： 图中英文翻译

<br>

## 12.4.1 概述

变分自编码器（Variational auto-encoder，VAE）是一类重要的生成模型（generative model），在深度学习中占有重要地位，它最开始的目的是用于降维或特征学习。它于2013年由 Diederik P.Kingma 和Max Welling [1] 提出。2016年 Carl Doersch 写了一篇 VAEs 的 tutorial [2]，对 VAEs 做了更详细的介绍，比文献[1]更易懂。VAE 模型与 GAN 相比，VAE 有更加完备的数学理论（引入了隐变量），理论推导更加显性，训练相对来说更加容易。

VAE 可以从**神经网络**的角度或者**概率图模型**的角度来解释，本文主要从概率图模型的角度尽量通俗地讲解其原理，并给出代码实现。

--> [Back to Menu](#124-变分自编码器-variational-auto-encoder-vae)

## 12.4.2 基本原理

### 1. 定义

VAE 全名叫 **变分自编码器**，是从之前的 auto-encoder 演变过来的，auto-encoder 也就是自编码器，自编码器，顾名思义，就是可以自己对自己进行编码，重构。所以 AE 模型一般都由两部分的网络构成，一部分称为 encoder, 从一个高维的输入映射到一个低维的隐变量上，另外一部分称为 decoder, 从低维的隐变量再映射回高维的输入，如图12.1所示。

<div align="center">
	<p align="center">
	<img src="https://pic2.zhimg.com/80/v2-c2acba45269364fcfd460d37848f441d_720w.jpg" alt="图12.1 VAE模型">
	<br>(a) VAE模型
	</p>
	<p align="center">
	<img src="https://img-blog.csdnimg.cn/20181103081537493.JPG">
	<br>(b) 编码与重构
	</p>
	图12.1 VAE模型
</div><br>

如上图所示，我们能观测到的数据是 $x$ ，而 $x$ 由隐变量 $z$ 产生，由 $z->x$ 是生成模型 $p_{\phi}(x|z)$ ，从自编码器（auto-encoder）的角度来看，就是 **解码器 decoder**；而由 $x->z$ 是识别模型（recognition model）$ q_{\theta}(z|x)$ ，类似于自编码器的 **编码器 encoder**。

简单而言，encoder 网络中的参数为 $\theta$, decoder 中网络中的参数为$\phi$， $\theta$ 就是让网络从 $x$ 到 $z$ 的映射，而 $\phi$ 可以让网络完成从 $z$ 到 $x$ 的重构。可以构造如下的**损失函数**：

$$
l_i(θ,ϕ)  = − E_{z ∼ q_{\theta}(z|x_i)} [log(p_{\phi}(x_i|z))] + KL(q_{\theta}(z|x_i) || p(z))     
$$
<p align="right"> 式（1）</p>

上面的第一部分，可以看做是重建 loss，就是从 $x ∼ z ∼ x$ 的这样一个过程，可以表示成上面的熵的形式，也可以表示成最小二乘的形式，这个取决于 $x$ 本身的分布。后面的 KL 可以看做是正则项，$q_{\theta}(z|x)$ 可以看成是根据 $x$ 推导出来的 $z$ 的一个后验分布，$p(z)$ 可以看成是 $z$ 的一个先验分布，我们希望这两个的分布尽可能的拟合，所以这一点是 VAE 与 GAN 的最大不同之处，VAE对隐变量 $z$ 是有一个假设的，而 GAN 里面并没有这种假设。 一般来说，$p(z)$ 都假设是均值为0，方差为1的高斯分布 $\mathcal{N}(0, 1)$。

如果没有 KL 项，那 VAE 就退化成一个普通的 AE 模型，无法做生成，VAE 中的隐变量是一个分布，或者说近似高斯的分布，通过对这个概率分布采样，然后再通过 decoder 网络，VAE 可以生成不同的数据，这样VAE模型也可以被称为生成模型。


> **<你可能会问>** *为什么叫变分自编码器？*
> <br> 推荐了解一下 [变分推断](https://www.zhihu.com/question/41765860)，这里摘取其中某位大牛的回答：“简单易懂的理解变分其实就是一句话：用简单的分布q去近似复杂的分布p。”
> 所以暂时如果不考虑其他内容，联系一下整个 VAE 结构，应该就能懂变分过程具体是指什么了。 VAE 中的隐变量 $z$ 的生成过程就是一个变分过程，我们希望用简单的 $z$ 来映射复杂的分布，这既是一个降维的过程，同时也是一个变分推断的过程。

<br> --> [Back to Menu](#124-变分自编码器-variational-auto-encoder-vae)

### 2. 理论基础：三要素

理解变分自编码器的基本原理只需要关注整个模型的三个关键元素：

- **编码网络（Encoder Network）**，也称 推断网络 。该 NN 用来生成隐变量的参数（隐变量由多个高斯分布组成）。对于隐变量 $z$ ，首先初始化时可以是标准高斯分布，然后通过这个 NN，通过不断计算后验概率 $q(z|x)$ 来逐步确定高斯分布的参数（均值和方差）。
- **隐变量(Latent Variable)**。作为 Encoder 过程的产物，隐变量至少能够包含一些输入数据的信息（降维的作用），同时也应该具有生成类似数据的潜力。
- **解码网络(Decoder Network)**，也称 生成网络。该 NN 用于根据隐变量生成数据，我们希望它既有能力还原 encoder 的数据，同时还能根据数据特征生成一些输入样本中不包含的数据。

<br> --> [Back to Menu](#124-变分自编码器-variational-auto-encoder-vae)

### 3. 推导过程
> 该部分推导主要参考李宏毅老师的 [课件视频](https://www.bilibili.com/video/BV1JE411g7XF?p=61)

#### Step 1 - 引入隐变量 z
隐变量(Latent Variable)。上面已经讲过隐变量的基本概念，这里介绍隐变量在 VAE模型中的作用及特点。
- 隐变量 $z$ 是可以认为是隐藏层数据，它是不限定数目的符合 高斯分布 特征的数据。（根据实际情况确定数目）
- $z$ 由输入数据 X XX 的采样以及参数生成，它既包含 $X$ 的信息（这个于 AutoEncoder 的隐藏层类似），同时也满足 高斯分布，方便接下来进行梯度下降或者其他优化技术 [1]。
- 隐变量的作用除了让生成网络尽可能还原原来的数据 $X$ ，同时也能生成原来数据中不存在的数据。

首先当我们引入隐变量 $z$ 以后，可以用 $z$ 来表示 $P(x)$ ：

$$ P(X) = \int_z{P(X|z)P(z)} \,{\rm d}z (1)$$

其中，用 $P(X|z)$ 替代了 $f(z)$ ，这样可以用概率公式明确表示 X 对 Z 的依赖性；$P(z)$ 即高斯分布；其中 $X|z $ ~ $\mu N((z),\sigma(z))$，其中的均值 $\mu (z)$ 和 方差 $\sigma(z)$ 需要通过运算获得。

<p align="center">
	<img src="https://img-blog.csdnimg.cn/20200923170440697.png#pic_center" alt="图12.2 VAE模型的一个图模型">
	<br>图12.2 VAE模型的一个图模型
</p>

如图12.2所示，标准的VAE模型是一个图模型，注意明显缺乏任何结构，甚至没有 **“编码器”** 路径：可以在不输入的情况下从模型中采样。这里，矩形是 **“板符号”**，这意味着我们可以在模型参数 $\theta$ 保持不变的情况下，从 $z$ 和 $X$ 中采样N次。

#### Step 2 - Decoder 过程
上面也曾讲过，VAE 模型中同样有 Encoder 与 Decoder 过程，也就是说模型对于相同的输入时，也应该有尽可能相同的输出。所以这里再次遇到 Maximum likelihood（极大似然）。

在公式 (1) 中，将 $P(X)$ 用 $Z$ 来表示，所以对任何输入数据，应该尽量保证最后由隐变量而转换回输出数据与输入数据尽可能相等。

$$L = \sum_x{log P(X)}  $$
<p align="right"> 式（2）</p>

为了让公式 2 的输出极大似然 $X$ ，神经网络登场，它的作用就是调参，来达到极大似然的目的。（注意这里虽然介绍在前，但其实在训练的时候与后面的 NN 是同时开始训练）

本步介绍的是如何实现与 AE 类似的功能，即保证输出数据极大似然与输入数据。

这里用到的网络称为 VAE 中的 生成网络，即根据隐变量 $Z$ 生成输出数据的网络。

#### Step 3 - Encoder 过程
这个步骤需要确定隐变量 $Z$  ，也就是 Encoder 过程。

这里需要用到另外一个分布，$q(z|x)$ ，其中 $q(z|x)$ ~ $N(\mu'(x), \sigma'(x)$ ，注意这里的均值和方差都是对 $X$ 进行的。

同样地，这个任务也交给 NN 去完成。

这里的网络称为 **推断网络**。

<p align="center">
	<img src="https://img-blog.csdnimg.cn/20200923211822920.png#pic_center" alt="图12.3 最大似然估计">
	<br>图12.3 最大似然估计
</p>

#### Step 4 - ELBO
这里对 $log\ P(X)$ 进行变形转换。
$$ \log P(X) = \int_z{q(z|x)\log P(x)} \,{\rm d}z $$

<p align="right"> 式（3）</p>

在公式3中， $q(z|x)$ 可以是任意分布。

为了让 Encode 过程也参与进来，这里引入 $q(z|x)$，推导步骤如下（具体过程也可参考如下图12.4所示过程）：

$$
\log P(X) = \int_z{q(z|x)\log P(x)} \,{\rm d}z \\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = \int_z{q(z|x)\log ({P(z,x)\over P(z|x)})} \,{\rm d}z \\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = \int_z{q(z|x)\log ({P(z,x)\over q(z|x)}{q(z|x)\over P(z|x)})} \,{\rm d}z \\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = \int_z{q(z|x)\log ({P(z,x)\over q(z|x)})} \,{\rm d}z + \int_z{q(z|x)\log ({q(z|x)\over P(z|x)})} \,{\rm d}z 
$$

<p align="right"> 式（4）</p>

然后，因为公式 $ \int_z{q(z|x)\log ({q(z|x)\over P(z|x)})} \,{\rm d}z 
$ 即计算 KL 散度 $KL(q(z|x)||P(z|x))$ ，所以这个式子运算值一定大于等于0。

所以公式4 一定大于等于 $\int_z{q(z|x)\log ({P(z,x)\over q(z|x)})} \,{\rm d}z$，取下界 lower bound。

为了方便，我们把这个公式记作 $L_b$b ，(ELBO)
$$L_b = \int_z{q(z|x)\log ({P(z,x)\over q(z|x)})} \,{\rm d}z$$
<p align="right"> 式（5）</p>

<p align="center">
	<img src="https://img-blog.csdnimg.cn/20200923214724344.png#pic_center" alt="图12.4 最大似然估计推导">
	<br>图12.4 最大似然估计推导
</p>

#### Step 5 - 消除 KL 项
根据公式 4 和公式 5，可以把 $\log P(X)$ 简单记为
$$\log P(X) = L_b + KL(q(z|x)\ ||\ P(z|x)) $$
<p align="right"> 式（6）</p>

这里首先需要提出一个重要结论：
<center><font color = red> log P(X) 值的大小与 q(z|x) 无关。</font></center><br>

所以不管怎么只调 $q(z|x)$，都不会让 $\log P(X)$ 增大。

所以可以通过调 $q(z|x)$ 让 $ KL(q(z|x)\ ||\ P(z|x))$ 尽可能小（调整为0），再通过调 $ELBO$ 来实现最大化 $P(X)$。

调整的最终结果是使得 $q(z|x)$ 尽可能接近 $p(z|x)$，换句话说，最终的 $ KL(q(z|x)\ ||\ P(z|x)) \approx 0$ 。所以这个步骤我们消除了这个 KL 项，剩下 $ELBO$ 等待解决，如图12.5所示。

<p align="center">
	<img src="https://img-blog.csdnimg.cn/20200924153534286.png#pic_center" alt="图12.5 最大似然概率估计">
	<br>图12.5 最大似然概率估计
</p>

#### Step 6 - 解决 ELBO
上面已经明确了目标：找到 $P(x|z)$ 和 $q(z|x)$ 使得 $L_b$ 尽可能大。

公式推导如下：
$$ L_b = \int_z{q(z|x)\log ({P(z,x)\over q(z|x)})} \,{\rm d}z \\ \ \ \ \ \ \ \ \ \ \ \ \ = \int_z{q(z|x)\log ({P(x|z)P(z)\over q(z|x)})} \,{\rm d}z \\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = \int_z{q(z|x)\log ({P(z)\over q(z|x)})} \,{\rm d}z + \int_z{q(z|x)\log ({P(x|z)})} \,{\rm d}z \\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = -KL(q(z|x)\ || \ P(z)) + \int_z{q(z|x)\log ({P(x|z)})} \,{\rm d}z $$
<p align="right"> 式（7）</p>

公式中第2步到第3步的过程用到的是对数函数的性质，即
$$\log AB = \log A + \log B \\$$

接下来需要 Minimize $ KL(q(z|x)\ || \ P(z))$，也就是调整$q(z|x)$ 来最小化，这里交给 NN 吧。

并且需要最大化另外一项，即$\int_z{q(z|x)\log ({P(x|z)})} \,{\rm d}z$，同样这份苦差事也交给 NN 去完成，如图12.6和12.7所示。

<p align="center">
	<img src="https://img-blog.csdnimg.cn/20200924180813767.png#pic_center" alt="图12.6 最大似然估计推导">
	<br>图12.6 最大似然估计推导
	<img src="https://img-blog.csdnimg.cn/20200924155823608.png?#pic_center" alt="图12.7 与NN连接">
	<br>图12.7 与NN连接
</p>

#### 推导结果
综合上面的推导，简单概括一下可以得到：

$$ \log P(X) = L_b + KL(q(z|x)\ ||\ P(z|x))$$
       

最终经过NN调参后，$KL(q(z|x)\ ||\ P(z|x))=0$ 在这里即 ELBO.
$$ELBO = \int_z{q(z|x)\log ({P(x|z)P(z)\over q(z|x)})} \,{\rm d}z $$
<p align="right"> 式（8）</p>

接着对公式8进一步展开，可以得到

$$ELBO = -KL(q(z|x)\ || \ P(z)) + \int_z{q(z|x)\log ({P(x|z)})} \,{\rm d}z \\ \ \ \ \ \ \ \ \ \ = -KL(q(z|x)\ || \ P(z)) + E_{q(z|x)}\log ({P(x|z)}) $$
<p align="right"> 式（9）</p>

公式 9 中的 $E$ 是指均值，把一个求积分转换为对 $\log({P(x|z)})$ 在 $q(z|x)$ 条件范围内的均值。

在对应的代码实现中，有两种实现方式，一种是用公式 9 ，一种直接用公式 8。如果直接用公式 8 的话，可以理解为：$\log({P(x|z)})$ 在 $q(z|x)$ 条件范围内$ \log ({P(x|z)P(z)\over q(z|x)})$的均值。

<br> --> [Back to Menu](#124-变分自编码器-variational-auto-encoder-vae)

## 12.4.3 VAE v.s. AE 区别与联系

（1）区别
- VAE 中隐藏层服从高斯分布，AE 中的隐藏层无分布要求
- 训练时，AE 训练得到 Encoder 和 Decoder 模型，而 VAE 除了得到这两个模型，还获得了隐藏层的分布模型（即高斯分布的均值与方差）
- AE 只能重构输入数据X，而 VAE 可以生成含有输入数据某些特征与参数的新数据。

（2）联系

- VAE 与 AE 完全不同，但是从结构上看都含有 Decoder 和 Encoder 过程。

<br> --> [Back to Menu](#124-变分自编码器-variational-auto-encoder-vae)

## 12.4.4 变分自编码器的代码实现
如果只是基于 MLP 的VAE，就是普通的全连接网络：
```python
import tensorflow as tf
from tensorflow.contrib import layers

## encoder 模块
def fc_encoder(x, latent_dim, activation=None):
    e = layers.fully_connected(x, 500, scope='fc-01')
    e = layers.fully_connected(e, 200, scope='fc-02')
    output = layers.fully_connected(e, 2 * latent_dim, activation_fn=activation,
                                    scope='fc-final')

    return output

## decoder 模块
def fc_decoder(z, observation_dim, activation=tf.sigmoid):
    x = layers.fully_connected(z, 200, scope='fc-01')
    x = layers.fully_connected(x, 500, scope='fc-02')
    output = layers.fully_connected(x, observation_dim, activation_fn=activation,
                                    scope='fc-final')
    return output
```

关于这几个 loss 的计算：
```python
   ## KL loss
    def _kl_diagnormal_stdnormal(mu, log_var):
        var = tf.exp(log_var)
        kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - log_var)
        return kl
   
   ## 基于高斯分布的重建loss
    def gaussian_log_likelihood(targets, mean, std):
        se = 0.5 * tf.reduce_sum(tf.square(targets - mean)) / (2*tf.square(std)) + tf.log(std)
        return se
        
   ## 基于伯努利分布的重建loss
    def bernoulli_log_likelihood(targets, outputs, eps=1e-8):
        log_like = -tf.reduce_sum(targets * tf.log(outputs + eps)
                                  + (1. - targets) * tf.log((1. - outputs) + eps))
        return log_like
```
可以看到，重建loss，如果是高斯分布，就是最小二乘，如果是伯努利分布，就是交叉熵，关于高斯分布的 KL loss 的详细推导，可以参考 [KL散度的实战——1维高斯分布](https://zhuanlan.zhihu.com/p/22464760)。

过程很复杂，结果很简单。如果有两个高斯分布 $\mathcal{N}_1 \sim(\mu_{1}, \sigma_{1}^{2})$，$\mathcal{N}_2 \sim(\mu_{2}, \sigma_{2}^{2})$，最后这两个分布的 KL 散度是：

$$ KL(N_1 || N_2) = log(\frac{\sigma_{2}}{\sigma_{1}}) + \frac{\sigma_{1}^{2} + (\mu_{1} - \mu_{2})^{2}}{2\sigma_{2}^{2}} - \frac{1}{2}$$

<p align="right"> 式（10）</p>

VAE 中，我们已经假设 $z$ 的先验分布是 $\mathcal{N}(0, 1)$，所以 $\mu_{2} = 0, \sigma_{2}^{2} =1$，代入上面的公式，可以得到：
 
$$loss_{KL}= -log({\sigma_{1}}) + \frac{\sigma_{1}^{2} + \mu_{1}^{2}}{2} - \frac{1}{2}$$
<p align="right"> 式（11）</p>

<br> --> [Back to Menu](#124-变分自编码器-variational-auto-encoder-vae)

## 12.4.5 卷积变分自编码器的实现与简单应用

本次实验采用的是 notebook，可以是自己电脑上安装的 jupyter notebook，也可以使用自己云服务器安装的，也可以考虑使用谷歌提供的 [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)。

文件地址：[``/notebooks/16_CVAE.ipynb``](../../../notebooks/16_CVAE.ipynb)

#### 1. 安装依赖
1. 确定使用的是 tensorflow 2.x
```
!pip show tensorflow
```
如果当前安装的不是 ``tensorflow 2.x`` 的话，请输入以下命令安装：
```
!pip install tensorflow==2.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
2. 安装 imageio

<code>!pip install imageio</code>


#### 2. 导入相关库与加载数据集
```python

import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

from IPython import display

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# 标准化图片到区间 [0., 1.] 内
train_images /= 255.
test_images /= 255.

# 二值化
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

# 使用 tf.data 来将数据分批和打乱
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)
```

#### 3. VAE 模型
准备工作做完了后，这里正式开始编写实现VAE模型。

##### (1) 重参数化技巧 (Reparameterization Trick)

训练过程中，为了生成样本 $z$ 以便于 decoder 操作，我们可以从 encoder 生成的分布中进行采样。但是，由于反向传播无法通过随机节点，因此此采样操作会产生瓶颈。

为了解决这个问题，我们使用了一个重新参数化的技巧。我们使用 decoder 参数和另一个参数 $\varepsilon$ 近似z，如下所示：

$$ z = \mu +\sigma \ \odot \ \varepsilon $$
<p align="right"> 式（12）</p>

其中 $\mu$ 和 $\sigma$ 表示高斯分布中的 **均值** 和 **标准差**。它们可以从 decoder 输出中导出。可以认为 $\varepsilon$ 是用来保持 $z$ 随机性的随机噪声。我们从标准正态分布生成。

现在的 $z$ 是 $q(z|x)$ 生成（通过参数 $\mu$，$\sigma$ 和 $\varepsilon$），这将使模型分别通过 $\mu$, $\sigma$ 在 encoder 中 反向传播梯度，同时通过 $\varepsilon$ 保持 $z$ 的随机性。


##### (2) 网络结构 (Network architecture)

对于 VAE 模型构建，

- 在 Encoder NN中，使用两个卷积层和一个完全连接的层。、
- 在 Decoder NN中，通过使用一个完全连接的层和三个卷积转置层来镜像这种结构。
注意，在训练VAE时，通常避免使用批次标准化，因为使用小批量的额外随机性可能会加剧抽样随机性之外的不稳定性。


```python
class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits
```


#### 4. 定义损失函数和优化器

如上所述，VAE 通过 $\log p(x)$ 极大似然 ELBO ( the evidence lower bound) 进行训练：
$$\log p(x) \ge \text{ELBO} = \mathbb{E}_{q(z|x)}\left[\log \frac{p(x, z)}{q(z|x)}\right].$$

<p align="right"> 式（13）</p>

实际操作中，我们优化了这种单样本蒙特卡罗估计：
$$\log p(x| z) + \log p(z) - \log q(z|x),$$

<p align="right"> 式（14）</p>

其中 $z$ 从 $q(z|x)$ 中采样。


```python
optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

#### 5. 训练模型与生成图片
(1) **训练**

- 我们从迭代数据集开始。
- 在每次迭代期间，我们将图像传递给编码器，以获得近似后验 $q(z|x)$ 的一组均值和对数方差参数(log-variance parameters）。
- 然后，我们应用 重参数化技巧 从 $q(z|x)$ 中采样。
- 最后，我们将重新参数化的样本传递给解码器，以获取生成分布 $p(x|z)$ 的 logit。

> **注意：** 由于我们使用的是由 keras 加载的数据集，其中训练集中有 6 万个数据点，测试集中有 1 万个数据点，因此我们在测试集上的最终 ELBO 略高于对 Larochelle 版 MNIST 使用动态二值化的文献中的报告结果。这里有个 [关于 logits 的 解释](https://www.zhihu.com/question/60751553)。

(2) **生成图片**

- 进行训练后，可以生成一些图片了。
- 我们首先从单位高斯先验分布 $p(z)$ 中采样一组潜在向量。
- 随后生成器将潜在样本 $z$ 转换为观测值的 logit，得到分布 $p(x|z)$。
- 这里我们画出伯努利分布的概率。

```python
epochs = 100
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]
generate_and_save_images(model, 0, test_sample)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(model, train_x, optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss(compute_loss(model, test_x))
  elbo = -loss.result()
  display.clear_output(wait=False)
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
  generate_and_save_images(model, epoch, test_sample)
```

一百次循环后，生成的图片如下图 12.8 所示。

<p align="center">
	<img src="https://img-blog.csdnimg.cn/2020101009492281.png#pic_center" alt="图12.8 生成图像">
	<br>图12.8 生成图像
</p>

同时可以生成 gif 图片来方便查看生成过程，如图12.9所示。

```python
anim_file = 'cvae.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
```

展示 gif 图片

```python
import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)
```

<p align="center">
	<img src="https://img-blog.csdnimg.cn/20201010101327220.gif#pic_center" alt="图12.9 展示 gif 图片">
	<br>图12.9 展示 gif 图片
</p>

#### 6. 生成过渡图像
最终生成的过度图像如下图 12.10 所示。
<p align="center">
	<img src="https://img-blog.csdnimg.cn/20201010102938870.png#pic_center" alt="图12.10 生成图像">
	<br>图12.10 生成图像
</p>


<br>

> **扩展阅读** - VAE均值与方差的故事
> <br>很久以前，有个叫VAE的生产车间，车间主任雇佣了 $\mu$ 与 $\sigma$ 。两个小伙，但并没有直接给他们安排工作任务，只告诉他们：好好干，如果你们做得不好，那就扣你们的工资。
> <br>刚开始。 $\mu$ 与 $\sigma$ 有些懵逼：连任务都不说清楚，就叫我们好好干？
> <br>第一天扣的工资最多，因为懵逼了一天。
> <br>第二天俩小伙对车间生产流程，生产目标熟悉了，扣得少了一些。
> <br>第三天他们更加谨慎了，毕竟谁都害怕扣工资，就这么点钱。
> <br>第四天、第五天．．．．．．后来 $\mu$ 和 $\sigma$ 都摸清楚了套路，懂得怎么增加收入。
> <br>终于有一天车间主任满意了，看着漂漂亮亮的产品，决定给 $\mu$ 和 $\sigma$ 评职称。众所周知，评上职称的工资一般都比较高。
> <br>最终被评为"车间最优均值"称号。
> <br>那 $\sigma$ 最终是不是就被评为"车间最美方差"呢？不是的，他被评为"车间最美标准差''。后来他们这个生产小组被评为''正态分布"，再后来车间主任用同样的手段忽悠更多的无业游民（随机数），通过很多次勤奋工作与调整（训练），组成一组又一组"正态分布"（隐变量）。
> <br>从此，VAE车间的萨与 $\sigma$ 勤勤恳恳干活，为 VAE 车间创造了很多产品，为大家所喜爱，而他们在车间勤奋努力不断调整自己的故事，也一时传为佳话。

<br> --> [Back to Menu](#124-变分自编码器-variational-auto-encoder-vae)

## 参考文献

[1] Kingma D P, Welling M. Auto-Encoding Variational Bayes[J]. stat, 2014, 1050: 10.

[2] DOERSCH C. Tutorial on Variational Autoencoders[J]. stat, 2016, 1050: 13.

[3] Blei, David M., "Variational Inference." Lecture from Princeton,
https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf.

[4] VAE 模型基本原理简单介绍 https://blog.csdn.net/smileyan9/article/details/107362252

[5] 机器学习：VAE(Variational Autoencoder) 模型 https://blog.csdn.net/matrix_space/article/details/83683811

[6] TensorFlow. Convolutional Variational Autoencoder. https://tensorflow.google.cn/tutorials/generative/cvae

<br>

## 附录： 图中英文翻译

| 图序号 | 英文原文 | 中文翻译 | 
| :--: | - | - |
| 图12.3 | Maximizing Likelihood | 最大似然估计 | 
| | |P(z) 是正态分布，μ(z), o(z) 是将被估计的参数|
| | Maximizing the likelihood of the observed x | 最大化x的似然估计 | 
| | Tunning the parameters to maximize likelihood L | 调整参数以使似然L最大化 |
| | We need another distribution q(z\|x) | 我们需要另外一个分布 q(z\|x) | 
| \ | \ | \ |
| 图12.4 | Maximizing Likelihood | 最大似然估计 |
| | q(z\|x) can be any distribution | q(z\|x)可以是任何一种分布 |
| | lower bound Lb | 取 Lb 的下界|
| \ | \ | \ |
| 图12.5 | Maximizing Likelihood | 最大似然估计 |
| | Find P(x\|z) and q(z\|x) maximizing Lb | 寻找 P(x\|z) 和 q(z\|x) 使得 Lb 最大 |
| | q(z\|x) will be an approximation of p(z\|x) in the end | q(z\|x) 最终将会被 p(z\|x) 近似化 |
| \ | \ | \ |
| 图12.7 | Connection with network | 连接网络 |
| | Refer to the Appendix B of the original VAE paper | 请参考附件B中的论文原文 |
| | this is the auto-encoder | 这就是自编码器 |

