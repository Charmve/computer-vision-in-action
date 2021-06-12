<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

# 第 12 章 生成对抗模型

- 第 12 章 [生成对抗模型](https://charmve.github.io/computer-vision-in-action/#/chapter6/chapter6)
    - 12.1 Pixel RNN/CNN
    - 12.2 自编码器 Auto-encoder
    - 12.3 生成对抗网络 GAN
      - 12.3.1 [概述](#1231-概述)
      - 12.3.2 [GAN的基本思想](#1232-gan的基本思想)
      - 12.3.3 [GAN浅析](#1233-gan浅析)
        - 12.3.3.1 [GAN的基本结构](#12331-gan的基本结构)
        - 12.3.3.2 [GAN的训练方式](#12332-gan的训练方式)
          - [关于生成器](#关于生成器)
          - [关于判别器](#关于判别器)
          - [如何训练](#如何训练)
      - 12.3.4 [训练相关理论基础](#1234训练相关理论基础)
      - 12.3.5 项目实战案例StyleGAN
        - [StyleGAN](https://github.com/Charmve/VOGUE-Try-On)
        - [StyleGAN 2.0](https://blog.csdn.net/Charmve/article/details/115315353)
      - [小结](#小结)
      - [参考文献](#参考文献)
    - [12.4 变分自编码器 Variational Auto-encoder, VAE](/chapter12_4-变分自编码器VAE.md#124-变分自编码器-variational-auto-encoder-vae)
      - [12.4.1 概述](/chapter12_4-变分自编码器VAE.md#1241-概述)    
      - [12.4.2 基本原理](/chapter12_4-变分自编码器VAE.md#1242-基本原理)                 
      - [12.4.3 VAE v.s. AE 区别与联系](/chapter12_4-变分自编码器VAE.md#1243-vae-vs-ae-区别与联系)    
      - [12.4.4 变分自编码器的代码实现](/chapter12_4-变分自编码器VAE.md#1244-变分自编码器的代码实现)    
      - [12.4.5 卷积变分自编码器的实现与简单应用](/chapter12_4-变分自编码器VAE.md#1245-卷积变分自编码器的实现与简单应用)             
      - 小结
      - [参考文献](/chapter12_4-变分自编码器VAE.md#参考文献) 
    - 小结
    - 参考文献

------

## 12.3 生成对抗网络 GAN

自2014年Ian Goodfellow提出了GAN（Generative Adversarial Network）以来，对GAN的研究可谓如火如荼。各种GAN的变体不断涌现，下图12.0是GAN相关论文的发表情况：

![image](https://user-images.githubusercontent.com/29084184/121766118-6411b080-cb82-11eb-8639-cee7dbe27162.png)

图12.0 GAN相关论文发表情况

大牛Yann LeCun甚至评价GAN为 “adversarial training is the coolest thing since sliced bread”。

那么到底什么是GAN呢？它又好在哪里？下面我们开始进行介绍。

![image](https://user-images.githubusercontent.com/29084184/121765675-370fce80-cb7f-11eb-8cdc-059573d30f40.png)

图12.1 生成对抗网络 GAN 论文

- 论文 https://arxiv.org/abs/1406.2661
- 源码 https://github.com/goodfeli/adversarial

### 12.3.1 概述
#### 12.3.1.1 什么是GAN？
生成对抗网络简称GAN，是由两个网络组成的，一个生成器网络和一个判别器网络。这两个网络可以是神经网络（从卷积神经网络、循环神经网络到自编码器）。我们之前学习过的机器学习或者神经网络模型主要能做两件事：预测和分类，这也是我们所熟知的。那么是否可以让机器模型自动来生成一张图片、一段语音？而且可以通过调整不同模型输入向量来获得特定的图片和声音。例如，可以调整输入参数，获得一张红头发、蓝眼睛的人脸，可以调整输入参数，得到女性的声音片段，等等。也就是说，这样的机器模型能够根据需求，自动生成我们想要的东西。因此，GAN 应运而生！

![image](https://user-images.githubusercontent.com/29084184/121765693-5c044180-cb7f-11eb-9993-5a2fafa1de53.png)

图12.2 生成对抗网络模型

#### 12.3.1.2 发展历史
生成对抗网络是由Ian Goodfellow等人于2014年在论文[《Generative Adversarial Networks》](https://arxiv.org/abs/1406.2661)中提出的，论文首页如图12.1所示。学术界公开接受了GAN，业界也欢迎GAN，GAN的崛起是不可避免的。

随着《Generative Adversarial Networks》提出后，GAN产生了广泛流行的架构，如DCGAN，StyleGAN，BigGAN，StackGAN，Pix2pix，Age-cGAN，CycleGAN，具体源码可参考 GitHub [@YadiraF实现](https://github.com/YadiraF/GAN)这些架构展示了非常有前途的结果。而随着GAN在理论与模型上的高速发展，它在计算机视觉、自然语言处理、人机交互等领域有着越来越深入的应用，并不断向着其它领域继续延伸。因此，本文将对GAN的理论与其应用做一个总结与介绍。

### 12.3.2 GAN的基本思想

GAN受博弈论中的零和博弈启发，将生成问题视作判别器和生成器这两个网络的对抗和博弈：生成器从给定噪声中（一般是指均匀分布或者正态分布）产生合成数据，判别器分辨生成器的的输出和真实数据。前者试图产生更接近真实的数据，相应地，后者试图更完美地分辨真实数据与生成数据。由此，两个网络在对抗中进步，在进步后继续对抗，由生成式网络得的数据也就越来越完美，逼近真实数据，从而可以生成想要得到的数据（图片、序列、视频等）。

如果将真实数据和生成数据服从两个分布，如图12.3所示：

![image](https://user-images.githubusercontent.com/29084184/121765858-7ee32580-cb80-11eb-8ce2-6d6017669334.png)

图12.3 判别分布与生成数据分布。蓝色虚线为判别分布D，黑色许虚线为真实的数据分布$P_{data}$ ,绿色实线为生成分布$P_{g}P$ 。

GAN从概率分布的角度来看，就是通过D来将生成分布推向真实分布，紧接着再优化D，直至到达图(d)所示，到达Nash均衡点，从而生成分布与真实分布重叠，生成极为接近真实分布的数据。

**#通俗解释：#**

GAN全称对抗生成网络，顾名思义是生成模型的一种，而他的训练则是处于一种对抗博弈状态中的。下面举例来解释一下GAN的基本思想。

![image](https://user-images.githubusercontent.com/29084184/121765883-b651d200-cb80-11eb-87af-792d5c225ad3.png)

图12.4 球员与教练员

> 假如你是一名篮球运动员，你想在下次比赛中得到上场机会。
> 于是在每一次训练赛之后你跟教练进行沟通：
> 
> 你：教练，我想打球
> 教练：（评估你的训练赛表现之后）... 算了吧
> （你通过跟其他人比较，发现自己的运球很差，于是你苦练了一段时间）
> 
> 你：教练，我想打球
> 教练：... 嗯 还不行
> （你发现大家投篮都很准，于是你苦练了一段时间的投篮）
> 
> 你：教练，我想打球
> 教练： ... 嗯 还有所欠缺
> （你发现你的身体不够壮，被人一碰就倒，于是你去泡健身房）
> 
> ......
> 
> 通过这样不断的努力和被拒绝，你最终在某一次训练赛之后得到教练的赞赏，获得了上场的机会。
> 值得一提的是在这个过程中，所有的候选球员都在不断地进步和提升。因而教练也要不断地通过对比场上球员和候补球员来学习分辨哪些球员是真正可以上场的，并且要“观察”得比球员更频繁。随着大家的成长教练也会会变得越来越严格。


现在大家对于GAN的思想应该有了感性的认识了，下面开始进一步窥探GAN的结构和思想。

### 12.3.3 GAN浅析
#### 12.3.3.1 GAN的基本结构
GAN的主要结构包括一个**生成器**G（Generator）和一个**判别器**D（Discriminator）。

在上面的例子中的球员就相当于生成器，我们需要他在球场上能有好的表现。而球员一开始都是初学者，这个时候就需要一个教练员来指导他们训练，告诉他们训练得怎么样，直到真的能够达到上场的标准。而这个教练就相当于判别器。

下面我们举另外一个手写字的例子来进行进一步窥探GAN的结构。

![image](https://user-images.githubusercontent.com/29084184/121765983-5dcf0480-cb81-11eb-9440-b2ad859f817e.png)

图12.5 GAN基本结构

我们现在拥有大量的手写数字的数据集，我们希望通过GAN生成一些能够以假乱真的手写字图片。主要由如下两个部分组成：

1. 定义一个模型来作为生成器（图三中蓝色部分Generator），能够输入一个向量，输出手写数字大小的像素图像。
2. 定义一个分类器来作为判别器（图三中红色部分Discriminator）用来判别图片是真的还是假的（或者说是来自数据集中的还是生成器中生成的），输入为手写图片，输出为判别图片的标签。


#### 12.3.3.2 GAN的训练方式

前面已经定义了一个生成器（Generator）来生成手写数字，一个判别器（Discrimnator）来判别手写数字是否是真实的，和一些真实的手写数字数据集。那么我们怎样来进行训练呢？


##### 关于生成器
对于生成器，输入需要一个n维度向量，输出为图片像素大小的图片。因而首先我们需要得到输入的向量。

> Tips: 这里的生成器可以是任意可以输出图片的模型，比如最简单的全连接神经网络，又或者是反卷积网络等。这里大家明白就好。
这里输入的向量我们将其视为携带输出的某些信息，比如说手写数字为数字几，手写的潦草程度等等。由于这里我们对于输出数字的具体信息不做要求，只要求其能够最大程度与真实手写数字相似（能骗过判别器）即可。所以我们使用随机生成的向量来作为输入即可，这里面的随机输入最好是满足常见分布比如均值分布，高斯分布等。

> Tips: 假如我们后面需要获得具体的输出数字等信息的时候，我们可以对输入向量产生的输出进行分析，获取到哪些维度是用于控制数字编号等信息的即可以得到具体的输出。而在训练之前往往不会去规定它。


##### 关于判别器
对于判别器不用多说，往往是常见的判别器，输入为图片，输出为图片的真伪标签。

> Tips: 同理，判别器与生成器一样，可以是任意的判别器模型，比如全连接网络，或者是包含卷积的网络等等。

##### 如何训练
上面进一步说明了生成器和判别器，接下来说明如何进行训练。

基本流程如下：

1. 初始化判别器D的参数 $\theta _d$ 和生成器G的参数 $\theta_g$ 。
2. 从真实样本中采样 $m$ 个样本 ${x^1, x^2,...,x^m}$ ，从先验分布噪声中采样 $m$ 个噪声样本 ${z^1, z^2,...,z^m}$ 并通过生成器获取 $m$ 个生成样本 ${\widetilde{x}^1,\widetilde{x}^2,...,\widetilde{x}^m}$ 。固定生成器G，训练判别器D尽可能好地准确判别真实样本和生成样本，尽可能大地区分正确样本和生成的样本。
3. **循环k次更新判别器之后，使用较小的学习率来更新一次生成器的参数**，训练生成器使其尽可能能够减小生成样本与真实样本之间的差距，也相当于尽量使得判别器判别错误。
4. 多次更新迭代之后，最终理想情况是使得判别器判别不出样本来自于生成器的输出还是真实的输出。亦即最终样本判别概率均为0.5。

> Tips: 之所以要训练k次判别器，再训练生成器，是因为要先拥有一个好的判别器，使得能够教好地区分出真实样本和生成样本之后，才好更为准确地对生成器进行更新。更直观的理解可以参考下图：

![image](https://user-images.githubusercontent.com/29084184/121766004-91aa2a00-cb81-11eb-9e3b-75f13f17a8d6.png)

图12.6 生成器判别器与样本示意图

注：图中的黑色虚线表示真实的样本的分布情况，蓝色虚线表示判别器判别概率的分布情况，绿色实线表示生成样本的分布。 $Z$ 表示噪声， $Z$ 到 $x$ 表示通过生成器之后的分布的映射情况。

我们的目标是使用生成样本分布（绿色实线）去拟合真实的样本分布（黑色虚线），来达到生成以假乱真样本的目的。

可以看到：
- 在（a）状态处于最初始的状态的时候，生成器生成的分布和真实分布区别较大，并且判别器判别出样本的概率不是很稳定，因此会先训练判别器来更好地分辨样本。
- 通过多次训练判别器来达到（b）样本状态，此时判别样本区分得非常显著和良好。然后再对生成器进行训练。
- 训练生成器之后达到（c）样本状态，此时生成器分布相比之前，逼近了真实样本分布。
- 经过多次反复训练迭代之后，最终希望能够达到（d）状态，生成样本分布拟合于真实样本分布，并且判别器分辨不出样本是生成的还是真实的（判别概率均为0.5）。也就是说我们这个时候就可以生成出非常真实的样本啦，目的达到。

### 12.3.4 训练相关理论基础

前面用了大白话来说明了训练的大致流程，下面会从交叉熵开始说起，一步步说明损失函数的相关理论，尤其是论文中包含min，max的公式如下图5形式：

$$\min \limits_{G} \max \limits_{D} V(D,G) = \mathbb{E}_{x \sim p_{\rm data(x))}}\Big[ \log D(x)\Big] + \mathbb{E}_{z \sim p_{\rm z(z))}} \Big[\log(1-D(G(z)))\Big]$$

图12.7 minmax公式

判别器在这里是一种分类器，用于区分样本的真伪，因此我们常常使用交叉熵（cross entropy）来进行判别分布的相似性，交叉熵公式如下：

$$ H(p,q) := -\sum_i p_i \log q_i$$

图12.8 交叉熵公式

> Tips: 公式中 $p_i$ 和 $q_i$ 为真实的样本分布和生成器的生成分布。由于交叉熵是非常常见的损失函数，这里默认大家都较为熟悉，就不进行赘述了。


在当前模型的情况下，判别器为一个二分类问题，因此可以对基本交叉熵进行更具体地展开如下：

$$H((x_1,y_1),D) = - y_1 \log D(x_1) - (1-y_1) \log (1-D(x_1))$$

图12.9 二分类交叉熵

> Tips: 其中，假定 $y_i$ 为正确样本分布，那么对应的 $(1-y_i)$ 就是生成样本的分布。 $D$ 表示判别器，则 $D(x_1)$ 表示判别样本为正确的概率， $(1-D(x_1))$ 则对应着判别为错误样本的概率。这里仅仅是对当前情况下的交叉熵损失的具体化。相信大家也还是比较熟悉。

将上式推广到N个样本后，将N个样本相加得到对应的公式如下：

$$H((x_i,y_i)_{i=1}^N,D) = - \sum_{i=1}^N y_i \log D(x_i) - \sum_{i=1}^N (1-y_i) \log (1-D(x_i))$$

图12.10 N个样本的情况时


OK，到目前为止还是基本的二分类，下面加入GAN中特殊的地方。

对于GAN中的样本点 $x_i$ ，对应于两个出处，要么来自于真实样本，要么来自于生成器生成的样本 $\widetilde{x}^1 ~ G(z)$ ( 这里的 $z$ 是服从于投到生成器中噪声的分布)。

其中，对于来自于真实的样本，我们要判别为正确的分布 $y_i$ 。来自于生成的样本我们要判别其为错误分布 $(1-y_i)$。将上面式子进一步使用概率分布的期望形式写出（为了表达无限的样本情况，相当于无限样本求和情况），并且让 $y_i$ 为 1/2 且使用 $G(z)$ 表示生成样本可以得到如下图8的公式：

![image](https://user-images.githubusercontent.com/29084184/121766068-01b8b000-cb82-11eb-8b4a-384b30418787.png)

$$H((x_i,y_i)_{i=1}^\infty,D) = - \frac{1}{2} \mathbb{E}_{x \sim p_{\rm data}}\Big[ \log D(x)\Big] - \frac{1}{2} \mathbb{E}_{z} \Big[\log (1-D(G(z)))\Big]$$

图12.11 GAN损失函数期望形式表达

OK，现在我们再回过头来对比原本的的 $\min \limits_{G} \max \limits_{D}$ 公式，发现他们是不是其实就是同一个东西呢！:-D

$$\min \limits_{G} \max \limits_{D} V(D,G) = \mathbb{E}_{x \sim p_{\rm data(x))}}\Big[ \log D(x)\Big] + \mathbb{E}_{z \sim p_{\rm z(z))}} \Big[\log(1-D(G(z)))\Big]$$

图12.12 损失函数的min max表达!


我们回忆一下上面12.2.2.3中介绍的流程理解一下这里的 $\min \limits_{G} \max \limits_{D}$ 。

- 这里的 $V(G,D)$ 相当于表示真实样本和生成样本的差异程度。
- 先看 $\max \limits_{D} V(D,G)$ 。这里的意思是固定生成器G，尽可能地让判别器能够最大化地判别出样本来自于真实数据还是生成的数据。
- 再将后面部分看成一个整体令 $L = \max \limits_{D} V(D,G)$ ，看 $\min \limits_{G}L$，这里是在固定判别器D的条件下得到生成器G，这个G要求能够最小化真实样本与生成样本的差异。
- 通过上述min max的博弈过程，理想情况下会收敛于生成分布拟合于真实分布。

### 12.3.5. 小结

本文大致介绍了GAN的整体情况。但是对于GAN实际上还有更多更完善的理论相关描述，进一步了解可以看相关的论文。并且在GAN一开始提出来的时候，实际上针对于不同的情况也有存在着一些不足，后面也陆续提出了不同的GAN的变体来完善GAN。

通过一个判别器而不是直接使用损失函数来进行逼近，更能够自顶向下地把握全局的信息。比如在图片中，虽然都是相差几像素点，但是这个像素点的位置如果在不同地方，那么他们之间的差别可能就非常之大。

![image](https://user-images.githubusercontent.com/29084184/121766088-2a40aa00-cb82-11eb-8352-69f681eb081e.png)

图12.13 不同像素位置的差别

比如上图10中的两组生成样本，对应的目标为字体2，但是图中上面的两个样本虽然只相差一个像素点，但是这个像素点对于全局的影响是比较大的，但是单纯地去使用使用损失函数来判断，那么他们的误差都是相差一个像素点，而下面的两个虽然相差了六个像素点的差距（粉色部分的像素点为误差），但是实际上对于整体的判断来说，是没有太大影响的。但是直接使用损失函数的话，却会得到6个像素点的差距，比上面的两幅图差别更大。而如果使用判别器，则可以更好地判别出这种情况(不会拘束于具体像素的差距)。

总之GAN是一个非常有意思的东西，现在也有很多相关的利用GAN的应用，比如利用GAN来生成人物头像，用GAN来进行文字的图片说明等等。后面我也会使用GAN来做一些简单的实验来帮助进一步理解GAN。

最后附上论文中的GAN算法流程，通过上面的介绍，这里应该非常好理解了。

![image](https://user-images.githubusercontent.com/29084184/121765697-64f51300-cb7f-11eb-8a8d-8e5fec45cac0.png)

图12.14 论文中的GAN算法流程


### 参考文献

[1] 陈诚. [通俗理解生成对抗网络GAN](https://zhuanlan.zhihu.com/p/33752313). 知乎. https://zhuanlan.zhihu.com/p/33752313

[2] 陈小虾. [生成对抗网络GAN详细推导](https://blog.csdn.net/ch18328071580/article/details/96690016). CSDN. https://blog.csdn.net/ch18328071580/article/details/96690016

[3] Goodfellow, Ian J., Pouget-Abadie, Jean, Mirza, Mehdi, Xu, Bing, Warde-Farley, David, Ozair, Sherjil, Courville, Aaron C., and Bengio, Yoshua. Generative adversarial nets. NIPS, 2014.

[4] [Understanding Generative Adversarial Networks](https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/)

[5]【李宏毅深度学习】Introduction of Generative Adversarial Network (GAN)（中文） https://www.bilibili.com/video/av17412504/?from=search&seid=12003526139493552118

[6] [Introductory guide to Generative Adversarial Networks (GANs) and their promise!](https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/)
