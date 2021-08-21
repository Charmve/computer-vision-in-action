<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>


**第 1 章 神经网络**

作者: 张伟 (Charmve)

日期: 2021/04/29

- 第 1 章 [神经网络]()
    - 1.1 线性回归
      - 1.1.1 基本原理
      - 1.1.2 从零实现线性回归
      - 1.1.3 线性回归的简洁实现
    - 1.2 [Softmax 回归](./docs/1_理论篇/chapter3_Image-Classification/chapter1.2_Softmax回归.md)
      - 1.2.1 softmax回归模型
      - 1.2.2 从零开始实现softmax回归
      - 1.2.3 softmax回归的简洁实现
    - 1.3 [多层感知器](./docs/1_理论篇/chapter1_Neural-Networks/chapter1.3_多层感知器MLP.md)
      - 1.3.1 基本原理
      - 1.3.2 从零开始实现多层感知器
      - 1.3.3 多层感知器的简洁实现
    - 1.4 [反向传播算法](./docs/1_理论篇/chapter1_Neural-Networks/chapter1.4_Back-Propagation.md)
    - 1.5 [神经网络](./docs/1_理论篇/chapter1_Neural-Networks/chapter1.5_neural-networks.md)
      - 1.5.1 [神经学观点](./docs/1_理论篇/chapter1_Neural-Networks/chapter1.5.1_神经学观点.md)
      - 1.5.2 [神经网络1-建立神经网络架构](https://cs231n.github.io/neural-networks-1/)
      - 1.5.3 [神经网络2-设置数据和损失](https://cs231n.github.io/neural-networks-2/)
      - 1.5.4 [神经网络3-学习和评估](https://cs231n.github.io/neural-networks-3/)
      - 1.5.5 [案例分析-最小神经网络案例研究](https://cs231n.github.io/neural-networks-case-study/)
    - 1.6 [实战项目 1 - 手写字分类](https://blog.csdn.net/Charmve/article/details/108531735)
    - 小结
    - 参考文献

---

从本章开始，我们将以深度学习的基础理论开始探索计算机视觉的奥秘。作为机器学习的一类，深度学习通常基于神经网络模型逐级表示越来越抽象的概念或模式。我们先从线性回归和 softmax 回归这两种单层神经网络入手，简要介绍机器学习中的基本概念。然后，我们由单层神经网络延伸到多层神经网络，并通过多层感知机引入计算机视觉中使用的深度学习模型。接着，为了进一步理解深度学习模型训练的本质，我们将详细解释正向传播和反向传播。这是深度学习中一切的开始。然后，引入深度学习发展中最核心的部分——神经网络模型，介绍其组成以及如何亲手训练一个神经网络模型。最后，我们通过一个手写字应用案例对本章内容学以致用，领略计算机视觉的魅力。

在本章的前几节，我们先介绍单层神经网络：线性回归和softmax回归，这是第一节。

# 1 线性回归

<p align="center">
    <a href="https://colab.research.google.com/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter01_Neural-Networks/01_line-regression.ipynb" target="_blank\">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" align="center" alt="Open in Colab">
    </a> 
    <a href="https://nbviewer.jupyter.org/format/slides/github/Charmve/computer-vision-in-action/blob/main/notebooks/chapter01_Neural-Networks/01_line-regression.ipynb\" target="_blank\">
      <img src="https://mybinder.org/badge_logo.svg" align="center" alt="Open in Binder">
    </a>
</p>


<h2>1.1 基本原理 <a class="headerlink" href="#练习">¶</a></h2>

线性回归输出是一个连续值，因此适用于回归问题。回归问题在实际中很常见，如预测房屋价格、气温、销售额等连续值的问题。与回归问题不同，分类问题中模型的最终输出是一个离散值。我们所说的图像分类、垃圾邮件识别、疾病检测等输出为离散值的问题都属于分类问题的范畴。softmax回归则适用于分类问题。

由于线性回归和softmax回归都是单层神经网络，它们涉及的概念和技术同样适用于大多数的深度学习模型。我们首先以线性回归为例，介绍大多数深度学习模型的基本要素和表示方法。

<h2>1.1.1 线性回归的基本要素 <a class="headerlink" href="#练习">¶</a></h2>

我们以一个简单的房屋价格预测作为例子来解释线性回归的基本要素。这个应用的目标是预测一栋房子的售出价格（元）。我们知道这个价格取决于很多因素，如房屋状况、地段、市场行情等。为了简单起见，这里我们假设价格只取决于房屋状况的两个因素，即面积（平方米）和房龄（年）。接下来我们希望探索价格与这两个因素的具体关系。

<h3>1.1.1.1. 模型 <a class="headerlink" href="#练习">¶</a></h3>

设房屋的面积为 $x_1$ ，房龄为 $x_2$ ，售出价格为 $y$ 。我们需要建立基于输入 $x_1$ 和 $x_2$ 来计算输出 $y$ 的表达式，也就是模型（model）。顾名思义，线性回归假设输出与各个输入之间是线性关系：

$$
\[\hat{y} = x_1 w_1 + x_2 w_2 + b,\]
$$

其中 $w_1$ 和 $w_2$ 是权重（weight）， $b$ 是偏差（bias），且均为标量。它们是线性回归模型的参数（parameter）。模型输出 $\hat{y}$ 是线性回归对真实价格 $y$ 的预测或估计。我们通常允许它们之间有一定误差。

<h3>1.1.1.2. 模型训练 <a class="headerlink" href="#练习">¶</a></h3>

接下来我们需要通过数据来寻找特定的模型参数值，使模型在数据上的误差尽可能小。这个过程叫作模型训练（model
training）。下面我们介绍模型训练所涉及的3个要素。

<h4>训练数据 <a class="headerlink" href="#训练数据">¶</a></h4>

我们通常收集一系列的真实数据，例如多栋房屋的真实售出价格和它们对应的面积和房龄。我们希望在这个数据上面寻找模型参数来使模型的预测价格与真实价格的误差最小。在机器学习术语里，该数据集被称为训练数据集（training
data set）或训练集（training
set），一栋房屋被称为一个样本（sample），其真实售出价格叫作标签（label），用来预测标签的两个因素叫作特征（feature）。特征用来表征样本的特点。
假设我们采集的样本数为 $n$ ，索引为 $i$ 的样本的特征为 $x_1^{(i)}$和 $x_2^{(i)}$ ，标签为 $y^{(i)}$ 。对于索引为 $i$ 的房屋，线性回归模型的房屋价格预测表达式为

$$
[hat{y}^{(i)} = x_1^{(i)} w_1 + x_2^{(i)} w_2 + b.]
$$

<h4>损失函数 <a class="headerlink" href="#损失函数">¶</a></h4>
在模型训练中，我们需要衡量价格预测值与真实值之间的误差。通常我们会选取一个非负数作为误差，且数值越小表示误差越小。一个常用的选择是平方函数。它在评估索引为 $i$ 的样本误差的表达式为

$$
[ell^{(i)}(w_1, w_2, b) = frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2,\]
$$

其中常数 $1/2$ 使对平方项求导后的常数系数为1，这样在形式上稍微简单一些。显然，误差越小表示预测价格与真实价格越相近，且当二者相等时误差为0。给定训练数据集，这个误差只与模型参数相关，因此我们将它记为以模型参数为参数的函数。在机器学习里，将衡量误差的函数称为损失函数（loss
function）。这里使用的平方误差函数也称为平方损失（square loss）。

通常，我们用训练数据集中所有样本误差的平均来衡量模型预测的质量，即

$$
\[\ell(w_1, w_2, b) =\frac{1}{n} \sum_{i=1}^n \ell^{(i)}(w_1, w_2, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right)^2.\]
$$

在模型训练中，我们希望找出一组模型参数，记为 $w_1^*, w_2^*, b^*$ ，来使训练样本平均损失最小：

$$
[w_1^*, w_2^*, b^* = \operatorname*{argmin}_{w_1, w_2, b}\  \ell(w_1, w_2, b).\]
$$

<h4>优化算法<a class="headerlink" href="#优化算法">¶</a></h4>

当模型和损失函数形式较为简单时，上面的误差最小化问题的解可以直接用公式表达出来。这类解叫作解析解（analytical
solution）。本节使用的线性回归和平方误差刚好属于这个范畴。然而，大多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作数值解（numerical
solution）。

在求数值解的优化算法中，小批量随机梯度下降（mini-batch stochastic gradient descent）在深度学习中被广泛使用。它的算法很简单：先选取一组模型参数的初始值，如随机选取；接下来对参数进行多次迭代，使每次迭代都可能降低损失函数的值。在每次迭代中，先随机均匀采样一个由固定数目训练数据样本所组成的小批量（mini-batch） $\mathcal{B}$ ，然后求小批量中数据样本的平均损失有关模型参数的导数（梯度），最后用此结果与预先设定的一个正数的乘积作为模型参数在本次迭代的减小量。

在训练本节讨论的线性回归模型的过程中，模型的每个参数将作如下迭代：

$$
\[\begin{split}\begin{aligned}
w_1 &amp;\leftarrow w_1 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{ \partial \ell^{(i)}(w_1, w_2, b)  }{\partial w_1} = w_1 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}x_1^{(i)} \left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right),\\
w_2 &amp;\leftarrow w_2 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{ \partial \ell^{(i)}(w_1, w_2, b)  }{\partial w_2} = w_2 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}x_2^{(i)} \left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right),\\
b &amp;\leftarrow b -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{ \partial \ell^{(i)}(w_1, w_2, b)  }{\partial b} = b -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}\left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right).
\end{aligned}\end{split}\]
$$

在上式中， $|\mathcal{B}|$ 代表每个小批量中的样本个数（批量大小，batch size）， $\eta$ 称作学习率（learning rate）并取正数。需要强调的是，这里的批量大小和学习率的值是人为设定的，并不是通过模型训练学出的，因此叫作超参数（hyperparameter）。我们通常所说的“调参”指的正是调节超参数，例如通过反复试错来找到超参数合适的值。在少数情况下，超参数也可以通过模型训练学出。本书对此类情况不做讨论。

<h3>1.1.1.3. 模型预测 <a class="headerlink" href="#模型预测" >¶</a></h3>

模型训练完成后，我们将模型参数 $w_1, w_2, b$ 在优化算法停止时的值分别记作 ${\hat{w}_1, \hat{w}_2, \hat{b}}$ 。

注意，这里我们得到的并不一定是最小化损失函数的最优解 ${w_1^*, w_2^*, b^*}$ ，而是对最优解的一个近似。然后，我们就可以使用学出的线性回归模型 $x_1 \hat{w}_1 + x_2 \hat{w}_2 + \hat{b}$ 来估算训练数据集以外任意一栋面积（平方米）为 $x_1$ 、房龄（年）为 $x_2$ 的房屋的价格了。这里的估算也叫作模型预测、模型推断或模型测试。

<h2>1.1.2 线性回归的表示方法 <a class="headerlink" href="#线性回归的表示方法" >¶</a></h2>

我们已经阐述了线性回归的模型表达式、训练和预测。下面我们解释线性回归与神经网络的联系，以及线性回归的矢量计算表达式。

<h3>1.1.2.1. 神经网络图 <a class="headerlink" href="#神经网络图" >¶</a></h3>

在深度学习中，我们可以使用神经网络图直观地表现模型结构。为了更清晰地展示线性回归作为神经网络的结构，图1.1使用神经网络图表示本节中介绍的线性回归模型。神经网络图隐去了模型参数权重和偏差。

<img alt="线性回归是一个单层神经网络" src="https://github.com/Charmve/computer-vision-in-action/blob/main/docs/imgs/chapter03/3.1_linreg.svg" /><p class="caption"><span class="caption-number">图 1.1 线性回归是一个单层神经网络 

在图1.1所示的神经网络中，输入分别为 $x_1$ 和 $x_2$ ，因此输入层的输入个数为2。输入个数也叫特征数或特征向量维度。图1.1中网络的输出为 $o$ ，输出层的输出个数为1。需要注意的是，我们直接将图1.1中神经网络的输出 $o$ 作为线性回归的输出，即 $\hat{y} = o$ 。由于输入层并不涉及计算，按照惯例，图1.1所示的神经网络的层数为1。所以，线性回归是一个单层神经网络。输出层中负责计算 $o$ 的单元又叫神经元。在线性回归中， $o$ 的计算依赖于 $x_1$ 和 $x_2$ 。也就是说，输出层中的神经元和输入层中各个输入完全连接。因此，这里的输出层又叫全连接层（fully-connected
layer）或稠密层（dense layer）。

<h3>1.1.2.2. 矢量计算表达式 <a class="headerlink" href="#矢量计算表达式" >¶</a></h3>
  
在模型训练或预测时，我们常常会同时处理多个数据样本并用到矢量计算。在介绍线性回归的矢量计算表达式之前，让我们先考虑对两个向量相加的两种方法。
下面先定义两个1000维的向量。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [1]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="kn">from</span> <span class="nn">mxnet</span> <span class="kn">import</span> <span class="n">nd</span>
<span class="kn">from</span> <span class="nn">time</span> <span class="kn">import</span> <span class="n">time</span>

<span class="n">a</span> <span class="o">=</span> <span class="n">nd</span><span class="o">.</span><span class="n">ones</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="mi">1000</span><span class="p">)</span>
<span class="n">b</span> <span class="o">=</span> <span class="n">nd</span><span class="o">.</span><span class="n">ones</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="mi">1000</span><span class="p">)</span>
</pre></div>
</div>
</div>

向量相加的一种方法是，将这两个向量按元素逐一做标量加法。
<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [2]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="p">()</span>
<span class="n">c</span> <span class="o">=</span> <span class="n">nd</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="mi">1000</span><span class="p">)</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1000</span><span class="p">):</span>
    <span class="n">c</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">a</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">+</span> <span class="n">b</span><span class="p">[</span><span class="n">i</span><span class="p">]</span>
<span class="n">time</span><span class="p">()</span> <span class="o">-</span> <span class="n">start</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>Out[2]:
</pre></div>
</div>
<div class="output_area highlight-none notranslate"><div class="highlight"><pre>
<span></span>0.15927410125732422
</pre></div>
</div>
</div>

向量相加的另一种方法是，将这两个向量直接做矢量加法。
  
<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [3]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="p">()</span>
<span class="n">d</span> <span class="o">=</span> <span class="n">a</span> <span class="o">+</span> <span class="n">b</span>
<span class="n">time</span><span class="p">()</span> <span class="o">-</span> <span class="n">start</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>Out[3]:
</pre></div>
</div>
<div class="output_area highlight-none notranslate"><div class="highlight"><pre>
<span></span>0.0002071857452392578
</pre></div>
</div>
</div>

结果很明显，后者比前者更省时。因此，我们应该尽可能采用矢量计算，以提升计算效率。

让我们再次回到本节的房价预测问题。如果我们对训练数据集里的3个房屋样本（索引分别为1、2和3）逐一预测价格，将得到

$$
\[\begin{split}\begin{aligned}
\hat{y}^{(1)} &amp;= x_1^{(1)} w_1 + x_2^{(1)} w_2 + b,\\
\hat{y}^{(2)} &amp;= x_1^{(2)} w_1 + x_2^{(2)} w_2 + b,\\
\hat{y}^{(3)} &amp;= x_1^{(3)} w_1 + x_2^{(3)} w_2 + b.
\end{aligned}\end{split}\]
$$
  
现在，我们将上面3个等式转化成矢量计算。设

$$
\[\begin{split}\boldsymbol{\hat{y}} =
\begin{bmatrix}
    \hat{y}^{(1)} \\
    \hat{y}^{(2)} \\
    \hat{y}^{(3)}
\end{bmatrix},\quad
\boldsymbol{X} =
\begin{bmatrix}
    x_1^{(1)} &amp; x_2^{(1)} \\
    x_1^{(2)} &amp; x_2^{(2)} \\
    x_1^{(3)} &amp; x_2^{(3)}
\end{bmatrix},\quad
\boldsymbol{w} =
\begin{bmatrix}
    w_1 \\
    w_2
\end{bmatrix}.\end{split}\]
$$

对3个房屋样本预测价格的矢量计算表达式为 $\boldsymbol{\hat{y}} = \boldsymbol{X} \boldsymbol{w} + b,$

其中的加法运算使用了广播机制（参见<a class="reference internal" href="../chapter_prerequisite/ndarray.html"><span class="doc">“数据操作”</span></a>一节）。例如：
  
<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [4]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">a</span> <span class="o">=</span> <span class="n">nd</span><span class="o">.</span><span class="n">ones</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
<span class="n">b</span> <span class="o">=</span> <span class="mi">10</span>
<span class="n">a</span> <span class="o">+$  <span class="n">b</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>Out[4]:
</pre></div>
</div>
<div class="output_area highlight-none notranslate"><div class="highlight"><pre>
<span></span>[11. 11. 11.]
&lt;NDArray 3 @cpu(0)&gt;
</pre></div>
</div>
</div>

广义上讲，当数据样本数为 $n$ ，特征数为 $d$ 时，线性回归的矢量计算表达式为
$$\[\boldsymbol{\hat{y}} = \boldsymbol{X} \boldsymbol{w} + b,\]$$
  
其中模型输出 $\boldsymbol{\hat{y}} \in \mathbb{R}^{n \times 1}$，批量数据样本特征 $\boldsymbol{X} \in \mathbb{R}^{n \times d}$ ，权重 $\boldsymbol{w} \in \mathbb{R}^{d \times 1}$ ，偏差 $b \in \mathbb{R}$ 。相应地，批量数据样本标签 $\boldsymbol{y} \in \mathbb{R}^{n \times 1}$ 。
  
设模型参数 $\boldsymbol{\theta} = [w_1, w_2, b]^\top$ ，我们可以重写损失函数为
  
$$
\[\ell(\boldsymbol{\theta})=\frac{1}{2n}(\boldsymbol{\hat{y}}-\boldsymbol{y})^\top(\boldsymbol{\hat{y}}-\boldsymbol{y}).\]
$$
  
小批量随机梯度下降的迭代步骤将相应地改写为

$$
\[\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}   \nabla_{\boldsymbol{\theta}} \ell^{(i)}(\boldsymbol{\theta}),\]
$$

其中梯度是损失有关3个为标量的模型参数的偏导数组成的向量：

$$
\[\begin{split}\nabla_{\boldsymbol{\theta}} \ell^{(i)}(\boldsymbol{\theta})=\\
\begin{bmatrix}\\
    \frac{ \partial \ell^{(i)}(w_1, w_2, b)  }{\partial w_1} \\
    \frac{ \partial \ell^{(i)}(w_1, w_2, b)  }{\partial w_2} \\
    \frac{ \partial \ell^{(i)}(w_1, w_2, b)  }{\partial b}\\
\end{bmatrix}\\
=\\
\begin{bmatrix}
    x_1^{(i)} (x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}) \\
    x_2^{(i)} (x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}) \\
    x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\\
\end{bmatrix}\\
= \\
\begin{bmatrix}
    x_1^{(i)} \\
    x_2^{(i)} \\
    1
\end{bmatrix}
(\hat{y}^{(i)} - y^{(i)}).\end{split}\]
$$


<h3>1.1.3. 小结 <a class="headerlink" href="#小结" >¶</a></h3>
<ul class="simple">
<li>和大多数深度学习模型一样，对于线性回归这样一种单层神经网络，它的基本要素包括模型、训练数据、损失函数和优化算法。</li>
<li>既可以用神经网络图表示线性回归，又可以用矢量计算表示该模型。</li>
<li>应该尽可能采用矢量计算，以提升计算效率。</li>
</ul>


<h3>1.1.4. 练习 <a class="headerlink" href="#练习" >¶</a></h3>
  
- 使用其他包（如NumPy）或其他编程语言（如MATLAB），比较相加两个向量的两种方法的运行时间。


<h2> 1.2 从零实现线性回归 <a class="headerlink" href="#练习" >¶</a></h2>

在了解了线性回归的背景知识之后，现在我们可以动手实现它了。尽管强大的深度学习框架可以减少大量重复性工作，但若过于依赖它提供的便利，会导致我们很难深入理解深度学习是如何工作的。因此，本节将介绍如何只利用 ``NDArray`` 和 ``autograd`` 来实现一个线性回归的训练。
  
首先，导入本节中实验所需的包或模块，其中的matplotlib包可用于作图，且设置成嵌入显示。
  
<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [1]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="o">%</span><span class="n">matplotlib</span> <span class="n">inline</span>
<span class="kn">from</span> <span class="nn">IPython</span> <span class="kn">import</span> <span class="n">display</span>
<span class="kn">from</span> <span class="nn">matplotlib</span> <span class="kn">import</span> <span class="n">pyplot</span> <span class="k">as</span> <span class="n">plt</span>
<span class="kn">from</span> <span class="nn">mxnet</span> <span class="kn">import</span> <span class="n">autograd</span><span class="p">,</span> <span class="n">nd</span>
<span class="kn">import</span> <span class="nn">random</span>
</pre></div>
</div>
</div>


<h3>1.2.1. 生成数据集<a class="headerlink" href="#生成数据集" >¶</a></h3>

我们构造一个简单的人工训练数据集，它可以使我们能够直观比较学到的参数和真实的模型参数的区别。设训练数据集样本数为1000，输入个数（特征数）为2。给定随机生成的批量样本特征 $\boldsymbol{X} \in \mathbb{R}^{1000 \times 2}$ ，我们使用线性回归模型真实权重 $\boldsymbol{w} = [2, -3.4]^\top$ 和偏差 $b = 4.2$ ，以及一个随机噪声项 $\epsilon$ 来生成标签
  
$$
\[\boldsymbol{y} = \boldsymbol{X}\boldsymbol{w} + b + \epsilon,\]
$$

其中噪声项 $\epsilon$ 服从均值为0、标准差为0.01的正态分布。噪声代表了数据集中无意义的干扰。下面，让我们生成数据集。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [2]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">num_inputs</span> <span class="o">=</span> <span class="mi">2</span>
<span class="n">num_examples</span> <span class="o">=</span> <span class="mi">1000</span>
<span class="n">true_w</span> <span class="o">=</span> <span class="p">[</span><span class="mi">2</span><span class="p">,</span> <span class="o">-</span><span class="mf">3.4</span><span class="p">]</span>
<span class="n">true_b</span> <span class="o">=</span> <span class="mf">4.2</span>
<span class="n">features</span> <span class="o">=</span> <span class="n">nd</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">normal</span><span class="p">(</span><span class="n">scale</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="n">num_examples</span><span class="p">,</span> <span class="n">num_inputs</span><span class="p">))</span>
<span class="n">labels</span> <span class="o">=</span> <span class="n">true_w</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">*</span> <span class="n">features</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">+</span> <span class="n">true_w</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">*</span> <span class="n">features</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">]</span> <span class="o">+</span> <span class="n">true_b</span>
<span class="n">labels</span> <span class="o">+=</span> <span class="n">nd</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">normal</span><span class="p">(</span><span class="n">scale</span><span class="o">=</span><span class="mf">0.01</span><span class="p">,</span> <span class="n">shape</span><span class="o">=</span><span class="n">labels</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
</pre></div>
</div>
</div>

注意， ``features`` 的每一行是一个长度为2的向量，而 ``labels`` 的每一行是一个长度为1的向量（标量）。

<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [3]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">features</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">labels</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>Out[3]:
</pre></div>
</div>
<div class="output_area highlight-none notranslate"><div class="highlight"><pre>
<span></span>(
 [2.2122064 0.7740038]
 &lt;NDArray 2 @cpu(0)&gt;,
 [6.000587]
 &lt;NDArray 1 @cpu(0)&gt;)
</pre></div>
</div>
</div>

通过生成第二个特征 ``features[:, 1]`` 和标签  ``labels`` 的散点图，可以更直观地观察两者间的线性关系。

<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [4]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="k">def</span> <span class="nf">use_svg_display</span><span class="p">():</span>
    <span class="c1"># 用矢量图显示</span>
    <span class="n">display</span><span class="o">.</span><span class="n">set_matplotlib_formats</span><span class="p">(</span><span class="s1">&#39;svg&#39;</span><span class="p">)</span>

<span class="k">def</span> <span class="nf">set_figsize</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.5</span><span class="p">,</span> <span class="mf">2.5</span><span class="p">)):</span>
    <span class="n">use_svg_display</span><span class="p">()</span>
    <span class="c1"># 设置图的尺寸</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">rcParams</span><span class="p">[</span><span class="s1">&#39;figure.figsize&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">figsize</span>

<span class="n">set_figsize</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">features</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">asnumpy</span><span class="p">(),</span> <span class="n">labels</span><span class="o">.</span><span class="n">asnumpy</span><span class="p">(),</span> <span class="mi">1</span><span class="p">);</span>  <span class="c1"># 加分号只显示图</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt empty docutils container">
</div>

<div class="output_area docutils container">
<img alt="" src="../../imags/chapter03/chapter_deep-learning-basics_linear-regression-scratch_7_0.svg" />
</div>

我们将上面的 ``plt`` 作图函数以及 ``use_svg_display`` 函数和 ``set_figsize`` 函数定义在 ``d2lzh`` 包里。以后在作图时，我们将直接调用 ``d2lzh.plt`` 。由于 ``plt`` 在 ``d2lzh`` 包中是一个全局变量，我们在作图前只需要调用 ``d2lzh.set_figsize()`` 即可打印矢量图并设置图的尺寸。

<h3>1.2.2. 读取数据集 <a class="headerlink" href="#读取数据集" >¶</a></h3>

在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。这里我们定义一个函数：它每次返回 ``batch_size`` （批量大小）个随机样本的特征和标签。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [5]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="c1"># 本函数已保存在d2lzh包中方便以后使用</span>
<span class="k">def</span> <span class="nf">data_iter</span><span class="p">(</span><span class="n">batch_size</span><span class="p">,</span> <span class="n">features</span><span class="p">,</span> <span class="n">labels</span><span class="p">):</span>
    <span class="n">num_examples</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">features</span><span class="p">)</span>
    <span class="n">indices</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">num_examples</span><span class="p">))</span>
    <span class="n">random</span><span class="o">.</span><span class="n">shuffle</span><span class="p">(</span><span class="n">indices</span><span class="p">)</span>  <span class="c1"># 样本的读取顺序是随机的</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">num_examples</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">):</span>
        <span class="n">j</span> <span class="o">=</span> <span class="n">nd</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">indices</span><span class="p">[</span><span class="n">i</span><span class="p">:</span> <span class="nb">min</span><span class="p">(</span><span class="n">i</span> <span class="o">+</span> <span class="n">batch_size</span><span class="p">,</span> <span class="n">num_examples</span><span class="p">)])</span>
        <span class="k">yield</span> <span class="n">features</span><span class="o">.</span><span class="n">take</span><span class="p">(</span><span class="n">j</span><span class="p">),</span> <span class="n">labels</span><span class="o">.</span><span class="n">take</span><span class="p">(</span><span class="n">j</span><span class="p">)</span>  <span class="c1"># take函数根据索引返回对应元素</span>
</pre></div>
</div>
</div>

让我们读取第一个小批量数据样本并打印。每个批量的特征形状为(10, 2)，分别对应批量大小和输入个数；标签形状为批量大小。

<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [6]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">batch_size</span> <span class="o">=</span> <span class="mi">10</span>

<span class="k">for</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="ow">in</span> <span class="n">data_iter</span><span class="p">(</span><span class="n">batch_size</span><span class="p">,</span> <span class="n">features</span><span class="p">,</span> <span class="n">labels</span><span class="p">):</span>
    <span class="k">print</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
    <span class="k">break</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt empty docutils container">
</div>
<div class="output_area docutils container">
<div class="highlight"><pre>

[[-0.62958384  0.3508038 ]
 [ 1.4136469  -1.416811  ]
 [ 0.22276424  0.2804388 ]
 [ 0.377959   -0.34565544]
 [-0.613936   -2.6247382 ]
 [-0.05510042 -0.5667109 ]
 [-1.3360355   1.0035783 ]
 [-2.338083   -0.32677847]
 [-0.18353732  0.5803962 ]
 [ 0.7247974  -0.8402941 ]]
&lt;NDArray 10x2 @cpu(0)&gt;
[ 1.7371575 11.866432   3.6991339  6.1261005 11.895512   6.0134444
 -1.8906523  0.6198239  1.8458312  8.513775 ]
&lt;NDArray 10 @cpu(0)&gt;
</pre></div></div>
</div>
</div>

<h3>1.2.3. 初始化模型参数 <a class="headerlink" href="#初始化模型参数" >¶</a></h3>

我们将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [7]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">w</span> <span class="o">=</span> <span class="n">nd</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">normal</span><span class="p">(</span><span class="n">scale</span><span class="o">=</span><span class="mf">0.01</span><span class="p">,</span> <span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="n">num_inputs</span><span class="p">,</span> <span class="mi">1</span><span class="p">))</span>
<span class="n">b</span> <span class="o">=</span> <span class="n">nd</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,))</span>
</pre></div>
</div>
</div>

之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此我们需要创建它们的梯度。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [8]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">w</span><span class="o">.</span><span class="n">attach_grad</span><span class="p">()</span>
<span class="n">b</span><span class="o">.</span><span class="n">attach_grad</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>

<h3>1.2.4. 定义模型 <a class="headerlink" href="#定义模型" >¶</a></h3>

下面是线性回归的矢量计算表达式的实现。我们使用 ``dot`` 函数做矩阵乘法。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [9]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="k">def</span> <span class="nf">linreg</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">w</span><span class="p">,</span> <span class="n">b</span><span class="p">):</span>  <span class="c1"># 本函数已保存在d2lzh包中方便以后使用</span>
    <span class="k">return</span> <span class="n">nd</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">w</span><span class="p">)</span> <span class="o">+</span> <span class="n">b</span>
</pre></div>
</div>
</div>
</div>

<h3>1.2.5. 定义损失函数 <a class="headerlink" href="#定义损失函数" >¶</a></h3>

我们使用上一节描述的平方损失来定义线性回归的损失函数。在实现中，我们需要把真实值 ``y`` 变形成预测值 ``y_hat`` 的形状。以下函数返回的结果也将和 ``y_hat`` 的形状相同。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [10]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="k">def</span> <span class="nf">squared_loss</span><span class="p">(</span><span class="n">y_hat</span><span class="p">,</span> <span class="n">y</span><span class="p">):</span>  <span class="c1"># 本函数已保存在d2lzh包中方便以后使用</span>
    <span class="k">return</span> <span class="p">(</span><span class="n">y_hat</span> <span class="o">-</span> <span class="n">y</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="n">y_hat</span><span class="o">.</span><span class="n">shape</span><span class="p">))</span> <span class="o">**</span> <span class="mi">2</span> <span class="o">/</span> <span class="mi">2</span>
</pre></div>
</div>
</div>


<h3>1.2.6. 定义优化算法 <a class="headerlink" href="#定义优化算法" >¶</a></h3>
以下的 ``sgd`` 函数实现了上一节中介绍的小批量随机梯度下降算法。它通过不断迭代模型参数来优化损失函数。这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和。我们将它除以批量大小来得到平均值。
<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [11]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="k">def</span> <span class="nf">sgd</span><span class="p">(</span><span class="n">params</span><span class="p">,</span> <span class="n">lr</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">):</span>  <span class="c1"># 本函数已保存在d2lzh包中方便以后使用</span>
    <span class="k">for</span> <span class="n">param</span> <span class="ow">in</span> <span class="n">params</span><span class="p">:</span>
        <span class="n">param</span><span class="p">[:]</span> <span class="o">=</span> <span class="n">param</span> <span class="o">-</span> <span class="n">lr</span> <span class="o">*</span> <span class="n">param</span><span class="o">.</span><span class="n">grad</span> <span class="o">/</span> <span class="n">batch_size</span>
</pre></div>
</div>
</div>


<h3>1.2.7. 训练模型 <a class="headerlink" href="#训练模型" >¶</a></h3>

在训练中，我们将多次迭代模型参数。在每次迭代中，我们根据当前读取的小批量数据样本（特征 ``X`` 和标签 ``y`` ），通过调用反向函数 ``backward`` 计算小批量随机梯度，并调用优化算法 ``sgd`` 迭代模型参数。由于我们之前设批量大小 ``batch_size`` 为10，每个小批量的损失 ``l`` 的形状为(10, 1)。回忆一下<a class="reference internal" href="../chapter_prerequisite/autograd.html"><span class="doc">“自动求梯度”</span></a>一节。由于变量 ``l`` 并不是一个标量，运行 ``l.backward()`` 将对 ``l`` 中元素求和得到新的变量，再求该变量有关模型参数的梯度。

在一个迭代周期（epoch）中，我们将完整遍历一遍 ``data_iter`` 函数，并对训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。这里的迭代周期个数 ``num_epochs`` 和学习率 ``lr`` 都是超参数，分别设3和0.03。在实践中，大多超参数都需要通过反复试错来不断调节。虽然迭代周期数设得越大模型可能越有效，但是训练时间可能过长。我们会在后面“优化算法”一章中详细介绍学习率对模型的影响。

<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [12]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">lr</span> <span class="o">=</span> <span class="mf">0.03</span>
<span class="n">num_epochs</span> <span class="o">=</span> <span class="mi">3</span>
<span class="n">net</span> <span class="o">=</span> <span class="n">linreg</span>
<span class="n">loss</span> <span class="o">=</span> <span class="n">squared_loss</span>

<span class="k">for</span> <span class="n">epoch</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">num_epochs</span><span class="p">):</span>  <span class="c1"># 训练模型一共需要num_epochs个迭代周期</span>
    <span class="c1"># 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X</span>
    <span class="c1"># 和y分别是小批量样本的特征和标签</span>
    <span class="k">for</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="ow">in</span> <span class="n">data_iter</span><span class="p">(</span><span class="n">batch_size</span><span class="p">,</span> <span class="n">features</span><span class="p">,</span> <span class="n">labels</span><span class="p">):</span>
        <span class="k">with</span> <span class="n">autograd</span><span class="o">.</span><span class="n">record</span><span class="p">():</span>
            <span class="n">l</span> <span class="o">=</span> <span class="n">loss</span><span class="p">(</span><span class="n">net</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">w</span><span class="p">,</span> <span class="n">b</span><span class="p">),</span> <span class="n">y</span><span class="p">)</span>  <span class="c1"># l是有关小批量X和y的损失</span>
        <span class="n">l</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>  <span class="c1"># 小批量的损失对模型参数求梯度</span>
        <span class="n">sgd</span><span class="p">([</span><span class="n">w</span><span class="p">,</span> <span class="n">b</span><span class="p">],</span> <span class="n">lr</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">)</span>  <span class="c1"># 使用小批量随机梯度下降迭代模型参数</span>
    <span class="n">train_l</span> <span class="o">=</span> <span class="n">loss</span><span class="p">(</span><span class="n">net</span><span class="p">(</span><span class="n">features</span><span class="p">,</span> <span class="n">w</span><span class="p">,</span> <span class="n">b</span><span class="p">),</span> <span class="n">labels</span><span class="p">)</span>
    <span class="k">print</span><span class="p">(</span><span class="s1">&#39;epoch </span><span class="si">%d</span><span class="s1">, loss </span><span class="si">%f</span><span class="s1">&#39;</span> <span class="o">%</span> <span class="p">(</span><span class="n">epoch</span> <span class="o">+</span> <span class="mi">1</span><span class="p">,</span> <span class="n">train_l</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span><span class="o">.</span><span class="n">asnumpy</span><span class="p">()))</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt empty docutils container">
</div>
<div class="output_area docutils container">
<div class="highlight"><pre>
epoch 1, loss 0.040563
epoch 2, loss 0.000153
epoch 3, loss 0.000051
</pre></div></div>
</div>

训练完成后，我们可以比较学到的参数和用来生成训练集的真实参数。它们应该很接近。

<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [13]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">true_w</span><span class="p">,</span> <span class="n">w</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>Out[13]:
</pre></div>
</div>
<div class="output_area highlight-none notranslate"><div class="highlight"><pre>
<span></span>([2, -3.4],
 [[ 1.9995433]
  [-3.3996048]]
 &lt;NDArray 2x1 @cpu(0)&gt;)
</pre></div>
</div>
</div>
<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [14]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">true_b</span><span class="p">,</span> <span class="n">b</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>Out[14]:
</pre></div>
</div>
<div class="output_area highlight-none notranslate"><div class="highlight"><pre>
<span></span>(4.2,
 [4.1994247]
 &lt;NDArray 1 @cpu(0)&gt;)
</pre></div>
</div>
</div>
</div>
<div class="section" id="小结">
<h3>1.2.8. 小结 <a class="headerlink" href="#小结" >¶</a></h3>
<ul class="simple">
<li>可以看出，仅使用 ``NDArray`` 和 ``autograd`` 模块就可以很容易地实现一个模型。接下来，本书会在此基础上描述更多深度学习模型，并介绍怎样使用更简洁的代码（见下一节）来实现它们。</li>
</ul>

<h3>1.2.9. 练习 <a class="headerlink" href="#练习" >¶</a></h3>
<ul class="simple">
<li>为什么 ``squared_loss`` 函数中需要使用 ``reshape`` 函数？</li>
<li>尝试使用不同的学习率，观察损失函数值的下降快慢。</li>
<li>如果样本个数不能被批量大小整除， ``data_iter`` 函数的行为会有什么变化？</li>
</ul>

<h2> 1.3 线性回归的简洁实现</h2>

随着深度学习框架的发展，开发深度学习应用变得越来越便利。实践中，我们通常可以用比上一节更简洁的代码来实现同样的模型。在本节中，我们将介绍如何使用MXNet提供的Gluon接口更方便地实现线性回归的训练。

<h3>3.3.1. 生成数据集 <a class="headerlink" href="#生成数据集" >¶</a></h3>

我们生成与上一节中相同的数据集。其中 ``features`` 是训练数据特征， ``labels`` 是标签。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [1]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="kn">from</span> <span class="nn">mxnet</span> <span class="kn">import</span> <span class="n">autograd</span><span class="p">,</span> <span class="n">nd</span>

<span class="n">num_inputs</span> <span class="o">=</span> <span class="mi">2</span>
<span class="n">num_examples</span> <span class="o">=</span> <span class="mi">1000</span>
<span class="n">true_w</span> <span class="o">=</span> <span class="p">[</span><span class="mi">2</span><span class="p">,</span> <span class="o">-</span><span class="mf">3.4</span><span class="p">]</span>
<span class="n">true_b</span> <span class="o">=</span> <span class="mf">4.2</span>
<span class="n">features</span> <span class="o">=</span> <span class="n">nd</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">normal</span><span class="p">(</span><span class="n">scale</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="n">num_examples</span><span class="p">,</span> <span class="n">num_inputs</span><span class="p">))</span>
<span class="n">labels</span> <span class="o">=</span> <span class="n">true_w</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">*</span> <span class="n">features</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">+</span> <span class="n">true_w</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">*</span> <span class="n">features</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">]</span> <span class="o">+</span> <span class="n">true_b</span>
<span class="n">labels</span> <span class="o">+=</span> <span class="n">nd</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">normal</span><span class="p">(</span><span class="n">scale</span><span class="o">=</span><span class="mf">0.01</span><span class="p">,</span> <span class="n">shape</span><span class="o">=</span><span class="n">labels</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
</pre></div>
</div>
</div>

<h3>1.3.2. 读取数据集 <a class="headerlink" href="#读取数据集" >¶</a></h3>

Gluon提供了 ``data`` 包来读取数据。由于 ``data`` 常用作变量名，我们将导入的 ``data`` 模块用添加了Gluon首字母的假名 ``gdata`` 代替。在每一次迭代中，我们将随机读取包含10个数据样本的小批量。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [2]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="kn">from</span> <span class="nn">mxnet.gluon</span> <span class="kn">import</span> <span class="n">data</span> <span class="k">as</span> <span class="n">gdata</span>

<span class="n">batch_size</span> <span class="o">=</span> <span class="mi">10</span>
<span class="c1"># 将训练数据的特征和标签组合</span>
<span class="n">dataset</span> <span class="o">=</span> <span class="n">gdata</span><span class="o">.</span><span class="n">ArrayDataset</span><span class="p">(</span><span class="n">features</span><span class="p">,</span> <span class="n">labels</span><span class="p">)</span>
<span class="c1"># 随机读取小批量</span>
<span class="n">data_iter</span> <span class="o">=</span> <span class="n">gdata</span><span class="o">.</span><span class="n">DataLoader</span><span class="p">(</span><span class="n">dataset</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="bp">True</span><span class="p">)</span>
</pre></div>
</div>
</div>

这里 ``data_iter`` 的使用与上一节中的一样。让我们读取并打印第一个小批量数据样本。

<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [3]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="k">for</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="ow">in</span> <span class="n">data_iter</span><span class="p">:</span>
    <span class="k">print</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
    <span class="k">break</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt empty docutils container">
</div>
<div class="output_area docutils container">
<div class="highlight"><pre>
[[-0.88425034  1.1412233 ]
 [ 0.9561315  -0.8629183 ]
 [ 0.74419165  0.30639967]
 [ 0.9676651   0.5122743 ]
 [-0.1824717  -0.6532737 ]
 [-0.19343433 -0.4232046 ]
 [-1.1197332  -1.0690794 ]
 [-1.1318913   0.33254048]
 [ 0.6360805  -0.5844077 ]
 [-0.5839395  -0.19038047]]
&lt;NDArray 10x2 @cpu(0)&gt;
[-1.4545211   9.049869    4.6543612   4.405501    6.064423    5.259082
  5.6068754   0.79540896  7.460776    3.688919  ]
&lt;NDArray 10 @cpu(0)&gt;
</pre></div></div>
</div>

<h3>1.3.3. 定义模型 <a class="headerlink" href="#定义模型" >¶</a></h3>

在上一节从零开始的实现中，我们需要定义模型参数，并使用它们一步步描述模型是怎样计算的。当模型结构变得更复杂时，这些步骤将变得更烦琐。其实，Gluon提供了大量预定义的层，这使我们只需关注使用哪些层来构造模型。下面将介绍如何使用Gluon更简洁地定义线性回归。

首先，导入 ``nn`` 模块。实际上，“nn”是 neural networks（神经网络）的缩写。顾名思义，该模块定义了大量神经网络的层。我们先定义一个模型变量 ``net`` ，它是一个 ``Sequential`` 实例。在Gluon中， ``Sequential`` 实例可以看作是一个串联各个层的容器。在构造模型时，我们在该容器中依次添加层。当给定输入数据时，容器中的每一层将依次计算并将输出作为下一层的输入。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [4]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="kn">from</span> <span class="nn">mxnet.gluon</span> <span class="kn">import</span> <span class="n">nn</span>

<span class="n">net</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Sequential</span><span class="p">()</span>
</pre></div>
</div>
</div>

回顾图1.1中线性回归在神经网络图中的表示。作为一个单层神经网络，线性回归输出层中的神经元和输入层中各个输入完全连接。因此，线性回归的输出层又叫全连接层。在Gluon中，全连接层是一个 ``Dense`` 实例。我们定义该层输出个数为1。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [5]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">net</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Dense</span><span class="p">(</span><span class="mi">1</span><span class="p">))</span>
</pre></div>
</div>
</div>

值得一提的是，在Gluon中我们无须指定每一层输入的形状，例如线性回归的输入个数。当模型得到数据时，例如后面执行 ``net(X)`` 时，模型将自动推断出每一层的输入个数。我们将在之后“深度学习计算”一章详细介绍这种机制。Gluon的这一设计为模型开发带来便利。

</div>
<div class="section" id="初始化模型参数">
<h3>1.3.4. 初始化模型参数 <a class="headerlink" href="#初始化模型参数" >¶</a></h3>

在使用 ``net`` 前，我们需要初始化模型参数，如线性回归模型中的权重和偏差。我们从MXNet导入 ``init`` 模块。该模块提供了模型参数初始化的各种方法。这里的 ``init`` 是 ``initializer`` 的缩写形式。我们通过 ``init.Normal(sigma=0.01)`` 指定权重参数每个元素将在初始化时随机采样于均值为0、标准差为0.01的正态分布。偏差参数默认会初始化为零。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [6]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="kn">from</span> <span class="nn">mxnet</span> <span class="kn">import</span> <span class="n">init</span>

<span class="n">net</span><span class="o">.</span><span class="n">initialize</span><span class="p">(</span><span class="n">init</span><span class="o">.</span><span class="n">Normal</span><span class="p">(</span><span class="n">sigma</span><span class="o">=</span><span class="mf">0.01</span><span class="p">))</span>
</pre></div>
</div>
</div>


<h2>1.3.5. 定义损失函数 <a class="headerlink" href="#定义损失函数" >¶</a></h2>

在Gluon中， ``loss`` 模块定义了各种损失函数。我们用假名 ``gloss`` 代替导入的 ``loss`` 模块，并直接使用它提供的平方损失作为模型的损失函数。

<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [7]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="kn">from</span> <span class="nn">mxnet.gluon</span> <span class="kn">import</span> <span class="n">loss</span> <span class="k">as</span> <span class="n">gloss</span>

<span class="n">loss</span> <span class="o">=</span> <span class="n">gloss</span><span class="o">.</span><span class="n">L2Loss</span><span class="p">()</span>  <span class="c1"># 平方损失又称L2范数损失</span>
</pre></div>
</div>
</div>
</div>
<div class="section" id="定义优化算法">
<h3>1.3.6. 定义优化算法 <a class="headerlink" href="#定义优化算法" >¶</a></h3>
同样，我们也无须实现小批量随机梯度下降。在导入Gluon后，我们创建一个 ``Trainer`` 实例，并指定学习率为0.03的小批量随机梯度下降（ ``sgd`` ）为优化算法。该优化算法将用来迭代 ``net`` 实例所有通过 ``add`` 函数嵌套的层所包含的全部参数。这些参数可以通过 ``collect_params`` 函数获取。
<div class="nbinput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [8]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="kn">from</span> <span class="nn">mxnet</span> <span class="kn">import</span> <span class="n">gluon</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">gluon</span><span class="o">.</span><span class="n">Trainer</span><span class="p">(</span><span class="n">net</span><span class="o">.</span><span class="n">collect_params</span><span class="p">(),</span> <span class="s1">&#39;sgd&#39;</span><span class="p">,</span> <span class="p">{</span><span class="s1">&#39;learning_rate&#39;</span><span class="p">:</span> <span class="mf">0.03</span><span class="p">})</span>
</pre></div>
</div>
</div>


<h3>1.3.7. 训练模型 <a class="headerlink" href="#训练模型" >¶</a></h3>

在使用Gluon训练模型时，我们通过调用 ``Trainer`` 实例的 ``step`` 函数来迭代模型参数。上一节中我们提到，由于变量 ``l`` 是长度为 ``batch_size`` 的一维 ``NDArray`` ，执行 ``l.backward()`` 等价于执行 ``l.sum().backward()`` 。按照小批量随机梯度下降的定义，我们在 ``step`` 函数中指明批量大小，从而对批量中样本梯度求平均。

<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [9]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">num_epochs</span> <span class="o">=</span> <span class="mi">3</span>
<span class="k">for</span> <span class="n">epoch</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">num_epochs</span> <span class="o">+</span> <span class="mi">1</span><span class="p">):</span>
    <span class="k">for</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="ow">in</span> <span class="n">data_iter</span><span class="p">:</span>
        <span class="k">with</span> <span class="n">autograd</span><span class="o">.</span><span class="n">record</span><span class="p">():</span>
            <span class="n">l</span> <span class="o">=</span> <span class="n">loss</span><span class="p">(</span><span class="n">net</span><span class="p">(</span><span class="n">X</span><span class="p">),</span> <span class="n">y</span><span class="p">)</span>
        <span class="n">l</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
        <span class="n">trainer</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">batch_size</span><span class="p">)</span>
    <span class="n">l</span> <span class="o">=</span> <span class="n">loss</span><span class="p">(</span><span class="n">net</span><span class="p">(</span><span class="n">features</span><span class="p">),</span> <span class="n">labels</span><span class="p">)</span>
    <span class="k">print</span><span class="p">(</span><span class="s1">&#39;epoch </span><span class="si">%d</span><span class="s1">, loss: </span><span class="si">%f</span><span class="s1">&#39;</span> <span class="o">%</span> <span class="p">(</span><span class="n">epoch</span><span class="p">,</span> <span class="n">l</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span><span class="o">.</span><span class="n">asnumpy</span><span class="p">()))</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt empty docutils container">
</div>
<div class="output_area docutils container">
<div class="highlight"><pre>
epoch 1, loss: 0.040660
epoch 2, loss: 0.000151
epoch 3, loss: 0.000051
</pre></div></div>
</div>

下面我们分别比较学到的模型参数和真实的模型参数。我们从 ``net`` 获得需要的层，并访问其权重（ ``weight`` ）和偏差（ ``bias`` ）。学到的参数和真实的参数很接近。

<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [10]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">dense</span> <span class="o">=</span> <span class="n">net</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">true_w</span><span class="p">,</span> <span class="n">dense</span><span class="o">.</span><span class="n">weight</span><span class="o">.</span><span class="n">data</span><span class="p">()</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>Out[10]:
</pre></div>
</div>
<div class="output_area highlight-none notranslate"><div class="highlight"><pre>
<span></span>([2, -3.4],
 [[ 1.9999468 -3.4000072]]
 &lt;NDArray 1x2 @cpu(0)&gt;)
</pre></div>
</div>
</div>
<div class="nbinput docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>In [11]:
</pre></div>
</div>
<div class="input_area highlight-python notranslate"><div class="highlight"><pre>
<span></span><span class="n">true_b</span><span class="p">,</span> <span class="n">dense</span><span class="o">.</span><span class="n">bias</span><span class="o">.</span><span class="n">data</span><span class="p">()</span>
</pre></div>
</div>
</div>
<div class="nboutput nblast docutils container">
<div class="prompt highlight-none notranslate"><div class="highlight"><pre>
<span></span>Out[11]:
</pre></div>
</div>
<div class="output_area highlight-none notranslate"><div class="highlight"><pre>
<span></span>(4.2,
 [4.1993876]
 &lt;NDArray 1 @cpu(0)&gt;)
</pre></div>
</div>
</div>

<h3>1.3.8. 小结 <a class="headerlink" href="#小结">¶</a></h3>
  
- 使用Gluon可以更简洁地实现模型。
- 在Gluon中， ``data`` 模块提供了有关数据处理的工具， ``nn`` 模块定义了大量神经网络的层， ``loss`` 模块定义了各种损失函数。</li>
- MXNet的 ``initializer`` 模块提供了模型参数初始化的各种方法。

<h3>1.3.9. 练习 <a class="headerlink" href="#练习">¶</a></h3>
<ul class="simple">
<li>如果将 ``l</span> <span class="pre">=</span> <span class="pre">loss(net(X),</span> <span class="pre">y)`` 替换成 ``l</span> <span class="pre">=</span> <span class="pre">loss(net(X),</span> <span class="pre">y).mean()`` ，我们需要将 ``trainer.step(batch_size)`` 相应地改成 ``trainer.step(1)`` 。这是为什么呢？</li>
<li>查阅MXNet文档，看看 ``gluon.loss`` 和 ``init`` 模块里提供了哪些损失函数和初始化方法。</li>
<li>如何访问 ``dense.weight`` 的梯度？</li>
</ul>

<br>

> 本文引用自 李沐 《动手写深度学习》 d2l-zh.ai 非常感谢, <a href = "https://charmve.github.io/computer-vision-in-action/#/REFERENCE">参考文献</a>!

