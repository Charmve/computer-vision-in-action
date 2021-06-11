<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

# 第 5 章 图神经网络

作者: 张伟 (Charmve)

日期: 2021/06/11

- 第 5 章 [图神经网络](https://charmve.github.io/computer-vision-in-action/#/chapter8/chapter8)
    - 5.1 [历史脉络](/docs/1_理论篇/chapter5_图神经网络/chapter5_图神经网络.md#51-历史脉络)
    - 5.2 [图神经网络(Graph Neural Network)](https://www.cnblogs.com/SivilTaram/p/graph_neural_network_1.html)
      - 5.2.1 [状态更新与输出](/docs/1_理论篇/chapter5_图神经网络/chapter5_图神经网络.md#51-常见数据集)
      - 5.2.2 [不动点理论](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.1_著名数据集.md#812-pytorch数据集及读取方法简介)
      - 5.2.3 [具体实现](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.1_著名数据集.md#813-数据增强简介)
      - 5.2.4 [模型学习]()
      - 5.2.5 [GNN与RNN](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.2_基准BenchMark.md)
      - 5.2.6 [GNN的局限]()
    - 5.3 [门控图神经网络(Gated Graph Neural Network)]
      - 5.3.1 状态更新
      - 5.3.2 实例1:到达判断
      - 5.3.3 实例2:语义解析
      - 5.3.4 GNN与GGNN
    - 5.4 [图卷积神经网络(GCNN)](https://www.cnblogs.com/SivilTaram/p/graph_neural_network_2.html)
      - 5.4.1 图卷积缘起
      - 5.4.2 图卷积框架(Framework)
      - 5.4.3 再谈卷积
        - 5.4.3.1 基础概念
        - 5.4.3.2 实例:掷骰子问题
      - 5.4.4 空域卷积(Spatial Convolution)
      - 5.4.5 消息传递网络(Message Passing Neural Network)
      - 5.4.6 图采样与聚合(Graph Sample and Aggregate)
      - 5.4.7 图结构序列化(PATCHY-SAN)
      - 5.4.8 频域卷积(Spectral Convolution)
        - 5.4.8.1 前置内容
        - 5.4.8.2 傅里叶变换(Fourier Transform)
        - 5.4.8.3 图上的傅里叶变换
        - 5.4.8.4 频域卷积网络(Spectral CNN)
        - 5.4.8.5 切比雪夫网络(ChebNet)
    - 5.5 [生成图表示](https://www.cnblogs.com/SivilTaram/p/graph_neural_network_3.html)
      - 5.5.1 图读出操作(ReadOut)
      - 5.5.2 基于统计的方法(Statistics Category)
      - 5.5.3 基于学习的方法(Learning Category)
      - 5.5.4 其他方法
    - 小结
    - 参考文献


## 5.1 历史脉络
在开始正文之前，笔者先带大家回顾一下图神经网络的发展历史。不过，因为图神经网络的发展分支非常之多，笔者某些叙述可能并不全面，一家之言仅供各位读者参考：

1. 图神经网络的概念最早在2005年提出。2009年Franco博士在其论文 [2]中定义了图神经网络的理论基础，笔者呆会要讲的第一种图神经网络也是基于这篇论文。
2. 最早的GNN主要解决的还是如分子结构分类等严格意义上的图论问题。但实际上欧式空间(比如像图像 Image)或者是序列(比如像文本 Text)，许多常见场景也都可以转换成图(Graph)，然后就能使用图神经网络技术来建模。
3. 2009年后图神经网络也陆续有一些相关研究，但没有太大波澜。直到2013年，在图信号处理(Graph Signal Processing)的基础上，Bruna(这位是LeCun的学生)在文献 [3]中首次提出图上的基于频域(Spectral-domain)和基于空域(Spatial-domain)的卷积神经网络。
4. 其后至今，学界提出了很多基于空域的图卷积方式，也有不少学者试图通过统一的框架将前人的工作统一起来。而基于频域的工作相对较少，只受到部分学者的青睐。
5. 值得一提的是，图神经网络与图表示学习(Represent Learning for Graph)的发展历程也惊人地相似。2014年，在word2vec [4]的启发下，Perozzi等人提出了DeepWalk [5]，开启了深度学习时代图表示学习的大门。更有趣的是，就在几乎一样的时间，Bordes等人提出了大名鼎鼎的TransE [6]，为知识图谱的分布式表示(Represent Learning for Knowledge Graph)奠定了基础。


## 5.2 图神经网络(Graph Neural Network)

首先要澄清一点，除非特别指明，本文中所提到的图均指图论中的图(Graph)。它是一种由若干个结点(Node)及连接两个结点的边(Edge)所构成的图形，用于刻画不同结点之间的关系。下面是一个生动的例子，图片来自论文[14]:


## 参考文献

[1]. A Comprehensive Survey on Graph Neural Networks, https://arxiv.org/abs/1901.00596

[2]. The graph neural network model, https://persagen.com/files/misc/scarselli2009graph.pdf

[3]. Spectral networks and locally connected networks on graphs, https://arxiv.org/abs/1312.6203

[4]. Distributed Representations of Words and Phrases and their Compositionality, http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases

[5]. DeepWalk: Online Learning of Social Representations, https://arxiv.org/abs/1403.6652

[6]. Translating Embeddings for Modeling Multi-relational Data, https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

[7]. Deep Learning on Graphs: A Survey, https://arxiv.org/abs/1812.04202

[8]. 如何理解Graph Convolutional Network（GCN）? https://www.zhihu.com/question/54504471

[9]. Almeida–Pineda recurrent backpropagation, https://www.wikiwand.com/en/Almeida–Pineda_recurrent_backpropagation

[10]. Gated graph sequence neural networks, https://arxiv.org/abs/1511.05493

[11]. Representing Schema Structure with Graph Neural Networks for Text-to-SQL Parsing, https://arxiv.org/abs/1905.06241

[12]. Spider1.0 Yale Semantic Parsing and Text-to-SQL Challenge, https://yale-lily.github.io/spider

[13]. https://www.wikiwand.com/en/Laplacian_matrix

[14]. Graph Neural Networks: A Review of Methods and Applications, https://arxiv.org/pdf/1812.08434
