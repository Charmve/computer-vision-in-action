<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

# 第 8 章 著名数据集及基准

作者: 张伟 (Charmve)

日期: 2021/06/06

- 第 8 章 [著名数据集及基准](https://charmve.github.io/computer-vision-in-action/#/chapter8/chapter8)
    - 8.1 [数据集](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.1_著名数据集.md)
        - 8.1.1 [常见数据集](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.1_著名数据集.md#811-常见数据集)
          - 8.1.1.1 [ImageNet](https://image-net.org/)
          - 8.1.1.2 [MNIST](http://yann.lecun.com/exdb/mnist/)
          - 8.1.1.3 [COCO](https://cocodataset.org/)
          - 8.1.1.4 [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
        - 8.1.2 [Pytorch数据集及读取方法简介](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.1_著名数据集.md#812-pytorch数据集及读取方法简介)
        - 8.1.3 [数据增强简介](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.1_著名数据集.md#813-数据增强简介)
        - [总结](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.1_著名数据集.md#总结)
    - 8.2 [基准](/docs/2_实战篇/chapter8_著名数据集及基准/chapter8.2_基准BenchMark.md)
    - 小结
    - 参考文献
    

## 8.2 基准

### 8.2.1 Benchmark简介
Benchmark是一个评价方式，在整个计算机领域有着长期的应用。正如维基百科上的解释“As computer architecture advanced, it became more difficult to compare the performance of various computer systems simply by looking at their specifications.Therefore, tests were developed that allowed comparison of different architectures.”Benchmark在计算机领域应用最成功的就是性能测试，主要测试负载的执行时间、传输速度、吞吐量、资源占用率等。

性能调优的两大利器是Benchmark和profile工具。Benchmark用压力测试挖掘整个系统的性能状况，而profile工具最大限度地呈现系统的运行状态和性能指标，方便用户诊断性能问题和进行调优。

### 8.2.2 Benchmark的组成

Benchmark的核心由3部分组成：数据集、 工作负载、度量指标。

#### 8.2.2.1 数据集
数据类型分为结构化数据、半结构化数据和非结构化数据。由于大数据环境下的数据类型复杂，负载多样，所以大数据Benchmark需要生成3种类型的数据和对应负载。

- 结构化数据：传统的关系数据模型，可用二维表结构表示。典型场景有电商交易、财务系统、医疗HIS数据库、政务信息化系统等等；

- 半结构化数据：类似XML、HTML之类，自描述，数据结构和内容混杂在一起。典型应用场景有邮件系统、Web搜索引擎存储、教学资源库、档案系统等等，可以考虑使用Hbase等典型的KeyValue存储；

- 非结构化数据：各种文档、图片、视频和音频等。典型的应用有视频网站、图片相册、交通视频监控等等。

#### 8.2.2.2 工作负载
互联网领域数据庞大，用户量大，成为大数据问题产生的天然土壤。对工作负载理解和设计可以从以下几个维度来看

- 密集计算类型：CPU密集型计算、IO密集型计算、网络密集型计算；

- 计算范式：SQL、批处理、流计算、图计算、机器学习；

- 计算延迟：在线计算、离线计算、实时计算；

- 应用领域：搜索引擎、社交网络、电子商务、地理位置、媒体、游戏。

#### 8.2.2.3 度量指标

性能高估的两大利器就是Benchmark和Profile工具。Benchmark用压力测试挖掘整个系统的性能状况，而Profile工具最大限度地呈现系统的运行时状态和性能指标，方便用户诊断性能问题和进行调优。

- 工具的使用
  - 在架构层面：perf、nmon等工具和命令；
  - 在JVM层面：btrace、Jconsole、JVisualVM、JMap、JStack等工具和命令；
  - 在Spark层面：web ui、console log，也可以修改Spark源码打印日志进行性能监控。

- 度量指标
- 从架构角度度量：浮点型操作密度、整数型操作密度、指令中断、cache命中率、TLB命中；
- 从Spark系统执行时间和吞吐的角度度量：Job作业执行时间、Job吞吐量、Stage执行时间、Stage吞吐量、Task执行时间、Task吞吐量；
- 从Spark系统资源利用率的角度度量：CPU在指定时间段的利用率、内存在指定时间段的利用率、磁盘在指定时间段的利用率、网络带宽在指定时间段的利用率；
- 从扩展性的角度度量：数据量扩展、集群节点数据扩展（scale out）、单机性能扩展（scale up）。

#### 8.2.3 Benchmark的运用

- Hibench：由Intel开发的针对Hadoop的基准测试工具，开源的，用户可以到Github库中下载

- Berkeley BigDataBench：随着Spark的推出，由AMPLab开发的一套大数据基准测试工具，官网介绍

- Hadoop GridMix：Hadoop自带的Benchmark，作为Hadoop自带的测试工具使用方便、负载经典，应用广泛

- Bigbench：由Teradata、多伦多大学、InfoSizing、Oracle开发，其设计思想和利用扩展具有研究价值，可以参阅论文Bigbench:Towards an industry standard benchmark for big data analytics。

- BigDataBenchmark：由中科院研发，官方介绍

- TPC-DS：广泛应用于SQL on Hadoop的产品评测

- 其他的Benchmark：Malstone、Cloud Harmony、YCSB、SWIM、LinkBench、DFSIO、Hive performance Benchmark(Pavlo)等等

### 8.2.4 Benchmark和baseline比较

Benchmark和baseline都有性能比较的意思。

先看看字典定义。

> **benchmark**：N-COUNT A benchmark is something whose quality or quantity is known and which can therefore be used as a standard with which other things can be compared.

通俗的讲，一个算法之所以被称为benchmark，是因为它的性能已经被广泛研究，人们对它性能的表现形式、测量方法都非常熟悉，因此可以作为标准方法来衡量其他方法的好坏。

这里需要区别state-of-the-art（SOTA），能够称为SOTA的算法表明其性能在当前属于最佳性能。如果一个新算法以SOTA作为benchmark，这当然是最好的了，但如果比不过SOTA，能比benchmark要好，且方法有一定创新，也是可以发表的。

> **baseline**：N-COUNT A baseline is a value or starting point on a scale with which other values can be compared.

通俗的讲，一个算法被称为baseline，基本上表示比这个算法性能还差的基本上不能接受的，除非方法上有革命性的创新点，而且还有巨大的改进空间和超越benchmark的潜力，只是因为是发展初期而性能有限。所以baseline有一个自带的含义就是“性能起点”。这里还需要指出其另一个应用语境，就是在算法优化过程中，一般version1.0是作为baseline的，即这是你的算法能达到的一个基本性能，在算法继续优化和调参数的过程中，你的目标是比这个性能更好，因此需要在这个base line的基础上往上跳。

简言之，

- benchmark一般是和同行中比较牛的算法比较，比牛算法还好，那你可以考虑发好一点的会议/期刊；

- baseline一般是自己算法优化和调参过程中自己和自己比较，目标是越来越好，当性能超过benchmark时，可以发表了，当性能甚至超过SOTA时，恭喜你，考虑投顶会顶刊啦。

### 8.2.4 计算机视觉典型问题的基准

