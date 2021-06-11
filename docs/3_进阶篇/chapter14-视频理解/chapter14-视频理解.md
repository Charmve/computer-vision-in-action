<p align="left">
  <a href="https://github.com/Charmve"><img src="https://img.shields.io/badge/GitHub-@Charmve-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
  <a href="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aTmRoV05pYjNJUkIzZk5ldWVGZEQ4YnZ4cXlzbXRtRktUTGdFSXZOMUdnTHhDNXV0Y1VBZVJ0T0lJa0hTZTVnVGowamVtZUVOQTJJMHhiU0xjQ3VrVVEvNjQw?x-oss-process=image/format,png" target="_blank" ><img src="https://img.shields.io/badge/公众号-@迈微AI研习社-000000.svg?style=flat-square&amp;logo=WeChat" alt="微信公众号"/></a>
  <a href="https://www.zhihu.com/people/MaiweiE-com" target="_blank" ><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@Charmve-000000.svg?style=flat-square&amp;logo=Zhihu" alt="知乎"/></a>
  <a href="https://space.bilibili.com/62079686" target="_blank"><img src="https://img.shields.io/badge/B站-@Charmve-000000.svg?style=flat-square&amp;logo=Bilibili" alt="B站"/></a>
  <a href="https://blog.csdn.net/Charmve" target="_blank"><img src="https://img.shields.io/badge/CSDN-@Charmve-000000.svg?style=flat-square&amp;logo=CSDN" alt="CSDN"/></a>
</p>

# 第 14 章 视频理解

作者: 张伟 (Charmve)

日期: 2021/06/06

- 第 14 章 [视频理解](https://charmve.github.io/computer-vision-in-action/#/chapter1/chapter1)
    - 14.1 [概述](#141-概述)
    - 14.2 [视频理解场景中的主要问题](#142-视频理解场景中的主要问题)
    - 14.3 [常用数据集](#143-常用数据集)
    - 14.4 [主流方法与模型架构](#144-主流方法与模型架构)
    - 14.5 [指标 METRICS](#145-指标-metrics)
    - 14.6 [可能的未来方向](#146-可能的未来方向)
    - [小结](#小结)
    - [参考文献](#参考文献)

----

# 第 14 章 视频理解

许多人认为，深度学习在图像理解问题上的成功可以复制到视频理解领域。但是，视频动作问题和提出的深度学习解决方案的范围比其2D图像同级的范围更广，种类更多。 查找，识别和预测动作是视频行为理解中最重要的任务。

相比图像，视频多了一维时序信息。如何利用好视频中的时序信息是研究这类方法的关键。本文简要回顾视频理解方面的近年进展，并对未来可能的研究方向作一展望。

## 14.1 概述

如图 14.1 所示，动作理解包括动作问题、视频动作数据集、数据准备技术、深度学习模型和评估指标。 这些理解过程涉及计算机视觉和计算性能、数据多样性、可转移性、模型鲁棒性和可理解性的监督学习原则。

在之后章节将会详细阐述各个部分。

![image](https://user-images.githubusercontent.com/29084184/121520421-77f3d000-ca25-11eb-98d4-d914cacf36ca.png)

图14.1 动作理解流程和基本原理概述

## 14.2 视频理解场景中的主要问题

如图 14.2 所示，我们将主要的动作理解问题组织到重叠的分类和搜索箱中。 分类问题涉及按动作类标记视频。 搜索问题涉及在时间或时空上寻找动作实例。

![image](https://user-images.githubusercontent.com/29084184/121522190-71665800-ca27-11eb-9332-1cc4011db08f.png)

图14.2 动作理解问题分类法

以下是对动作理解问题的各种定义：

- **动作识别** (action recognition, AR) 是根据输入中发生的动作对完整输入（整个视频或指定片段）进行分类的过程。 如果动作实例跨越输入的整个长度，则该问题称为**修剪动作识别**。 如果动作实例没有跨越整个输入，则该问题称为**未修剪动作识别**。 未修剪的动作识别通常更具挑战性，因为模型需要完成动作分类任务，同时忽略输入的非动作背景部分。

![image](https://user-images.githubusercontent.com/29084184/121521613-de2d2280-ca26-11eb-8726-68790df4e1a9.png)

图14.3 动作理解问题：动作识别（action recognition, AR）、动作预测（action prediction, AP）、时序动作建议（temporal action proposal, TAP）、时序动作定位/检测（temporal action localization/detection, TAL/D）、时空动作建议（spatiotemporal action proposal, SAP）和时空动作定位/检测（spatiotemporal action localization/detection, SAL/ D)。 视频被描绘为一个 3D 体积，其中 𝑁 帧沿着时间维度密集堆叠。

- **动作预测** (action prediction， AP) 是根据尚未观察到的动作对不完整输入进行分类的过程。 一个子问题是动作预期 (action anticipation, AA)，其中动作的任何部分都没有被观察到，
分类完全基于观察到的上下文线索。 另一个是早期动作预测 (early action prediction, EAP)，其中已经观察到动作实例的一部分，但不是全部。动作识别 和 动作预测 都是分类问题，但动作预测通常需要一个带有时间注释的数据集，
以便在“动作前（before-action）”段和“动作中（during-action）”段之间有明确的分隔符用于动作预测或“开始动作（start-action）” 和用于早期动作预测的“最终动作（end-action）”。

- **时序动作建议** (Temporal Action Proposal, TAP) 是通过指示每个动作实例的开始和结束标记，将输入视频划分为动作和不动作段（连续帧系列）的过程。 时间动作定位/检测 (temporal action localization/detection, TAL/D) 是创建时间动作建议并对每个动作进行分类的过程。

> 由于在编写此章节时，还未找到合适的翻译，姑且全文先按照这样的中文翻译，之后找到更好的在更换。Temporal Action Proposal -> 时序动作建议, Proposal -> 建议。

- **时空动作建议** (Spatiotemporal Action Proposal, SAP) 是在动作和不动作区域之间通过空间（边界框）和时间（每帧或片段的开始和结束标记）对输入视频进行分区的过程。 如果将链接策略应用于跨多个帧的边界框，则在空间和时间维度上受限的动作区域通常称为管或小管。 时空动作定位/检测 (SAL/D) 是创建时空动作建议并对每个帧的边界框（或动作管）进行分类的过程
应用链接策略）。


## 14.3 常用数据集

视频分类主要有两种数据集，剪辑过(trimmed)的视频和未经剪辑的视频。剪辑的视频中包含一段明确的动作，时间较短标记唯一，而未剪辑的视频还包含了很多无用信息。如果直接对未剪辑的视频进行处理是未来的一大研究方向。

- HMDB-51：6,766视频，51类。剪辑的视频，每个视频不超过10秒。内容包括人面部、肢体、和物体交互的动作等。
- UCF-101：13,320视频，101类，共27小时。剪辑的视频，每个视频不超过10秒。内容包含化妆刷牙、爬行、理发、弹奏乐器、体育运动等。
- Charades：9.848视频(7,985训练，1,863测试)，157类。未剪辑的视频，每个视频大约30秒。每个视频有多个标记，以及每个动作的开始和结束时间。
- Sports-1M：1,100,000视频(70%训练、20%验证、10%测试)，487类，内容包含各种体育运动。
- ActivityNet (v1.3)：19,994视频(10,024训练，4,926验证，5,044测试)，200类，共700小时。内容包括饮食、运动、家庭活动等。
- Kinetics：246k训练视频，20k验证视频，400类。

相比图像分类，视频的类别/动作数目要少很多，而且常常有一定歧义，例如take和put要和后面名词结合才会有具体含义(如take medication, take shoes, take off shoes)。Sigurdsson等人[]发现人类对这些动词也容易感到混淆。另外，视频中动作开始和结束的时间也不够明确。

下表1展示包括数据集名称、出版年份、截至 2020 年 8 月在 Google Scholar 上的引用、动作类的数量、动作实例的数量、参与者：
人类 (H) 和/或非人类 (N)、注释：动作类 ( C)、时间标记 (T)、时空边界框/掩码 (S) 和主题/目的。 

表 1. 30 个具有历史影响力、当前最先进和新兴的视频动作数据集基准。

![image](https://user-images.githubusercontent.com/29084184/121525275-b6d85480-ca2a-11eb-9491-66cb531d6845.png)

一些竞赛引入了最先进的数据集、优化模型和标准化指标THUMOS Challenges 通过 2013 年国际计算机视觉会议 (ICCV)、2014 年欧洲计算机视觉会议 (ECCV) 和 2015 年计算机视觉和模式识别会议 (CVPR) 进行, 这些主要集中在 AR 和 TAL/D 任务上。 
ActivityNet 大规模活动识别挑战赛 (ActivityNet Large Scale Activity Recognition Challenges ) 于 2016 年至 2020 年由 CVPR 举行，并逐渐扩展到包括修剪过的 AR、未修剪过的 AR、TAP、TAL/D 和 SAL/D 比赛的范围。 
其他挑战已模仿 THUMOS 和 ActivityNet，例如 2019 年在 ICCV 举行的多模态视频分析研讨会和时刻挑战1 (Moments in Time Challenge1)，如下表2所示。 

表 2. 2013-2020 年突出的视频动作理解挑战赛

![image](https://user-images.githubusercontent.com/29084184/121525569-13d40a80-ca2b-11eb-8bce-41dbaf0e44dd.png)

## 14.4 主流方法与模型架构

### 14.4.1 经典方法

Wang Heng 等人[2][3] 提出DT和iDT方法，如下图14.4所示。DT利用光流得到视频中的运动轨迹，再沿着轨迹提取特征。iDT对相机运动进行了补偿，同时由于人的运动比较显著，iDT用额外的检测器检测人，以去除人对相邻帧之间投影矩阵估计的影响。这是深度学习方法成熟之前效果最好的经典方法，该方法的弊端是特征维度高(特征比原始视频还要大)、速度慢。实践中，早期的深度学习方法在和iDT结合之后仍能取得一定的效果提升，现在深度学习方法的性能已较iDT有大幅提升，因此iDT渐渐淡出视线。

![image](https://user-images.githubusercontent.com/29084184/121620923-bf1da780-ca9d-11eb-8418-8a584f525512.png)

图14.4 提取和表征密集轨迹的方法。 左特征点在每个空间尺度的网格上密集采样; 中间跟踪是通过密集光流场中的中值滤波在 L 帧的相应空间尺度上进行的; 右轨迹形状由相对点坐标表示，描述符（HOG，HOF，MBH）沿轨迹在 N×N 个像素邻域内计算，分为 $nσ×nσ×nτ$ 个单元.

### 14.4.2 前沿模型架构

随着深度学习、计算性能等各方面的提升，该方向吸引了大量学者进行研究，在这里视频行为分类问题整理了目前（2021.06）前沿的模型架构。由于该领域的快速发展性质，大家可在 PaperWithCode 上及时查看最新研究成果的更新。


#### 14.4.2.1 行为识别模型

Action Recognition Models，

如图 14.5 所示，我们将 AR 架构大致分为单流架构、双流架构、时间分割架构和两阶段学习几个不同复杂程度的类别。

- 第一类是**单流架构**，它从视频中采样或提取一个 2D 或 3D 输入特征，并将其馈送到 CNN，CNN 的输出是模型的预测。 虽然在某些任务上出奇地有效，但单流方法通常缺乏时间分辨率来充分执行 AR，而无需应用第 7 章节中讨论的最先进的混合模块。

![image](https://user-images.githubusercontent.com/29084184/121529829-5c8dc280-ca2f-11eb-9662-de3bc375f5f1.png)

图14.5  动作识别模型示例。 RGB 和 Motion Single-Stream 架构在一个采样特征上训练 2D、3D 或混合 CNN。 双流架构融合 RGB 和 Motion 流。时间分割架构将视频分成多个片段，在单流或多流架构上处理每个片段，并融合输出。 两阶段架构使用时间分割来提取特征向量并将其输入卷积或循环网络。

- 第二类是**双流架构**，一个流用于 RGB 学习，一个流用于运动特征学习 [4, 5]。然而，计算光流或其他手工制作的特征在计算上是昂贵的。因此，最近的几个模型使用“隐藏”运动流，
其中运动表示是学习而不是手动确定的。这些包括 MotionNet [326]，其操作类似于标准的双流方法，以及 MARS [6] 和 D3D [7]，它们在流之间执行中间融合。 Feichetenhofer 等人[8] 
探索了流之间的门控技术, 虽然这些模型通常在计算上受限于两个流，但可能有更多流用于其他模态 [9, 10]。

- 第三类是**时间分割架构**，由单流、双流或多流构建而成，可解决动作的长期依赖性。时间段网络 (TSN) 方法 [11, 12] 将输入视频划分为 𝑁 段，从这些段中采样，并通过平均段级输出来创建视频级预测。每个段流之间共享模型权重。 T-C3D [13]、TRN [14]、ECO [15] 和 SlowFast [58] 通过执行多分辨率分割和/或融合建立在时间分割的基础上。

- 第四类是**两阶段学习**，在我们的 AR 方法分类中的最高复杂度。第一阶段使用时间分割方法来提取段嵌入的特征向量，第二阶段对这些特征进行训练。这些包括 3D 融合和 CNN+LSTM 方法。


#### 14.4.2.2 行为预测模型

Action Prediction Models.

![image](https://user-images.githubusercontent.com/29084184/121531810-57ca0e00-ca31-11eb-9e9a-1161d671e4d9.png)

图 14.6 动作预测模型示例。 生成模型创建用于预测的未来时间步长的表示（通常通过编码器-解码器方案）。 非生成模型是一个广泛的类别，适用于直接从输入的观察部分创建预测的模型。

Rasouli (2020)指出，循环技术在这些方法中占主导地位。我们将这些高度多样化的动作预测模型分为生成式或非生成式系列，如图14.6所示。

- **生成式架构**产生不确定特征（"future" features ），然后对这些预测进行分类，这通常采用编码器-解码器方案的形式。例子包括 RED [69]，它使用强化学习模块来改进编码器 - 解码器 IRL [307]，
它使用 C2D 逆强化学习策略来预测未来帧，Conv3D [82]，它使用 C3D 生成看不见的特征预测，RGN [316] 在训练期间使用带有卡尔
曼滤波器的递归生成和预测方案，以及 RU-LSTM [65, 66]，它使用具有模态注意力的多模态滚动-展开编码器-解码器。


- **非生成式架构**是所有其他方法的广泛组合。这些直接从观察到的特征创建预测。示例包括 F-RNN-EL [104]，它使用指数损失将多模态
CNN+LSTM 融合策略偏向最近的预测，MS-LSTM [215] 使用两个 LSTM 阶段进行动作感知和上下文-感知学习，MM-LSTM [7] 将 MS-LSTM 
扩展到任意多种模态，FN [46] 使用三阶段 LSTM 方法，TP-LSTM [275] 使用时间金字塔学习结构。

本节中的许多示例是为动作预期（当尚未观察到动作的任何部分时）开发的，但它们也适用于早期动作识别（当已观察到动作的一部分时）。
此外，如果第 14.4.2.1 节中描述的动作识别模型能够从提供的部分和视频上下文中获得足够的语义含义，则它们可能适用于一些早期的动作识别任务。 

#### 14.4.2.3 时序动作建议模型

Temporal Action Proposal Models

如图 14.7 所示，TAP 方法可以分为三个类：自顶向下、自下而上和混合架构。

![image](https://user-images.githubusercontent.com/29084184/121532665-243bb380-ca32-11eb-9dc9-a8ee8c3b202f.png)

图 14.7 时间行动建议模型示例。 自顶向下模型使用滑动窗口方法来创建段级建议；自下而上模型使用具有分组策略的框架或短段级别的动作得分预测来生成建议； 
混合模型并行使用自上而下和自下而上的策略。

- 第一类是**自顶向下**架构，它由使用滑动窗口来导出段级建议的模型组成。示例包括使用 CNN 特征提取器和循环网络的 DAP [52] 和 SST [20]、
使用多尺度滑动窗口的 S-CNN [227] 和使用多尺度池化策略的 TURN TAP [68]。

- 第二类是**自下而上**架构，它使用双流帧级或短段级提取的特征来导出“动作性”置信度预测。然后将各种分组策略应用于这些密集预测以创建完整的建议。
例子包括 TAG [319]，它使用洪泛算法将这些转换为多尺度分组，BSN [157] 和 BMM [155]，它们使用额外的“开始”和“结束”特征来实现不同的提案生
成和提案评估技术，以及RecapNet [276] 使用残差因果网络而不是通用 1D CNN 来计算置信度预测。 R-C3D [293] 和 TAL-Net [27] 使用基于区域的
方法将图像中的 2D 对象提议适应视频中的一维动作提议。许多自下而上的架构需要对输出进行非最大抑制 (NMS) 以减少冗余提议的权重。

- 第三类是**混合架构**，它结合了自顶向下和自底向上的方法， 这些通常会并行创建分段建议和行动性分数，然后使用行动性来改进建议。 
示例包括 CDC [226]、CTAP [67]、MGG [164] 和 DPP [144]。


#### 14.4.2.4 时序动作定位/检测模型

Temporal Action Localization/Detection Models

如图 14.8  所示，有两个主要的 TAL/D 方法系列。该分类法是由 Xia 等人引入的。 (2020) [290]。

![image](https://user-images.githubusercontent.com/29084184/121533539-f73bd080-ca32-11eb-8913-c232a4b42db8.png)

图 14.8 时间动作定位/检测模型示例。 一级架构一起进行动作提议和分类，而两级架构先创建提议，然后使用动作识别模型对每个提议进行分类。


第一类是**两阶段**架构，其中第一阶段创建提案，第二阶段对它们进行分类。因此，要创建两阶段架构，您可以将第 14.4.2.3 节中描述的
任何 TAP 模型与第 14.4.2.1 节中描述的 AR 模型配对。值得注意的是，几乎所有探索 TAP 方法的论文也将他们的工作扩展到了 TAL/D。

第二类是**一阶段**的架构，其中提议和分类一起发生。 示例包括 SSAD [156]，它创建一个片段级动作评分序列，一维 CNN 从中提取多尺
度检测，SS-TAD [228] 其中并行循环存储单元创建提议和分类，Decouple-SSAD [98] 建立在 SSAD 上，具有三流解耦锚网络，GTAN [165] 
使用多尺度高斯内核，双流 SSD [199] 融合带有 OF 检测的 RGB 检测，以及在分类之前完成边界细化的 RBC [115]。

#### 14.4.2.5 时空动作定位/检测模型

如图 14.9 所示，有两大类最先进的 SAL/D 方法，帧级（区域）提议模型和段级（管）提议模型。

![image](https://user-images.githubusercontent.com/29084184/121534145-8812ac00-ca33-11eb-9b0b-0de6566d1d3a.png)

图 14.9 时空动作定位/检测模型示例。 帧级（区域）提议模型将帧级检测链接在一起，而段级（管）提议模型为短段创建小的“小管（tubelets）”并将小管连接到更长的管中。

第一类是**帧级（区域）提议架构**，它使用各种区域提议算法（例如 R-CNN [77]、Fast R-CNN [76]、Faster R-CNN [207]、早期+晚期融合 Faster R-CNN [300]）从帧中导出边界框，
然后应用帧链接算法。 示例包括 MR-TS [192]、CPLA [297]、ROAD [232]、AVA I3D [81]、RTPR [151] 和 PntMatch [300]。

第二类是**段级（管）提议架构**，它使用各种方法来创建段级临时小管或“小管”，然后使用管链接算法， 这些模型的示例包括 T-CNN [94]、ACT-检测器 [117] 和 STEP [296]。


一些SOTA的模型并不适合这两个系列中的任何一个，但值得注意：

- 张等人 (2019) [312] 使用跟踪网络和图卷积网络来推导人物对象检测；
- VATX [74] 使用多头、多层转换器增强了 I3D 方法；
- STAGE [251] 介绍了一种时间图注意方法


## 14.5 指标 METRICS

选择正确的指标对于正确评估模型至关重要。 在本节中，我们定义常用指标并指出其用法示例。 
我们不会涵盖二元分类指标，因为我们编目的动作数据集绝大多数都有两个以上的类。 请注意，任何时候我们提到准确度值时，
误差值都可以很容易地计算为 $𝑒 = 1 − 𝑎$。 为了更方便阅读，我们在指标中使用以下符号：

• $𝑋 = {𝑥_{(1)}, ..., 𝑥_{(𝑛)}}$: 𝑛 输入视频集
• $𝑌 = {𝑦_{(1)}, ..., 𝑦_{(𝑛)}}$: 输入视频的一组 𝑛 基本事实注释
• $𝑀 : 𝑋 → 𝑌b$: 将输入视频映射到预测注释的函数（又名模型）
• $𝑌b = {𝑦_{(1)}, ...,𝑦_{(𝑛)}}$: 𝑛 模型输出集
• $𝐶 = {1, ...,𝑚}$: 𝑚动作类一组
• $𝑇𝑃_𝑗: N → {0, 1}$: 一个函数将列表 $𝐿_𝑗$ 中的等级映射到 1，如果该等级的项目是真阳性，否则为 0

其中一些指标还使用联合交叉（IoU）的形式，这是衡量两个区域相似性的一种方法。 图 14.10 描绘了空间 IoU、时间 IoU 和时空 IoU。

![image](https://user-images.githubusercontent.com/29084184/121528364-f8b6ca00-ca2d-11eb-83c2-27ceeb1abdde.png)

图14.10 联合的交叉类型（IoU）说明：空间、时间和时空。 IoU 也称为 Jaccard 指数或 Jaccard 相似系数


## 14.6 可能的未来方向

- **利用多示例学习进行视频分析**。未剪辑视频中有很多无关内容，并非视频中所有的帧都对应于该视频标记，这符号多示例学习的设定。虽然Zhu等人在CVPR'16和Kar等人在CVPR'17的工作中对这方面已有一些探索，但仍有后续精进的空间。
- **精度与效率**。Two-stream和3D卷积的方法相比，大致来说前者的效果更好，但前者需要逐帧图像前馈网络，而后者一次可以处理多帧，因此前者效率不如后者，尤其是预先计算并保存光流是一个很繁重的负担。如何能同时利用两者的优点是未来一个可能的研究方向，Feichtenhofer等人在CVPR'16已有初步的工作。LSTM能捕获的长距离依赖程度有限，并且更难训练，速度也更慢，因此ConvLSTM的方法在视频分析中用的不多。
- **资源受限下的视频分析**。相比图像数据，处理视频数据需要更大的计算和存储资源。现实应用中很多是资源受限的，如何在这种场景下进行视频分析是一大挑战。将视频解压为能输入网络的一帧帧图像也需要不小的资源开销，Wu等人在CVPR'18提出直接利用原始视频输入，并利用视频压缩编码中的运动信息。
- **更大、更通用数据集**。哪种方法更好和用什么数据集(解决什么任务)有很大关系。如果视频本身就比较静止，或者单帧图像已经包含了足够的信息，那么用逐帧单独处理的策略已经可以取得很好的结果。
- **视频=图像+音频**。视频是一种多模态的数据形式，能否利用音频信息辅助视频分析呢。Aytar等人在NIPS'16的工作中利用图像辅助音频分析。


## 小结

在本章节中展示了封装在动作理解中的一系列问题，列出了用作基准和预训练源的数据集，描述了数据准备步骤和策略，组织了深度学习模型构建块和最先进的模型系列。 我们希望本章节澄清了术语，扩展了您对这些问题的理解，并激发了您在计算机视觉和深度学习的交叉领域在这个快速发展的领域中进行研究的灵感。 
本文还通过通用数据集、模型构建块和指标展示了这些动作理解问题空间之间的异同。 为此，我们还希望这可以促进喷发出交叉学科和前沿研究与应用思维的火花，并在最后给出了作者对视频理解方向浅薄的未来展望。

## 参考文献

[1] G. A. Sigurdsson, et al. What actions are needed for understanding human actions in videos? ICCV'17.

[2] H. Wang, et al. Dense trajectories and motion boundary descriptors for action recognition. IJCV'13.

[3] H. Wang and C. Schmid. Action recognition with improved trajectories. ICCV'13.

[4] Joao Carreira and Andrew Zisserman. 2017. Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. arXiv:1705.07750 [cs.CV]

[5] Karen Simonyan and Andrew Zisserman. 2014. Two-Stream Convolutional Networks for Action Recognition in Videos. In Advances in Neural Information Processing Systems 27, Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger (Eds.). Curran Associates, Inc., 568–576. http://papers.nips.cc/paper/5353-twostream-convolutional-networks-for-action-recognition-in-videos.pdf

[6] Nieves Crasto, Philippe Weinzaepfel, Karteek Alahari, and Cordelia Schmid. 2019. MARS: Motion-Augmented RGB
Stream for Action Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR).

[7] Jonathan Stroud, David Ross, Chen Sun, Jia Deng, and Rahul Sukthankar. 2020. D3D: Distilled 3D Networks for Video Action Recognition. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).

[8] Christoph Feichtenhofer, Axel Pinz, and Richard P. Wildes. 2017. Spatiotemporal Multiplier Networks for Video
Action Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

[9] Liangliang Wang, Lianzheng Ge, Ruifeng Li, and Yajun Fang. 2017. Three-stream CNNs for action recognition. Pattern
Recognition Letters 92 (2017), 33 – 40. https://doi.org/10.1016/j.patrec.2017.04.004

[10] Le Wang, Jinliang Zang, Qilin Zhang, Zhenxing Niu, Gang Hua, and Nanning Zheng. 2018. Action Recognition by an Attention-Aware Temporal Weighted Convolutional Neural Network. Sensors 18, 7 (Jun 2018), 1979. https://doi.org/10.3390/s18071979

[11] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. 2016. Temporal
Segment Networks: Towards Good Practices for Deep Action Recognition. In Computer Vision – ECCV 2016, Bastian
Leibe, Jiri Matas, Nicu Sebe, and Max Welling (Eds.). Springer International Publishing, Cham, 20–36.

[12] L. Wang, Y. Xiong, Z. Wang, Y. Qiao, D. Lin, X. Tang, and L. Van Gool. 2019. Temporal Segment Networks for Action
Recognition in Videos. IEEE Transactions on Pattern Analysis and Machine Intelligence 41, 11 (2019), 2740–2755.

[13] Kun Liu, Wu Liu, Chuang Gan, Mingkui Tan, and Huadong Ma. 2018. T-c3d: Temporal convolutional 3d network for
real-time action recognition. In Thirty-second AAAI conference on artificial intelligence.

[14] Bolei Zhou, Alex Andonian, Aude Oliva, and Antonio Torralba. 2018. Temporal Relational Reasoning in Videos. In
Proceedings of the European Conference on Computer Vision (ECCV).

[15] Mohammadreza Zolfaghari, Kamaljeet Singh, and Thomas Brox. 2018. ECO: Efficient Convolutional Network for
Online Video Understanding. In Proceedings of the European Conference on Computer Vision (ECCV)

[16] MATTHEW HUTCHINSON and VIJAY GADEPALLY. Video Action Understanding: A Tutorial. https://arxiv.org/pdf/2010.06647.pdf


